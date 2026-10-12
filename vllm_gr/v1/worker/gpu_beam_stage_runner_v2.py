# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelRunner V2 execution hooks for the persistent GR stage runner.

The stage-runner core (``gpu_beam_stage_runner.GPUBeamStageRunner``) is
runner-agnostic; only the native execution surface differs.  This module maps
that surface onto the V2 ModelRunner
(``vllm.v1.worker.gpu.model_runner.GPUModelRunner``):

- forward: ``set_forward_context`` + the torch.compile'd ``runner.model``
  (V2 has no ``_model_forward`` helper; the same route is proven by the V2
  beam-decode graph capture, ``gpu_model_runner_v2_graph``).
- prefill logits: native ``execute_model`` publishes ``ExecuteModelState``
  carrying ``hidden_states``; logits are computed from
  ``hidden_states[logits_indices]`` (the ``patched_sample_v2`` contract).
- CPU input fence: not needed — V2 builds a fresh ``InputBatch`` per step and
  never reuses pinned CPU input buffers across steps.
- native state consumption: the V2 per-step state machine
  (``finish_requests`` → ``free_states`` → ``add_requests`` →
  ``update_requests`` → ``block_tables.apply_staged_writes``), with
  ``add_requests`` called only for new requests. Persistent GR decode and
  retirement need no repeated native sampler buffer refresh.
- decode block ids: read from ``GRStageMetadata.kv_blocks`` (published by the
  scheduler); V2 keeps no host-side request state with block ids.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker.gpu_beam_stage_runner import _StageExecutionHooks


def _can_fuse_prefill_state_writes(worker: Any, scheduler_output: Any, request: Any) -> bool:
    """Check the fresh single-row write contract before native registration."""
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.states import RequestState

    states = worker.req_states
    blocks = worker.block_tables
    if (
        worker.device.type != "cuda"
        or worker.use_pp
        or worker.use_dcp
        or worker.dp_size != 1
        or worker.parallel_config.tensor_parallel_size != 1
        or worker.parallel_config.prefill_context_parallel_size != 1
        or worker.supports_mm_inputs
        or worker.is_encoder_decoder
        or type(worker.model_state) is not DefaultModelState
        or worker.model_state.rope_state is not None
        or type(states) is not RequestState
        or type(blocks) is not BlockTables
        or states.num_reqs != 0
        or request.num_computed_tokens != 0
        or scheduler_output.scheduled_cached_reqs.req_ids
        or scheduler_output.scheduled_spec_decode_tokens
        or scheduler_output.scheduled_encoder_inputs
        or scheduler_output.has_structured_output_requests
        or blocks.num_kv_cache_groups != 1
        or blocks.blocks_per_kv_block != [1]
        or len(request.block_ids) != 1
    ):
        return False
    num_tokens = len(request.prefill_token_ids)
    num_blocks = len(request.block_ids[0])
    if (
        num_tokens != len(request.prompt_token_ids)
        or num_tokens != scheduler_output.num_scheduled_tokens[request.req_id]
        or not 0 < num_tokens <= states.all_token_ids.gpu.shape[1]
        or not 0 < num_blocks <= blocks.block_tables[0].gpu.shape[1]
    ):
        return False
    # The fused flush must own every queued write. A pending update from any
    # other dispatch keeps the existing per-tensor flush path.
    return not any(
        tensor._staged_write_indices
        for tensor in (
            states.total_len,
            states.all_token_ids,
            states.num_computed_tokens,
            blocks.block_tables[0],
        )
    )


def _apply_fused_prefill_state_writes(worker: Any, req_index: int) -> None:
    """Flush native registration writes with one H2D and one ordered kernel."""
    from vllm.utils.torch_utils import async_tensor_h2d

    from vllm_gr.ops.cuda.gr_prefill_inputs import apply_gr_prefill_state_writes_v2

    states = worker.req_states
    block_table = worker.block_tables.block_tables[0]
    tokens = states.all_token_ids._staged_write_contents
    block_ids = block_table._staged_write_contents
    payload = [
        states.total_len._staged_write_contents[0],
        states.num_computed_tokens._staged_write_contents[0],
    ]
    payload.extend(tokens)
    payload.extend(block_ids)
    # Keep per-dispatch storage and the native pinned transfer helper. The
    # allocators retain the host copy until H2D completion and order device
    # reuse on this stream; no mutable host buffer is shared across requests.
    packed = async_tensor_h2d(payload, torch.int32, worker.device)
    apply_gr_prefill_state_writes_v2[((max(len(tokens), len(block_ids)) + 255) // 256,)](
        packed,
        states.total_len.gpu,
        states.all_token_ids.gpu,
        states.num_computed_tokens.gpu,
        block_table.gpu,
        req_index,
        len(tokens),
        len(block_ids),
        TOKEN_STRIDE=states.all_token_ids.gpu.stride(0),
        BLOCK_TABLE_STRIDE=block_table.gpu.stride(0),
        BLOCK=256,
    )
    # Clear only after successful submission. The native block-table flush
    # still follows update_requests and refreshes num_blocks, but must not
    # submit the same block IDs twice. CUDA errors propagate normally.
    for tensor in (
        states.total_len,
        states.all_token_ids,
        states.num_computed_tokens,
        block_table,
    ):
        tensor.clear_staged_writes()


def try_add_gr_prefill_request(worker: Any, scheduler_output: Any) -> bool:
    """Register an admitted GR Prefill without unused native sampler state.

    Keep the native request/model/KV lifecycle, including slot reuse. GR owns
    sampling after this forward, so native sampling parameters are neither
    initialized nor flushed. Decline before mutation for ordinary or mixed
    batches and for features requiring the native sampling path.
    """
    stages = getattr(scheduler_output, "gr_stage_metadata", None)
    if not stages or getattr(worker, "_current_scheduler_output", None) is not scheduler_output:
        return False
    new_reqs = scheduler_output.scheduled_new_reqs
    if len(stages) != 1 or len(new_reqs) != 1 or len(scheduler_output.num_scheduled_tokens) != 1:
        return False
    request = new_reqs[0]
    stage = stages.get(request.req_id)
    params = request.sampling_params
    if (
        stage is None
        or stage.kind is not GRStageKind.PREFILL
        or scheduler_output.num_scheduled_tokens.get(request.req_id, 0) <= 0
        or request.prompt_token_ids is None
        or request.prefill_token_ids is None
        or request.mm_features
        or request.lora_request is not None
        or worker.encoder_cache is not None
        or worker.lora_config is not None
        or worker.speculative_config is not None
        or worker.is_pooling_model
        or not worker.is_first_pp_rank
        or not worker.is_last_pp_rank
        or params is None
        or params.prompt_logprobs is not None
    ):
        return False

    fuse_writes = _can_fuse_prefill_state_writes(worker, scheduler_output, request)
    # Keep native CPU registration and slot lifecycle. The optional fused
    # flush writes the same native state; other GR Prefills retain individual
    # staged writes. Native sampler initialization remains unnecessary here.
    worker._remove_request(request.req_id)
    worker.req_states.add_request(
        req_id=request.req_id,
        prompt_len=len(request.prompt_token_ids),
        all_token_ids=request.prefill_token_ids,
        num_computed_tokens=request.num_computed_tokens,
    )
    req_index = worker.req_states.req_id_to_index[request.req_id]
    worker.model_state.add_request(req_index, request)
    worker.block_tables.append_block_ids(req_index, request.block_ids, overwrite=True)
    worker.lora_state.add_request(request.req_id, req_index, request.lora_request)
    # Reset per-slot flags even when the reused slot previously requested
    # prompt logprobs. This worker is not used by GR sampling itself.
    assert worker.prompt_logprobs_worker is not None
    worker.prompt_logprobs_worker.add_request(request.req_id, req_index, params)
    if fuse_writes:
        worker.req_states.prompt_len.copy_to_uva()
        worker.req_states.prefill_len.copy_to_uva()
        _apply_fused_prefill_state_writes(worker, req_index)
    else:
        worker.req_states.apply_staged_writes()
    worker.model_state.apply_staged_writes()
    return True


class V2StageExecutionHooks(_StageExecutionHooks):
    """_StageExecutionHooks bound to the V2 GPUModelRunner surface."""

    persistent_graph_install = False

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention: Any,
        slot_mappings: Any,
        capture: bool,
    ) -> torch.Tensor:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.cudagraph_dispatcher import BatchDescriptor

        worker = self.worker
        # The V2 model is torch.compile'd (no _model_forward helper).  Capture
        # runs under the shared BeamDecodeGraph, whose _BeamDecodeModel routes
        # a FULL runtime mode + the fixed-width BatchDescriptor into the graph
        # registration (the descriptor must match the startup capture width —
        # same contract as the V1 hooks); eager decode keeps the
        # metadata-rebuild path (NONE).
        with set_forward_context(
            attention,
            worker.vllm_config,
            num_tokens=input_ids.numel(),
            cudagraph_runtime_mode=CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=input_ids.numel(), uniform=True),
            slot_mapping=slot_mappings,
        ):
            return worker.model(
                input_ids=input_ids,
                positions=positions,
            )

    def prefill_logits(
        self, original: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        from vllm.v1.worker.gpu.input_batch import post_update_pool

        worker = self.worker
        from contextlib import nullcontext

        from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
            GRPrefillCudaGraphManager,
        )

        manager = getattr(worker, "cudagraph_manager", None)
        scope = (
            manager.prefill_scope(scheduler_output)
            if isinstance(manager, GRPrefillCudaGraphManager)
            else nullcontext()
        )
        with scope:
            out = original(worker, scheduler_output, *args, **kwargs)
        state = getattr(worker, "execute_model_state", None)
        if out is not None or state is None or state.hidden_states is None:
            raise RuntimeError(
                "native GR Prefill must publish hidden states through execute_model_state"
            )
        # Same contract as the native V2 sampler
        # (``GPUModelRunner.sample``: ``hidden_states[input_batch.logits_indices]``,
        # unconditional).  A Host-side bounds check on ``logits_indices`` would
        # force a device->host sync right after the Prefill forward — the
        # sampler's own indexing is the authoritative contract and needs none.
        sample_hidden = state.hidden_states[state.input_batch.logits_indices]
        logits = worker.model.compute_logits(sample_hidden)
        # GR bypasses native sample_tokens/postprocess, which normally advances
        # this GPU counter. update_requests only updates its CPU mirror; without
        # the device update a continuation gathers the previous chunk again.
        # Every prefill chunk consumes its actual query length (not graph
        # padding), even when it produces no Beam output. Generated token
        # history remains owned by GR, so use the count-only native update.
        post_update_pool(
            state.input_batch.idx_mapping,
            worker.req_states.num_computed_tokens.gpu,
            state.input_batch.query_start_loc,
        )
        return logits

    def prefill_stage_fence(self) -> None:
        # V2 prepares a fresh InputBatch per execute_model call; there are no
        # reused pinned CPU input buffers to fence against.
        return None

    def consume_native_state(self, scheduler_output: Any) -> None:
        worker = self.worker
        # Preserve native lifecycle and KV updates on every dispatch. Native
        # add_requests also refreshes sampler UVA buffers when its request
        # list is empty; GR sampling bypasses that sampler, so avoid those
        # copies on persistent decode and retirement frames.
        worker.finish_requests(scheduler_output)
        worker.free_states(scheduler_output)
        if scheduler_output.scheduled_new_reqs:
            worker.add_requests(scheduler_output)
        worker.update_requests(scheduler_output)
        worker.block_tables.apply_staged_writes()

    def decode_block_ids(self, session_id: str) -> list[list[int]]:
        # The scheduler publishes kv_blocks on every dispatch; V2 has no
        # host-side request state to consult.  Reaching here means the
        # metadata path was not used — a protocol error upstream.
        raise RuntimeError(
            "V2 stage execution requires GRStageMetadata.kv_blocks "
            f"(session {session_id} dispatched without scheduler block publication)"
        )


def make_v2_stage_hooks(worker: Any) -> V2StageExecutionHooks:
    """Construct the V2 execution hooks for one worker."""
    return V2StageExecutionHooks(worker)
