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
  ``update_requests`` → ``block_tables.apply_staged_writes``).  Under the
  persistent GR contract this runs only at the prefill→decode boundary and
  the retirement control frame; decode steps mint no new requests.
- decode block ids: read from ``GRStageMetadata.kv_blocks`` (published by the
  scheduler); V2 keeps no host-side request state with block ids.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_gr.v1.worker.gpu_beam_stage_runner import _StageExecutionHooks


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
        # Mirror the inline state machine at the top of V2 execute_model.
        # Under the persistent GR contract the scheduled request set is stable
        # (no per-step re-mint), so on decode steps this is a bookkeeping
        # no-op; it matters at the prefill→decode boundary (registering the
        # prompt's KV state) and on the retirement control frame (removing
        # the finished request).
        worker.finish_requests(scheduler_output)
        worker.free_states(scheduler_output)
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
