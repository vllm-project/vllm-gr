# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""ModelRunner V2 execution hooks for the persistent GR stage runner.

The stage-runner core (``gpu_beam_stage_runner.GPUBeamStageRunner``) is
runner-agnostic; only the native execution surface differs.  This module maps
that surface onto the V2 ModelRunner
(``vllm.v1.worker.gpu.model_runner.GPUModelRunner``):

- forward: ``set_forward_context`` + the torch.compile'd ``runner.model``
  (V2 has no ``_model_forward`` helper; the same route is proven by the V2
  beam-decode graph capture, ``gpu_model_runner_v2_resources``).
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

from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeCaptureUnsupported
from vllm_gr.v1.worker.gpu_beam_stage_runner import (
    BeamRequestBinding,
    GPUBeamStageRunner,
    _hidden_logits,
    should_replay_decode_graph,
)
from vllm_gr.v1.worker.gpu_stage_hooks import CudaStageHooks

logger = init_logger(__name__)


class CudaStageHooksV2(CudaStageHooks):
    """V2 native operations and graph policy for the shared CUDA stage core."""

    persistent_graph_install: ClassVar[bool] = False
    # Both runners allocate and use the transposed head only when slicing is requested.
    supports_transposed_head: ClassVar[bool] = True
    specialize_constraint_steps: ClassVar[bool] = True

    def initialize_inputs(self, owner: GPUBeamStageRunner, rows: int) -> None:
        from vllm_gr.v1.worker.gpu_model_runner_v2_resources import V2InputBuffers

        worker, beam = self.worker, owner.beam
        block_size = worker.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        inputs = getattr(worker, "_gr_input_buffers", None)
        if inputs is None:
            inputs = V2InputBuffers(
                beam.state.beam_width,
                worker.model_config.max_model_len,
                block_size,
                beam.state.device,
                max_batch=beam.workspace.max_batch,
            )
            worker._gr_input_buffers = inputs
        if (
            inputs.width != beam.state.beam_width
            or inputs.max_batch != beam.workspace.max_batch
            or inputs.max_model_len != worker.model_config.max_model_len
            or inputs.block_size != block_size
        ):
            raise RuntimeError("GR fixed input capacity changed during the worker lifetime")
        owner.inputs = inputs
        owner.input_ids = inputs.input_ids[:rows]
        owner.positions = inputs.positions[:rows]
        owner.slots = inputs.slots[:rows]
        owner.execution_mask = inputs.execution_mask[:rows]

    @staticmethod
    def required_constraint_rows(batch_size: int, beam_width: int) -> int:
        return batch_size * beam_width

    def prepare_new_adapter(self, view: BeamBatchExecutionView) -> None:
        # Every V2 adapter uses the same Worker storage. Keep captured graphs.
        pass

    def decode_graph(self, batch_size: int, beam_width: int) -> Any:
        from vllm_gr.v1.worker.beam_decode_graph import get_beam_decode_graph
        from vllm_gr.v1.worker.gpu_model_runner_v2_resources import V2BeamDecodeGraph

        return get_beam_decode_graph(
            self.worker,
            batch_size=batch_size,
            beam_width=beam_width,
            graph_type=V2BeamDecodeGraph,
        )

    def decode_logits(self, owner: GPUBeamStageRunner) -> torch.Tensor | None:
        assert owner.inputs is not None
        from vllm_gr.v1.worker.model_runner_common import _get_gr_config

        worker = self.worker
        batch_size = len(owner.bindings)
        cfg = _get_gr_config(worker)
        bucket = getattr(getattr(owner, "metadata", None), "prefix_length_bucket", 0)
        geometry = (
            (batch_size, owner.inputs.width, bucket) if bucket else (batch_size, owner.inputs.width)
        )
        if geometry in getattr(worker, "_gr_runtime_decode_capture_failures", {}):
            owner.graph = None
            return owner._forward()
        enabled = (
            not worker.model_config.enforce_eager
            and cfg.beam_graph_enabled
            and batch_size in cfg.beam_online_batch_sizes
            and not getattr(worker, "_gr_async_decode_graph_capture_failed", False)
            and batch_size not in getattr(worker, "_gr_decode_capture_failures", {})
        )
        owner.graph = self.decode_graph(batch_size, owner.inputs.width) if enabled else None
        if owner.graph is not None and bucket:
            owner.graph.select(batch_size, owner.inputs.width, bucket)
        # Dynamic sessions own per-width buffers. Rebuild after schedule/batch
        # changes, including the first fixed-width session following them.
        if (
            owner.graph is not None
            and not owner.graph.ready
            and (
                bucket
                or owner.dynamic is not None
                or getattr(worker, "_gr_dynamic_graphs_discarded", False)
            )
        ):
            try:
                owner.graph.capture(owner._forward_hidden, persistent_install=False)
            except BeamDecodeCaptureUnsupported as error:
                # Capture has restored the model wrapper. Drain its side stream
                # before reusing live request buffers; unhealthy devices must fail.
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Decode capture did not leave capture mode") from error
                torch.accelerator.synchronize(worker.device)
                owner.graph.discard_selected()
                failures = getattr(worker, "_gr_runtime_decode_capture_failures", None)
                if failures is None:
                    failures = worker._gr_runtime_decode_capture_failures = {}
                failures[geometry] = str(error)
                owner.graph = None
                logger.warning(
                    "GR runtime Decode capture unsupported; using eager for B=%d W=%d: %s",
                    batch_size,
                    owner.inputs.width,
                    error,
                )
                return owner._forward()
        if should_replay_decode_graph_v2(worker, owner.graph, batch_size=batch_size):
            return _hidden_logits(owner, owner.graph.replay())
        return owner._forward()

    def consume_retirement_state(
        self, scheduler_output: Any, bindings: tuple[BeamRequestBinding, ...]
    ) -> None:
        self.consume_native_state(
            replace(
                scheduler_output,
                finished_req_ids=scheduler_output.finished_req_ids
                | {binding.native_request_id for binding in bindings},
            )
        )

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
        self, original: Callable[..., Any], scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
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
        # GR consumes this forward instead of the native sampler. Release the
        # pending native state so a later sampler cannot consume stale logits.
        worker.execute_model_state = None
        # Same contract as the native V2 sampler
        # (``GPUModelRunner.sample``: ``hidden_states[input_batch.logits_indices]``,
        # unconditional).  A Host-side bounds check on ``logits_indices`` would
        # force a device->host sync right after the Prefill forward — the
        # sampler's own indexing is the authoritative contract and needs none.
        batch = state.input_batch
        stages = getattr(scheduler_output, "gr_stage_metadata", None)
        if not stages or len(stages) != 1:
            raise ValueError("GR Prefill requires one logical batch's metadata")
        request_ids = next(iter(stages.values())).request_ids
        native_ids = batch.req_ids
        if (
            len(set(request_ids)) != len(request_ids)
            or len(set(native_ids)) != len(native_ids)
            or set(request_ids) != set(native_ids)
            or len(batch.cu_num_logits_np) != len(request_ids) + 1
            or any(b - a != 1 for a, b in zip(batch.cu_num_logits_np, batch.cu_num_logits_np[1:]))
        ):
            raise ValueError("GR Prefill requires exactly one logit row per batch member")
        indices = batch.logits_indices
        if tuple(native_ids) != tuple(request_ids):
            from vllm.utils.torch_utils import async_tensor_h2d

            native_rows = {request_id: row for row, request_id in enumerate(native_ids)}
            order = async_tensor_h2d(
                [native_rows[request_id] for request_id in request_ids],
                device=worker.device,
                dtype=torch.long,
            )
            indices = indices[order]
        # Restore only the sampled rows. Native request/KV order remains intact.
        sample_hidden = state.hidden_states[indices]
        logits = worker.model.compute_logits(sample_hidden)
        # GR bypasses native sample_tokens/postprocess, which normally advances
        # this GPU counter. update_requests only updates its CPU mirror; without
        # the device update a continuation gathers the previous chunk again.
        # Complete GR Prefill consumes its actual query length (not graph
        # padding), including after a Prefix Cache hit. Generated token history
        # remains owned by GR, so use the count-only native update.
        worker.postprocess_num_computed_tokens(batch)
        return logits

    def finish_native_output(self) -> None:
        """Finish output submission without repeating native Prefill updates."""
        self.worker.execute_model_state = None
        # Preserve the existing GR reset even if native V2 did not create this
        # V1-style field. This does not invoke a connector or clear GR buffers.
        self.worker.kv_connector_output = None

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


def should_replay_decode_graph_v2(worker: Any, graph: Any, *, batch_size: int) -> bool:
    """Unselected/refused shapes fall back; missing selected graphs remain fatal."""
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    cfg = _get_gr_config(worker)
    requested = not worker.model_config.enforce_eager and cfg.beam_graph_enabled
    if requested and batch_size not in cfg.beam_online_batch_sizes:
        if graph is not None:
            raise RuntimeError("Unconfigured Decode shape has a published graph")
        return False
    if requested and batch_size in getattr(worker, "_gr_decode_capture_failures", {}):
        if graph is not None:
            raise RuntimeError("Decode capture failed but a graph is still published")
        return False
    return should_replay_decode_graph(worker, graph)
