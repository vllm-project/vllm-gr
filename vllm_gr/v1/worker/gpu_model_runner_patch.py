# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU ModelRunner wrappers for grouped Beam execution."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
from vllm.distributed.ec_transfer import has_ec_transfer
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.logger import init_logger
from vllm.v1.utils import record_function_or_nullcontext

from vllm_gr.v1.beam.beam_decision import (
    _create_persistent_gr_state,
    maybe_run_worker_beam_decision,
)
from vllm_gr.v1.worker.constraint_hook import (
    apply_constrained_sample_override,
    consume_prelaunched_cuda_prefill,
    prepare_cuda_prefill_constraint,
    try_sample_with_constraint_backend,
)

# Shared device-neutral beam-graph helpers live in model_runner_common; they
# are re-exported here so callers/tests that historically reached them via
# this module keep working (and monkeypatching this alias still patches the
# GPU patch behavior, which resolves them through this module's globals).
from vllm_gr.v1.worker.model_runner_common import (
    _WORKER_BOOKKEEPING_PATCH_MARKER,
    _WORKER_DETERMINE_PATCH_MARKER,
    _WORKER_EXECUTE_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    _WORKER_SAMPLE_PATCH_MARKER,
    _WORKER_SAMPLE_TOKENS_PATCH_MARKER,
    _beam_collapse_bookkeeping,
)
from vllm_gr.v1.worker.model_runner_common import (  # noqa: E402,F401
    _beam_full_graph_ready as _common_beam_full_graph_ready,
)
from vllm_gr.v1.worker.model_runner_common import (
    _beam_graph_enabled_for_run as _common_beam_graph_enabled_for_run,
)
from vllm_gr.v1.worker.model_runner_common import (
    _beam_remap_inputs,
)
from vllm_gr.v1.worker.model_runner_common import (
    _create_beam_context as _common_create_beam_context,
)
from vllm_gr.v1.worker.model_runner_common import (
    _materialize_logits_indices,
    _reduce_sampler_output_for_worker_decision,
    get_beam_graph_runtime_stats,
    make_patched_execute_model as _make_common_patched_execute_model,
)

logger = init_logger(__name__)

_BEAM_POSTPROCESS_LOAD_PATCH_MARKER = "_vllm_gr_beam_postprocess_load_patched"
_PREFILL_COMPUTE_LOGITS_PATCH_MARKER = "_vllm_gr_prefill_compute_logits_patched"
_PROFILE_DIAGNOSTIC_HOOK_MARKER = "_vllm_gr_profile_diagnostic_hook_installed"
_INPUT_PREP_READY_PATCH_MARKER = "_vllm_gr_cuda_input_prep_ready_patched"


@dataclass(slots=True)
class _CudaGRRequestState:
    """Small request binding over runner-owned fixed CUDA workspace."""

    workspace: Any
    prefix_binding: Any = None
    pending_reorder_parents: Any = None
    pending_reorder_prefix: Any = None
    released: bool = False
    control_epoch: int | None = None
    control_phase: int | None = None

    @property
    def device_memory_bytes(self) -> int:
        # The storage is accounted once by the runner workspace pool.
        return 0


class CudaGRDeviceBackend:
    """Persistent GR backend for the upstream GPU ModelRunner V1."""

    def _require_workspace(self, runner: Any, state: Any) -> Any:
        from vllm_gr.v1.beam.beam_decision import get_beam_postprocess_workspace

        device = torch.device(getattr(runner, "device", "cpu"))
        if device.type != "cuda":
            raise RuntimeError(f"CUDA GR backend requires a CUDA runner, got {device}")
        workspace = get_beam_postprocess_workspace(runner, state.beam_width)
        if workspace is None:
            raise RuntimeError(
                "persistent CUDA GR requires an exact-width prewarmed Beam workspace"
            )
        if workspace.max_batch != 1:
            raise RuntimeError("persistent CUDA GR backend requires a B=1 workspace")
        if state.max_tokens > workspace.max_decode_steps:
            raise RuntimeError(
                "persistent CUDA GR generation length exceeds the prewarmed workspace: "
                f"request={state.max_tokens}, workspace={workspace.max_decode_steps}"
            )
        return workspace

    @staticmethod
    def _extract_prefix_binding(runner: Any, state: Any) -> Any:
        """Bind the native prefix block row once without a request-time clone."""

        input_batch = getattr(runner.input_batch, "_obj", runner.input_batch)
        req_index = input_batch.req_id_to_index[state.request_id]
        # vLLM 0.22.1 exposes a non-iterable MultiGroupBlockTable here.
        # Its public group collection is ``block_tables`` and integer access
        # delegates to that collection; do not assume the container itself
        # implements len()/iteration.
        native_block_tables = input_batch.block_table
        block_tables = native_block_tables.block_tables
        if len(block_tables) != 1:
            raise RuntimeError(
                "persistent CUDA GR currently supports one native KV-cache group"
            )
        native_block_table = native_block_tables[0]
        block_size = int(native_block_table.block_size)
        num_blocks = (int(state.session._state.prefix_len) + block_size - 1) // block_size
        table = native_block_table.get_device_tensor(input_batch.num_reqs)
        return (table[req_index, :num_blocks],)

    @staticmethod
    def _control_output(workspace: Any, device_step: int, token_id: int) -> torch.Tensor:
        from vllm_gr.v1.worker.cuda_async_output import fill_control

        return fill_control(workspace, device_step, token_id)

    def bootstrap_from_prefill(
        self,
        runner: Any,
        state: Any,
        sampler_output: Any,
        metadata: dict[str, Any],
    ):
        from vllm_gr.v1.beam.beam_decision import _run_cuda_prefill_decision
        from vllm_gr.v1.worker.gr_model_runner_state import GRBackendStep

        workspace = self._require_workspace(runner, state)
        decision = _run_cuda_prefill_decision(runner, sampler_output)
        if decision is None:
            raise RuntimeError("persistent CUDA GR prefill has no device Beam decision")
        token_ids = decision["out_token_ids"]
        scores = decision["out_log_probs"]
        if tuple(token_ids.shape) != (1, state.beam_width):
            raise RuntimeError("persistent CUDA GR prefill decision width mismatch")

        backend_state = state.backend_state
        if not isinstance(backend_state, _CudaGRRequestState):
            backend_state = _CudaGRRequestState(workspace=workspace)
            state.backend_state = backend_state
        elif backend_state.workspace is not workspace:
            raise RuntimeError("persistent CUDA GR workspace changed across state reuse")
        backend_state.released = False
        # Cached common states may retain backend_state across sessions. A
        # phase/epoch belongs to this bootstrap, never to the cached object.
        backend_state.control_epoch = None
        backend_state.control_phase = None
        backend_state.pending_reorder_parents = None
        backend_state.pending_reorder_prefix = None
        backend_state.prefix_binding = self._extract_prefix_binding(runner, state)
        state.prefix_binding = backend_state.prefix_binding

        width = state.beam_width
        steps = state.max_tokens
        parent_history = workspace.gr_parent_history[:1, :width, :steps]
        logprob_history = workspace.gr_logprob_history[:1, :width, :steps]
        parent_history.zero_()
        logprob_history.zero_()
        logprob_history[0, :, 0].copy_(scores[0])
        if bool(getattr(sampler_output, "_gr_deferred_prefill_status", False)):
            workspace.gr_invalid_status[:1].copy_(workspace.prefill_staging[:1])
        else:
            workspace.gr_invalid_status[:1].zero_()

        decision_state = state.session.context.decision
        decision_state.sequence[:1, :width, 0].copy_(token_ids[0])
        decision_state.beam_scores[:1, :width].copy_(scores)
        decision_state.worker_decode_steps = 1
        torch.gt(
            scores,
            float("-inf"),
            out=workspace.gr_active_mask[:1, :width],
        )

        finished = state.max_tokens == 1
        control = self._control_output(
            workspace,
            1,
            state.finish_token_id if finished else state.continue_token_id,
        )
        return GRBackendStep(
            device_step=1,
            next_token_ids=token_ids,
            beam_scores=decision_state.beam_scores[:1, :width],
            active_mask=workspace.gr_active_mask[:1, :width],
            control_output=control,
        )

    def prepare_decode_inputs(
        self,
        runner: Any,
        state: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
        metadata: dict[str, Any],
    ) -> None:
        backend_state = state.backend_state
        if not isinstance(backend_state, _CudaGRRequestState) or backend_state.released:
            raise RuntimeError("persistent CUDA GR backend state is not active")
        width = state.beam_width
        gr_output = getattr(scheduler_output, "gr_outputs", {}).get(state.request_id)
        if (
            gr_output is None
            or int(gr_output.logical_token_count) != 1
            or int(gr_output.physical_row_count) != width
        ):
            raise RuntimeError("persistent CUDA GR physical-row sidecar is missing or invalid")
        req_ids = list(getattr(runner.input_batch, "req_ids", ()))
        if req_ids != [state.request_id]:
            raise RuntimeError(
                "persistent CUDA GR currently requires an isolated B=1 runner batch"
            )
        if len(num_scheduled_tokens) != 1 or int(num_scheduled_tokens[0]) != width:
            raise RuntimeError(
                "persistent CUDA GR adapter did not publish the physical Beam width"
            )
        input_ids = runner.input_ids.gpu
        if input_ids.numel() < width:
            raise RuntimeError("GPU ModelRunner input buffer is smaller than Beam width")
        input_ids[:width].copy_(state.next_token_ids.reshape(-1)[:width])

        expected_decode_step = state.step - 1
        if state.session.decode_step != expected_decode_step:
            raise RuntimeError(
                "persistent CUDA GR Beam session/device step mismatch: "
                f"session={state.session.decode_step}, expected={expected_decode_step}"
            )
        state.session.context.decode_step_buf.fill_(expected_decode_step)
        state.session.context._gr_persistent_prefix_binding = state.prefix_binding

    def advance_beam(
        self,
        runner: Any,
        state: Any,
        sampler_output: Any,
        metadata: dict[str, Any],
        *,
        is_last_step: bool,
    ):
        from vllm_gr.v1.beam.beam_decision import run_beam_decision
        from vllm_gr.v1.worker.gr_model_runner_state import GRBackendStep

        backend_state = state.backend_state
        if not isinstance(backend_state, _CudaGRRequestState) or backend_state.released:
            raise RuntimeError("persistent CUDA GR backend state is not active")
        workspace = backend_state.workspace
        width = state.beam_width
        decision_state = state.session.context.decision
        if decision_state.worker_decode_steps != state.step:
            raise RuntimeError(
                "persistent CUDA GR decision/device step mismatch: "
                f"decision={decision_state.worker_decode_steps}, state={state.step}"
            )
        workspace.gr_previous_scores[:1, :width].copy_(
            decision_state.beam_scores[:1, :width]
        )
        state.session._state.is_last_step = is_last_step
        decision = run_beam_decision(
            runner,
            state.session,
            sampler_output,
            is_last_step=is_last_step,
            result_width=metadata.get("result_width"),
            reorder_kv=False,
        )
        if decision is None:
            raise RuntimeError("persistent CUDA GR decode has no device Beam decision")
        if bool(getattr(sampler_output, "_gr_deferred_decode_status", False)):
            torch.maximum(
                workspace.gr_invalid_status[:1],
                workspace.prefill_staging[:1],
                out=workspace.gr_invalid_status[:1],
            )

        token_ids = decision["out_token_ids"]
        parent_ids = decision["out_token_index"]
        scores = decision["out_log_probs"]
        result_width = int(token_ids.shape[1])
        gather_indices = workspace.gr_gather_indices[0, :result_width]
        gather_indices.copy_(parent_ids[0])
        torch.index_select(
            workspace.gr_previous_scores[0, :width],
            0,
            gather_indices,
            out=workspace.gr_parent_scores[0, :result_width],
        )
        workspace.gr_parent_history[0, :result_width, state.step].copy_(parent_ids[0])
        torch.sub(
            scores[0],
            workspace.gr_parent_scores[0, :result_width],
            out=workspace.gr_logprob_history[0, :result_width, state.step],
        )
        backend_state.pending_reorder_parents = decision.get("_gr_reorder_parents")
        backend_state.pending_reorder_prefix = decision.get("_gr_reorder_prefix")
        workspace.gr_active_mask[:1, :width].fill_(False)
        torch.gt(
            scores,
            float("-inf"),
            out=workspace.gr_active_mask[:1, :result_width],
        )

        next_device_step = state.device_step + 1
        control = self._control_output(
            workspace,
            next_device_step,
            state.finish_token_id if is_last_step else state.continue_token_id,
        )
        runner._beam_decision_tensors = decision
        return GRBackendStep(
            device_step=next_device_step,
            next_token_ids=token_ids,
            beam_scores=decision_state.beam_scores[:1, :result_width],
            active_mask=workspace.gr_active_mask[:1, :result_width],
            control_output=control,
        )

    def reorder_kv(
        self,
        runner: Any,
        state: Any,
        step_result: Any,
        *,
        is_last_step: bool,
    ) -> None:
        del runner, step_result
        backend_state = state.backend_state
        if not isinstance(backend_state, _CudaGRRequestState):
            raise RuntimeError("persistent CUDA GR backend state is missing")
        parents = backend_state.pending_reorder_parents
        prefix = backend_state.pending_reorder_prefix
        backend_state.pending_reorder_parents = None
        backend_state.pending_reorder_prefix = None
        if is_last_step:
            if parents is not None or prefix is not None:
                raise RuntimeError("terminal CUDA GR decision unexpectedly requested KV reorder")
            return
        if parents is None or prefix is None:
            raise RuntimeError("non-terminal CUDA GR decision is missing KV reorder data")
        from vllm_gr.v1.beam.beam_decision import _reorder_kv_same_step

        _reorder_kv_same_step(state.session, parents[0], prefix[0])

    @staticmethod
    def _pack_final_result(state: Any, workspace: Any) -> tuple[torch.Tensor, int, int]:
        """Backtrace raw lineage once and pack the final device ABI."""

        width = int(state.next_token_ids.shape[1])
        steps = state.step
        count = width * steps
        staging = workspace.gr_final_result
        staging[:1].copy_(workspace.gr_invalid_status[:1])
        packed = staging[workspace.max_batch :]
        sequences = state.session.context.decision.sequence[0, :width, :steps]
        packed[:count].view(width, steps).copy_(sequences)
        packed_parents = packed[count : 2 * count].view(width, steps)
        packed_logprobs = packed[2 * count : 3 * count].view(width, steps).view(
            torch.float32
        )
        current_indices = workspace.gr_backtrace_indices[0, 0, :width]
        next_indices = workspace.gr_backtrace_indices[1, 0, :width]
        torch.arange(width, dtype=torch.int64, device=workspace.device, out=current_indices)
        for generation_step in range(steps - 1, -1, -1):
            torch.index_select(
                workspace.gr_parent_history[0, :, generation_step],
                0,
                current_indices,
                out=workspace.gr_backtrace_parents[0, :width],
            )
            torch.index_select(
                workspace.gr_logprob_history[0, :, generation_step],
                0,
                current_indices,
                out=workspace.gr_backtrace_logprobs[0, :width],
            )
            packed_parents[:, generation_step].copy_(
                workspace.gr_backtrace_parents[0, :width]
            )
            packed_logprobs[:, generation_step].copy_(
                workspace.gr_backtrace_logprobs[0, :width]
            )
            next_indices.copy_(workspace.gr_backtrace_parents[0, :width])
            current_indices, next_indices = next_indices, current_indices
        packed[3 * count : 3 * count + width].copy_(
            state.beam_scores[0, :width].view(torch.int32)
        )
        return staging, 3 * count + width, width

    @staticmethod
    def _check_final_status(host_status: Any) -> None:
        if any(int(status) != 0 for status in host_status):
            raise RuntimeError(
                "constraint Backend returned no legal candidate for persistent CUDA GR"
            )

    def finalize_output(self, runner: Any, state: Any) -> dict[str, Any]:
        from vllm_gr.v1.worker.gr_model_runner_state import GR_FINAL_OUTPUT_FORMAT

        del runner
        backend_state = state.backend_state
        if not isinstance(backend_state, _CudaGRRequestState) or backend_state.released:
            raise RuntimeError("persistent CUDA GR backend state is not active")
        workspace = backend_state.workspace
        steps = state.step
        staging, payload_words, width = self._pack_final_result(state, workspace)
        copy_words = workspace.max_batch + payload_words
        host = workspace.gr_host_final_result[:copy_words]
        event = workspace.gr_final_copy_event
        if event is None or not host.is_pinned():
            raise RuntimeError("persistent CUDA GR final staging must be pinned and event-backed")
        stream = torch.cuda.current_stream(workspace.device)
        with record_function_or_nullcontext("gr_final_d2h"):
            host.copy_(staging[:copy_words], non_blocking=True)
            event.record(stream)
            event.synchronize()
        host_numpy = workspace.gr_host_final_result_numpy[:copy_words]
        self._check_final_status(host_numpy[: workspace.max_batch])
        return {
            "request_id": state.request_id,
            "buffer_format": GR_FINAL_OUTPUT_FORMAT,
            "shape": [width, steps],
            "buffer": host_numpy[
                workspace.max_batch : workspace.max_batch + payload_words
            ].tobytes(),
            "is_last_step": True,
            "step": steps,
            "device_step": state.device_step,
        }

    def release(self, state: Any) -> None:
        backend_state = state.backend_state
        if isinstance(backend_state, _CudaGRRequestState):
            backend_state.pending_reorder_parents = None
            backend_state.pending_reorder_prefix = None
            backend_state.prefix_binding = None
            state.prefix_binding = None
            if hasattr(state.session.context, "_gr_persistent_prefix_binding"):
                state.session.context._gr_persistent_prefix_binding = None
            backend_state.released = True


def create_gr_device_backend(runner):
    """Select the CUDA V1 backend once during ModelRunner initialization."""

    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    if bool(getattr(_get_gr_config(runner), "persistent_runner_state", False)):
        if not getattr(runner, "use_async_scheduling", False):
            raise RuntimeError("persistent CUDA GR requires ASYNC output transport")
        from vllm_gr.v1.worker.cuda_async_output import CudaOutputTransport
        from vllm_gr.v1.worker.cuda_completion import load_native_completion_api

        runner._gr_cuda_output_transport = CudaOutputTransport(runner, load_native_completion_api())
    return CudaGRDeviceBackend()


def make_patched_shutdown(original):
    @wraps(original)
    def shutdown(self, *args, **kwargs):
        transport = getattr(self, "_gr_cuda_output_transport", None)
        if transport is not None:
            transport.shutdown()
        # A failed transport drain must not return the common Beam KV/session.
        return original(self, *args, **kwargs)

    return shutdown


def _persistent_cuda_geometry(scheduler_output: Any) -> tuple[str, int] | None:
    beam_data = getattr(scheduler_output, "beam_data", {}) or {}
    persistent = [
        req_id
        for req_id, metadata in beam_data.items()
        if metadata.get("persistent_runner_state") and metadata.get("is_beam_decode")
    ]
    if not persistent:
        return None
    if len(persistent) != 1 or set(scheduler_output.num_scheduled_tokens) != set(persistent):
        raise RuntimeError(
            "persistent CUDA GR decode currently requires one isolated B=1 request"
        )
    request_id = persistent[0]
    sidecar = getattr(scheduler_output, "gr_outputs", {}).get(request_id)
    if sidecar is None:
        raise RuntimeError("persistent CUDA GR decode is missing its typed sidecar")
    if int(sidecar.logical_token_count) != 1:
        raise RuntimeError("persistent CUDA GR native accounting must remain one logical token")
    physical_rows = int(sidecar.physical_row_count)
    if physical_rows != int(beam_data[request_id]["beam_width"]):
        raise RuntimeError("persistent CUDA GR physical row count must equal Beam width")
    if scheduler_output.num_scheduled_tokens.get(request_id) != 1:
        raise RuntimeError("persistent CUDA GR SchedulerOutput must carry logical count one")
    if int(scheduler_output.total_num_scheduled_tokens) != 1:
        raise RuntimeError("persistent CUDA GR total logical token count must be one")
    return request_id, physical_rows


def make_patched_execute_model_gpu(original):
    """Expose physical W only inside the CUDA V1 execution frame."""

    common_execute = _make_common_patched_execute_model(original)
    if getattr(common_execute, "_vllm_gr_gpu_physical_execute_patched", False):
        return common_execute

    def guarded_execute(self, scheduler_output, *args, **kwargs):
        try:
            return common_execute(self, scheduler_output, *args, **kwargs)
        except BaseException as error:
            transport = getattr(self, "_gr_cuda_output_transport", None)
            if transport is not None:
                transport.quarantine(error)
            raise

    @wraps(common_execute)
    def execute_with_physical_geometry(self, scheduler_output, *args, **kwargs):
        transport = getattr(self, "_gr_cuda_output_transport", None)
        if transport is not None:
            transport.check_live()
        geometry = _persistent_cuda_geometry(scheduler_output)
        if geometry is None:
            return guarded_execute(self, scheduler_output, *args, **kwargs)
        if getattr(self, "speculative_config", None) is not None:
            raise RuntimeError("persistent CUDA GR does not support speculative decode")
        parallel_config = getattr(self, "parallel_config", None)
        if getattr(parallel_config, "pipeline_parallel_size", 1) != 1:
            raise RuntimeError("persistent CUDA GR does not support pipeline parallelism")
        if has_kv_transfer_group():
            raise RuntimeError(
                "persistent CUDA GR does not support an active KV connector"
            )
        if has_ec_transfer():
            raise RuntimeError(
                "persistent CUDA GR does not support active EC transfer"
            )
        request_id, physical_rows = geometry
        scheduler_output.num_scheduled_tokens[request_id] = physical_rows
        scheduler_output.total_num_scheduled_tokens = physical_rows
        scheduler_output._gr_cuda_logical_geometry = (request_id, 1, physical_rows)
        try:
            return guarded_execute(self, scheduler_output, *args, **kwargs)
        finally:
            scheduler_output.num_scheduled_tokens[request_id] = 1
            scheduler_output.total_num_scheduled_tokens = 1
            del scheduler_output._gr_cuda_logical_geometry

    setattr(execute_with_physical_geometry, _WORKER_EXECUTE_PATCH_MARKER, True)
    setattr(execute_with_physical_geometry, "_vllm_gr_gpu_physical_execute_patched", True)
    return execute_with_physical_geometry


def make_patched_synchronize_input_prep(original):
    """Avoid waiting on an already-completed GR Decode input-preparation event.

    Keep the upstream single-buffer ownership protocol, including its wait
    when the previous H2D is not complete. The guard must enclose _update_states
    as well as _prepare_inputs: native block zeroing and block-table updates
    also reuse pinned sources. No buffer pointers or Event generations change.

    The physical-geometry marker exists only inside a validated persistent
    Decode execute frame. Unlike _current_beam_data it does not survive into
    sample_tokens or a later dummy run. Ordinary/Prefill/dummy calls delegate.
    """
    if getattr(original, _INPUT_PREP_READY_PATCH_MARKER, False):
        return original

    @wraps(original)
    @contextmanager
    def input_prep_when_ready(self):
        output = getattr(self, "_current_scheduler_output", None)
        active_decode = getattr(output, "_gr_cuda_logical_geometry", None) is not None
        event = getattr(self, "prepare_inputs_event", None)
        if not active_decode or event is None:
            with original(self):
                yield
            return

        # A query error must propagate before any Host staging is modified.
        # A false result is not a release of the previous DMA's ownership.
        if not event.query():
            self._gr_input_prep_waits = getattr(self, "_gr_input_prep_waits", 0) + 1
            with original(self):
                yield
            return

        self._gr_input_prep_ready_skips = getattr(self, "_gr_input_prep_ready_skips", 0) + 1
        try:
            yield
        finally:
            # Same end-of-region boundary as upstream, even on preparation
            # failure. Leaving GR therefore hands ordinary requests the same
            # event and the current contents of every original Host buffer.
            event.record()

    setattr(input_prep_when_ready, _INPUT_PREP_READY_PATCH_MARKER, True)
    return input_prep_when_ready




def get_gr_memory_stats(worker) -> dict[str, int]:
    """Add CUDA graph-pool accounting to the shared device-neutral totals."""

    from vllm_gr.v1.worker.gr_model_runner_state import get_gr_memory_stats as common_stats

    stats = common_stats(worker)
    runner = getattr(worker, "model_runner", worker)
    device = getattr(runner, "device", None)
    if getattr(device, "type", None) != "cuda":
        return stats
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    allocated = 0
    reserved = 0
    for segment in torch.cuda.memory._snapshot().get("segments", ()):
        if segment.get("device") != device_index:
            continue
        if tuple(segment.get("segment_pool_id", (0, 0))) == (0, 0):
            continue
        allocated += int(segment.get("allocated_size", 0))
        reserved += int(segment.get("total_size", 0))
    stats["graph_pool_allocated_bytes"] = allocated
    stats["graph_pool_reserved_bytes"] = reserved
    transport = getattr(runner, "_gr_cuda_output_transport", None)
    stats["gr_control_pinned_bytes"] = transport.stats()["control_pinned_bytes"] if transport else 0
    return stats


def get_gr_profile_runtime_stats(worker) -> dict:
    """Named, read-only CUDA Worker RPC for attributable profiling."""
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("GPU worker has no model_runner for profiling diagnostics")
    device = getattr(runner, "device", None)
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device is None or device_type != "cuda":
        raise RuntimeError(f"GPU profiling diagnostic requires a CUDA runner, got {device!r}")
    properties = torch.cuda.get_device_properties(device)
    runner_class = type(runner)
    return {
        "backend": "gpu",
        "worker_class": f"{type(worker).__module__}.{type(worker).__qualname__}",
        "runner_class": f"{runner_class.__module__}.{runner_class.__qualname__}",
        "device": str(device),
        "device_name": str(getattr(properties, "name", properties)),
        "device_properties": repr(properties),
        "memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "beam_graph": get_beam_graph_runtime_stats(worker),
        "input_prep": {
            "ready_event_skips": int(getattr(runner, "_gr_input_prep_ready_skips", 0)),
            "not_ready_upstream_waits": int(getattr(runner, "_gr_input_prep_waits", 0)),
        },
        "control_output": (
            runner._gr_cuda_output_transport.stats()
            if getattr(runner, "_gr_cuda_output_transport", None) is not None else None
        ),
    }


def install_worker_diagnostic_hook(worker_cls: type) -> int:
    """Install the CUDA diagnostic as a safe named utility RPC."""
    if getattr(worker_cls, _PROFILE_DIAGNOSTIC_HOOK_MARKER, False):
        return 0
    worker_cls.get_gr_profile_runtime_stats = get_gr_profile_runtime_stats  # type: ignore[attr-defined]
    setattr(worker_cls, _PROFILE_DIAGNOSTIC_HOOK_MARKER, True)
    return 1


def _install_cuda_prefill_compute_logits_hook(runner) -> bool:
    """Place the prepared TopK chain immediately after LM-Head returns."""

    model = getattr(runner, "model", None)
    original = getattr(model, "compute_logits", None)
    if not callable(original):
        return False
    if getattr(original, _PREFILL_COMPUTE_LOGITS_PATCH_MARKER, False):
        return True

    @wraps(original)
    def compute_logits_with_prefill_prelaunch(*args, **kwargs):
        logits = original(*args, **kwargs)
        from vllm_gr.v1.worker.constraint_hook import (
            launch_prepared_cuda_prefill_constraint,
        )

        launch_prepared_cuda_prefill_constraint(runner, logits)
        return logits

    setattr(compute_logits_with_prefill_prelaunch, _PREFILL_COMPUTE_LOGITS_PATCH_MARKER, True)
    model.compute_logits = compute_logits_with_prefill_prelaunch
    return True


def initialize_beam_postprocess_workspaces(
    runner,
    *,
    max_batch: int | None = None,
):
    """Prewarm the configured CUDA Beam postprocess bucket on ``runner``.

    The production Worker currently owns a one-slot Beam context.  ``max_batch``
    remains an explicit construction input so operator tests and a future
    multi-request Worker can reserve more than one batch row without changing
    the workspace layout.
    """

    device = torch.device(getattr(runner, "device", "cpu"))
    if device.type != "cuda":
        return None

    from vllm_gr.v1.beam.beam_decision import (
        BeamPostprocessWorkspace,
        BeamPostprocessWorkspaceKey,
        BeamPostprocessWorkspacePool,
    )
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    gr_cfg = _get_gr_config(runner)
    width = gr_cfg.beam_max_width
    if not bool(gr_cfg.beam_worker_decision) or width is None:
        return None

    context = getattr(runner, "_beam_context", None)
    resolved_max_batch = (
        int(max_batch) if max_batch is not None else int(getattr(context, "max_batch", 1))
    )
    dtype = getattr(getattr(runner, "model_config", None), "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"CUDA Beam postprocess workspace requires model_config.dtype, got {dtype!r}"
        )

    key = BeamPostprocessWorkspaceKey(int(width), int(width), dtype)
    existing = getattr(runner, "_beam_postprocess_workspaces", None)
    if existing is not None:
        workspace = existing.require(key)
        if workspace.max_batch != resolved_max_batch or workspace.max_decode_steps != int(
            gr_cfg.beam_max_decode_steps
        ):
            raise RuntimeError(
                "prewarmed Beam postprocess workspace geometry changed after "
                "initialization; rebuilding fixed-address buffers is forbidden"
            )
        return existing

    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=resolved_max_batch,
        beam_width=int(width),
        top_k=int(width),
        max_decode_steps=int(gr_cfg.beam_max_decode_steps),
        device=device,
        logits_dtype=dtype,
        pin_host=True,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    runner._beam_postprocess_workspaces = pool
    transport = getattr(runner, "_gr_cuda_output_transport", None)
    if transport is not None:
        transport.prepare_workspace(workspace)
    logger.info(
        "vllm-gr: prewarmed CUDA Beam postprocess workspace "
        "(batch=%d, width=%d, top_k=%d, dtype=%s, device_bytes=%d, pinned_bytes=%d)",
        resolved_max_batch,
        width,
        width,
        dtype,
        workspace.device_memory_bytes,
        workspace.pinned_memory_bytes,
    )
    return pool


def make_load_model_with_beam_postprocess_workspace(original):
    """Initialize fixed Beam postprocess storage after model weights load."""

    if getattr(original, _BEAM_POSTPROCESS_LOAD_PATCH_MARKER, False):
        return original

    @wraps(original)
    def load_model_with_beam_postprocess_workspace(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        workspace_pool = initialize_beam_postprocess_workspaces(self)
        if workspace_pool is not None:
            # The pool is allocated after native load_model has measured the
            # model weights, but before Worker memory profiling.  Treat its
            # device storage as model-owned memory so KV-cache capacity is not
            # computed from bytes that are already permanently resident.
            self.model_memory_usage = int(getattr(self, "model_memory_usage", 0)) + int(
                workspace_pool.device_memory_bytes
            )
        _install_cuda_prefill_compute_logits_hook(self)
        return result

    setattr(
        load_model_with_beam_postprocess_workspace,
        _BEAM_POSTPROCESS_LOAD_PATCH_MARKER,
        True,
    )
    return load_model_with_beam_postprocess_workspace


def make_patched_prepare_inputs_gpu(_orig_prepare_inputs):
    """Wrap the GPU two-item ``_prepare_inputs`` contract."""
    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    def prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        beam_data = getattr(scheduler_output, "beam_data", {})
        logits_indices, spec_decode_metadata = _orig_prepare_inputs(
            self, scheduler_output, num_scheduled_tokens
        )

        # Upstream GPU _prepare_inputs materializes the native logical token
        # into self.input_ids.gpu. Publish the persistent physical Beam rows
        # afterwards so the native write cannot overwrite the backend's D2D
        # token update. GPU attention metadata is built after this wrapper
        # returns, while the NPU adapter intentionally uses the opposite order.
        for req_id, mdata in beam_data.items():
            if mdata.get("persistent_runner_state") and mdata.get("is_beam_decode"):
                state = getattr(self, "_gr_model_runner_states", {}).get(req_id)
                if state is None:
                    raise RuntimeError(
                        f"persistent GR state is missing for request {req_id!r}"
                    )
                state.prepare_decode_inputs(
                    self,
                    scheduler_output,
                    num_scheduled_tokens,
                    mdata,
                )

        # GPU attention metadata is built after _prepare_inputs returns.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        if not beam_data:
            return logits_indices, spec_decode_metadata

        # The native request advances with one logical control token while the
        # grouped model invocation consumes W physical Beam tokens. Write each
        # request into its packed-token slice so this path does not depend on
        # the GR request occupying batch row zero.
        token_offset = 0
        for batch_idx, req_id in enumerate(self.input_batch.req_ids):
            scheduled_tokens = (
                num_scheduled_tokens[batch_idx] if batch_idx < len(num_scheduled_tokens) else 0
            )
            mdata = beam_data.get(req_id)
            if not mdata or not mdata.get("persistent_request"):
                token_offset += scheduled_tokens
                continue
            if mdata.get("persistent_runner_state") and mdata.get("is_beam_decode"):
                # Physical row expansion and D2D input writes belong to the
                # selected GPU backend's prepare_decode_inputs implementation.
                pass
            else:
                next_token_ids = mdata.get("next_token_ids")
                if next_token_ids:
                    width = int(mdata["beam_width"])
                    if len(next_token_ids) != width or scheduled_tokens != width:
                        raise RuntimeError(
                            "persistent GR next_token_ids width does not match "
                            "the scheduled Beam rows"
                        )
                    self.input_ids.gpu[token_offset : token_offset + width].copy_(
                        torch.tensor(
                            next_token_ids,
                            dtype=self.input_ids.gpu.dtype,
                            device=self.device,
                        )
                    )
            token_offset += scheduled_tokens

        # Move state allocation ahead of the final Prefill forward; native
        # input, slot-mapping and attention preparation stay unchanged.
        for req_id, mdata in beam_data.items():
            if (
                mdata.get("persistent_runner_state")
                and mdata.get("is_constraint_prefill")
                and mdata.get("is_last_prefill", False)
            ):
                with torch.autograd.profiler.record_function("gr_prefill_prearm"):
                    _create_persistent_gr_state(self, req_id, mdata)

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        with torch.autograd.profiler.record_function("prefill_constraint_prepare"):
            prepare_cuda_prefill_constraint(self, scheduler_output)
        return (
            _materialize_logits_indices(self, new_logits_indices),
            spec_decode_metadata,
        )

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        persistent_decode = any(
            metadata.get("persistent_runner_state") and metadata.get("is_beam_decode")
            for metadata in getattr(scheduler_output, "beam_data", {}).values()
        )
        marker = (
            record_function_or_nullcontext("gr_decode_prepare")
            if persistent_decode
            else nullcontext()
        )
        with marker:
            # Include Beam row/sampling-metadata remapping, not just the
            # upstream preparation and persistent input publication.
            return prepare_inputs(self, scheduler_output, num_scheduled_tokens)

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_sample_gpu(_orig_sample):
    """Run canonical constraints first, retaining the legacy override path."""
    if getattr(_orig_sample, _WORKER_SAMPLE_PATCH_MARKER, False):
        return _orig_sample

    @wraps(_orig_sample)
    def patched_sample(self, logits, spec_decode_metadata=None):
        sampler_output = consume_prelaunched_cuda_prefill(self, logits, spec_decode_metadata)
        if sampler_output is None:
            sampler_output = try_sample_with_constraint_backend(
                self,
                logits,
                spec_decode_metadata,
            )
        if sampler_output is None:
            sampler_output = _orig_sample(self, logits, spec_decode_metadata)
            apply_constrained_sample_override(self, logits, sampler_output)
        maybe_run_worker_beam_decision(self, sampler_output)
        return sampler_output

    setattr(patched_sample, _WORKER_SAMPLE_PATCH_MARKER, True)
    return patched_sample


def make_patched_sample_tokens_gpu(_orig_sample_tokens):
    """Attach a CUDA worker decision produced during async sampling."""
    if getattr(_orig_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False):
        return _orig_sample_tokens

    @wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output, *args, **kwargs):
        from vllm_gr.v1.worker.model_runner_common import _attach_beam_search_result
        from vllm_gr.v1.worker.cuda_async_output import (
            install_async_output_adapter, sampling_scope,
        )
        from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

        install_async_output_adapter(AsyncGPUModelRunnerOutput)
        # The ephemeral native state is authoritative; _current_scheduler_output
        # may still refer to a completed request or housekeeping dispatch.
        state = getattr(self, "execute_model_state", None)
        scheduler_output = getattr(state, "scheduler_output", None)
        try:
            with sampling_scope(self, scheduler_output):
                out = _orig_sample_tokens(self, grammar_output, *args, **kwargs)
                _attach_beam_search_result(self, out)
                return out
        except BaseException as error:
            transport = getattr(self, "_gr_cuda_output_transport", None)
            if transport is not None:
                transport.quarantine(error)
            raise

    setattr(patched_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, True)
    return patched_sample_tokens


def make_patched_bookkeeping_sync_gpu(_orig_bookkeeping_sync):
    """Wrap the GPU seven-item ``_bookkeeping_sync`` contract."""
    if getattr(_orig_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, False):
        return _orig_bookkeeping_sync

    @wraps(_orig_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        if hasattr(self, "_beam_logprobs_cache") and self._beam_logprobs_cache:
            if hasattr(scheduler_output, "finished_req_ids"):
                for req_id in scheduler_output.finished_req_ids:
                    self._beam_logprobs_cache.pop(req_id, None)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            return _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        # A worker decision already selected W survivors.  Shrink the ordinary
        # sampler tensors before upstream's synchronous D2H bookkeeping; doing
        # this after _orig_bookkeeping_sync is too late and copies/converts the
        # redundant W-by-W candidate matrix on every decode step.
        _reduce_sampler_output_for_worker_decision(self, sampler_output)
        restore_to_list = None
        delete_to_list = False
        host_ready_sampled = (
            not getattr(self, "use_async_scheduling", False)
            and bool(getattr(self, "_beam_result_host_ready", False))
            and sampler_output is not None
            and isinstance(getattr(sampler_output, "sampled_token_ids", None), torch.Tensor)
            and sampler_output.sampled_token_ids.device.type == "cpu"
        )
        if host_ready_sampled:
            # The dedicated CUDA prefill path already synchronized its one
            # pinned D2H and exposes CPU views over that exact staging. Keep all
            # upstream bookkeeping, but prevent _to_list from enqueueing and
            # synchronizing a second copy of the sampled token.
            original_to_list = getattr(self, "_to_list", None)
            if callable(original_to_list):
                instance_dict = getattr(self, "__dict__", {})
                had_instance_to_list = "_to_list" in instance_dict
                previous_instance_to_list = instance_dict.get("_to_list")
                try:
                    self._to_list = lambda value: value.tolist()
                except (AttributeError, TypeError):
                    pass
                else:
                    if had_instance_to_list:
                        restore_to_list = previous_instance_to_list
                    else:
                        delete_to_list = True
        if (
            sampler_output is not None
            and isinstance(getattr(sampler_output, "sampled_token_ids", None), torch.Tensor)
            and sampler_output.sampled_token_ids.shape[0] > len(request_row_configs)
        ):
            # vLLM 0.22 sizes its synchronous pinned sampled-token buffer by
            # max_num_seqs (logical requests), while the legacy Beam path has
            # one sampler row per parent.  Only the first sampled ID per
            # logical request is bookkeeping data; all actual Beam candidates
            # are carried by logprobs_tensors.  Collapse this small tensor
            # before upstream's D2H copy and leave the W/W^2 logprob payload
            # untouched for the compatibility path.
            physical_row = 0
            logical_sampled_rows = []
            for req_id, _, logits_rows, *_ in request_row_configs:
                if req_id is not None:
                    logical_sampled_rows.append(
                        sampler_output.sampled_token_ids[physical_row : physical_row + 1]
                    )
                physical_row += logits_rows
            sampler_output.sampled_token_ids = torch.cat(
                logical_sampled_rows,
                dim=0,
            )

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            if delete_to_list:
                delattr(self, "_to_list")
            elif restore_to_list is not None:
                self._to_list = restore_to_list
            self.input_batch = orig_input_batch

        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
        )

        materialize_sync_output = bool(
            not getattr(self, "use_async_scheduling", False)
            and sampler_output is not None
            and getattr(sampler_output, "sampled_token_ids", None) is not None
            and getattr(sampler_output, "logprobs_tensors", None) is not None
        )
        if materialize_sync_output:
            # Collapse runs after upstream bookkeeping because it must restore
            # the logical request map.  In synchronous mode there is no later
            # AsyncGPUModelRunnerOutput conversion, so materialize the now-
            # collapsed tensors here.  Returning ``[[]]`` was accepted by an
            # older scheduler but vLLM 0.22 requires LogprobsLists (including
            # ``slice_request``) for the flattened candidate payload.
            collapsed_logprobs = (
                sampler_output.logprobs_tensors.tolists()
                if logprobs_lists is not None and sampler_output.logprobs_tensors is not None
                else None
            )
            collapsed_sampled = sampler_output.sampled_token_ids[:n].tolist()
        else:
            # AsyncGPUModelRunnerOutput copies the post-collapse tensors later.
            collapsed_logprobs = [[] for _ in range(n)] if logprobs_lists is not None else None
            collapsed_sampled = [[] for _ in range(n)]

        return (
            num_nans_in_logits,
            collapsed_logprobs,
            collapsed_sampled,
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    setattr(patched_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, True)
    return patched_bookkeeping_sync


def _beam_full_graph_ready(runner, num_tokens: int) -> bool:
    """Deprecated local alias — see model_runner_common._beam_full_graph_ready.

    Kept (delegating) so tests and callers that monkeypatch THIS module's
    attribute affect the GPU determine patch, which resolves the helper
    through this module's globals.
    """
    return _common_beam_full_graph_ready(runner, num_tokens)


def _beam_graph_enabled_for_run(runner) -> bool:
    """Deprecated local alias — see model_runner_common._beam_graph_enabled_for_run."""
    return _common_beam_graph_enabled_for_run(runner)


def make_patched_determine_batch_execution_and_padding_gpu(_orig_determine):
    """Override cudagraph_mode to FULL for beam decode steps.

    In standard vllm, cascade attention sets ``use_cascade_attn=True``,
    which causes ``_determine_batch_execution_and_padding`` to call
    ``dispatch_cudagraph(disable_full=True)`` — preventing FULL CUDA Graph
    capture.  For beam search decode, we override the result to force
    ``CUDAGraphMode.FULL`` with ``uniform_decode=True``, the same explicit
    override the vllm-GR ACL Graph integration applies for beam decode.

    This patch is the single entry point that enables CUDA Graph capture
    for beam decode: all downstream logic (CUDAGraphWrapper lazy capture
    and replay) flows from the ``cudagraph_runtime_mode`` value set here.
    """
    if getattr(_orig_determine, _WORKER_DETERMINE_PATCH_MARKER, False):
        return _orig_determine

    @wraps(_orig_determine)
    def patched_determine(self, *args, **kwargs):
        from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

        result = _orig_determine(self, *args, **kwargs)

        beam_info = getattr(self, "_current_beam_info", None)
        if beam_info is not None:
            beam_data = beam_info.get("beam_data", {})
            beam_keys = {
                req_id
                for req_id, mdata in beam_data.items()
                if mdata.get("is_beam_decode", False) and "decode_step" in mdata
            }
            if beam_keys:
                # Beam decode FULL mode requires an active BeamSearchSession
                # — the metadata builder and Triton kernels read slot/pool/
                # suffix-buf pointers from it.  If _get_or_create_beam_session
                # failed (e.g. config unavailable), beam_info has no session
                # and the FULL path would crash on None access.
                if beam_info.get("beam_session") is None:
                    if result[0] == CUDAGraphMode.FULL:
                        result = (
                            CUDAGraphMode.NONE,
                            BatchDescriptor(num_tokens=result[1].num_tokens),
                        ) + result[2:]
                    return result

                # FULL is granted only for a PURE beam-decode batch: the
                # captured beam graph contains no standard-attention path, so
                # a mixed batch replaying it would compute garbage for the
                # non-beam requests.  ``req_ids`` (filled by the patched
                # _prepare_inputs) enumerates the whole batch.
                batch_req_ids = set(beam_info.get("req_ids") or [])
                pure_beam_batch = batch_req_ids == beam_keys
                if not pure_beam_batch:
                    # Mixed batch: never touch the dispatch result beyond
                    # making sure a native FULL graph is not replayed for beam
                    # tokens (it contains no beam kernels).
                    if result[0] == CUDAGraphMode.FULL:
                        result = (
                            CUDAGraphMode.NONE,
                            BatchDescriptor(num_tokens=result[1].num_tokens),
                        ) + result[2:]
                    return result

                # The startup beam graph is captured only for the default
                # no-LoRA descriptor and ``_beam_full_graph_ready`` is keyed
                # by width alone.  Granting FULL here would rewrite the
                # descriptor without has_lora/num_active_loras and silently
                # route an active-LoRA beam batch to the LoRA-free graph
                # (and a native FULL graph lacks beam kernels anyway), so
                # downgrade to eager until LoRA-aware beam graphs are
                # captured per full descriptor.
                if getattr(result[1], "has_lora", False):
                    if result[0] == CUDAGraphMode.FULL:
                        result = (
                            CUDAGraphMode.NONE,
                            BatchDescriptor(num_tokens=result[1].num_tokens),
                        ) + result[2:]
                    return result

                # Beam graphs are opt-in via GRConfig (beam_max_width) and
                # can be disabled for A/B comparison (beam_graph_enabled /
                # CLI --no-beam-graph).  Disabled or unconfigured -> fully
                # eager, nothing below changes the dispatch result.
                if not _beam_graph_enabled_for_run(self):
                    return result

                # Exact-width gate: compare the UNPADDED scheduled token count
                # against the captured graph's width.  ``result[1].num_tokens``
                # is the dispatcher-padded size, so a width of e.g. 100 with a
                # graph captured at 128 would otherwise replay a graph whose
                # tail 28 rows are garbage (only accidentally masked by
                # cu_seqlens / logits_indices).  For a pure beam-decode batch
                # the unpadded count is exactly the beam width.
                num_tokens_arg = kwargs.get("num_tokens")
                if num_tokens_arg is None and args:
                    num_tokens_arg = args[0]
                real_num_tokens = (
                    int(num_tokens_arg) if num_tokens_arg is not None else result[1].num_tokens
                )

                # The beam graph is captured once at startup; the runtime is
                # replay-only.  Grant FULL iff a graph for this exact width
                # (== real_num_tokens for a pure beam batch) exists in the
                # captured set; otherwise fall back to eager — and if the
                # native dispatch already returned FULL, downgrade to NONE so
                # the beam tokens never replay a native (beam-kernel-free) graph.
                if not _beam_full_graph_ready(self, real_num_tokens):
                    if result[0] == CUDAGraphMode.FULL:
                        result = (
                            CUDAGraphMode.NONE,
                            BatchDescriptor(num_tokens=real_num_tokens),
                        ) + result[2:]
                    return result

                new_descriptor = _beam_graph_batch_descriptor(real_num_tokens)
                logger.debug(
                    "Beam decode: cudagraph_mode %s -> FULL, batch_descriptor %s -> %s",
                    result[0],
                    result[1],
                    new_descriptor,
                )
                # Count only real runtime dispatches.  Startup beam capture
                # calls ``_dummy_beam_forward`` directly and therefore never
                # reaches this branch.  The counter gives integration tests
                # and profiling harnesses an authoritative, worker-local
                # assertion that a request selected the captured FULL graph;
                # it is deliberately outside the graph and has no CUDA work.
                self._beam_graph_full_dispatch_count = (
                    int(getattr(self, "_beam_graph_full_dispatch_count", 0)) + 1
                )
                result = (CUDAGraphMode.FULL, new_descriptor) + result[2:]

        return result

    setattr(patched_determine, _WORKER_DETERMINE_PATCH_MARKER, True)
    return patched_determine


# ---------------------------------------------------------------------------
# Beam decode CUDA Graph capture helpers
# ---------------------------------------------------------------------------

_BEAM_DUMMY_REQ_ID = "vllm-gr-beam-graph-dummy"


def _beam_graph_batch_descriptor(width: int):
    """Return the exact FULL-graph shape for one logical GR request.

    Physical Beam rows are tokens, not independent native requests.  vLLM's
    FULL graph contract requires ``num_reqs`` to be exact (FA scheduler
    metadata may specialize on it), so the previous ``None`` value was not a
    valid way to distinguish the Beam graph from native decode graphs.
    """

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor

    return BatchDescriptor(num_tokens=width, num_reqs=1, uniform=True)


def _create_beam_context(runner, gr_cfg, max_batch: int = 1):
    """Deprecated local alias — see model_runner_common._create_beam_context.

    Kept (delegating) for callers/tests importing it from this module.
    """
    return _common_create_beam_context(runner, gr_cfg, max_batch=max_batch)


def _capture_create_dummy_context(
    runner, gr_cfg, width: int, steps_upper: int, prefix_len: int
) -> tuple:
    """Create the BeamAttentionContext, dummy session, and beam_info for capture.

    Returns:
        ``(context, session, beam_info, batch_descriptor)`` — all needed by
        ``_dummy_beam_forward`` and the capture orchestration loop.
    """
    from vllm_gr.v1.beam.beam_search_session import BeamSearchSession

    context = _create_beam_context(runner, gr_cfg, max_batch=1)
    slot = context.registry.allocate(_BEAM_DUMMY_REQ_ID, beam_width=width, prefix_len=prefix_len)
    session = BeamSearchSession(session_id=_BEAM_DUMMY_REQ_ID, slot=slot, context=context)
    session.begin_session(width, prefix_len)

    runner._beam_context = context
    runner._beam_sessions = {_BEAM_DUMMY_REQ_ID: session}

    beam_data = {
        _BEAM_DUMMY_REQ_ID: {
            "prefix_len": prefix_len,
            "cache_len": prefix_len,
            "max_decode_steps": steps_upper,
            "is_beam_decode": True,
            "beam_width": width,
            "beam_decode_steps": 1,
            "parent_beam_ids": [0] * width,
            "decode_step": 0,
        }
    }
    beam_info = {
        "beam_data": beam_data,
        "req_ids": [_BEAM_DUMMY_REQ_ID],
        "beam_session": session,
    }

    batch_descriptor = _beam_graph_batch_descriptor(width)
    return context, session, beam_info, batch_descriptor


def _dummy_beam_forward(
    runner,
    context,
    beam_info,
    batch_descriptor,
    width: int,
    prefix_len: int,
    capture: bool,
) -> None:
    """Execute one dummy forward for beam-decode CUDA Graph capture (or warmup).

    This function mutates several runner buffers in a **sequence-sensitive**
    order.  The ordering constraints are:

    1. ``decode_step_buf.fill_`` — device scalar, must precede model forward.
    2. ``input_ids`` / ``positions`` — runner persistent GPU buffers, must
       precede the model call that reads them.
    3. ``optimistic_seq_lens_cpu`` → ``seq_lens.copy_`` — CPU write must
       complete before the H2D copy.
    4. ``query_start_loc`` — CPU write → H2D copy, same ordering.
    5. ``block_table.commit_block_table`` — must follow the seq_lens setup.
    6. ``slot_mappings`` fill — must precede ``_build_attention_metadata``.
    7. ``BEAM_DATA_VAR.set`` / ``reset`` — contextvars scope around the
       entire metadata + forward call.

    Args:
        runner: The GPU ModelRunner.
        context: The BeamAttentionContext (for ``decode_step_buf``).
        beam_info: The beam context dict to publish via ``BEAM_DATA_VAR``.
        batch_descriptor: The ``BatchDescriptor`` for this capture width.
        width: Beam width (== num_tokens for uniform decode).
        prefix_len: Dummy prefix length (arbitrary under host-param upper bounds).
        capture: ``True`` for the FULL-mode capture forward; ``False`` for
            eager warmup forwards (NONE mode — primes Triton JIT + FA2 shapes).
    """
    from vllm.forward_context import set_forward_context
    from vllm.v1.cudagraph_dispatcher import CUDAGraphMode

    from vllm_gr.v1.beam.context import BEAM_DATA_VAR

    # 1. Refresh the per-step device buffer (same as runtime path).
    context.decode_step_buf.fill_(0)

    # 2. Dummy inputs in the runner's persistent buffers (fixed addresses).
    input_ids = runner.input_ids.gpu[:width]
    input_ids.fill_(1)
    positions = runner.positions[:width]
    positions.copy_(
        torch.arange(prefix_len, prefix_len + width, dtype=positions.dtype, device=runner.device)
    )

    # 3. Dummy batch state, mirroring _dummy_run's bookkeeping:
    #    1 req with `width` scheduled tokens; seq_len = prefix_len + width.
    #    MUST: CPU write precedes the H2D copy below.
    runner.optimistic_seq_lens_cpu[:1] = prefix_len + width
    runner.optimistic_seq_lens_cpu[1:] = 0
    runner.seq_lens.copy_(runner.optimistic_seq_lens_cpu, non_blocking=True)

    # 4. MUST: CPU write precedes copy_to_gpu.
    runner.query_start_loc.np[0] = 0
    runner.query_start_loc.np[1] = width
    runner.query_start_loc.np[2:] = 0
    runner.query_start_loc.copy_to_gpu()

    # 5. Commit block table for 1 request.
    runner.input_batch.block_table.commit_block_table(1)

    # 6. Dummy runs have no real slot assignments — fill with -1 so
    #    concat_and_cache kernels skip the KV write (same as _dummy_run).
    slot_mappings_by_group, _ = runner._get_slot_mappings(
        num_tokens_padded=width,
        num_reqs_padded=1,
        num_tokens_unpadded=width,
    )
    if slot_mappings_by_group is not None:
        for sm in slot_mappings_by_group.values():
            sm.fill_(-1)

    # 7. Publish beam context and run the model forward.
    token = BEAM_DATA_VAR.set(beam_info)
    try:
        metadata, _ = runner._build_attention_metadata(
            num_tokens=width,
            num_reqs=1,
            max_query_len=width,
            num_tokens_padded=width,
            num_reqs_padded=1,
            for_cudagraph_capture=capture,
            slot_mappings=slot_mappings_by_group,
        )
        with set_forward_context(
            metadata,
            runner.vllm_config,
            num_tokens=width,
            cudagraph_runtime_mode=(CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE),
            batch_descriptor=batch_descriptor,
        ):
            runner.model(input_ids=input_ids, positions=positions)
    finally:
        BEAM_DATA_VAR.reset(token)


def _verify_and_record_captured_graph(runner, width: int) -> bool:
    """Walk CUDAGraphWrapper instances once to confirm a beam graph was captured.

    After capture, verify that vllm's CUDAGraphWrapper registered a graph for
    the beam ``BatchDescriptor(num_tokens=width, num_reqs=1, uniform=True)``.
    If found, record ``width`` in ``runner._beam_graph_keys`` for O(1) runtime
    lookups.

    Returns:
        ``True`` if a matching graph was found and recorded.
    """
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.cudagraph_dispatcher import CUDAGraphMode

    desc = _beam_graph_batch_descriptor(width)
    for wrapper in CUDAGraphWrapper._all_instances:
        if wrapper.runtime_mode != CUDAGraphMode.FULL:
            continue
        entry = wrapper.concrete_cudagraph_entries.get(desc)
        if entry is not None and entry.cudagraph is not None:
            # Record the captured width so _beam_full_graph_ready can look it
            # up directly (O(1)) without traversing vllm-internal state.
            beam_keys = getattr(runner, "_beam_graph_keys", None)
            if beam_keys is None:
                beam_keys = set()
                runner._beam_graph_keys = beam_keys
            beam_keys.add(width)
            return True
    return False


def _beam_graph_descriptor_is_free(width: int) -> bool:
    """Reject a descriptor already owned by an earlier native FULL graph.

    ``capture_beam_decode_graph`` runs after vLLM's native capture.  Calling a
    CUDAGraphWrapper with an occupied key replays that graph; it does not
    recapture it.  Treating the existing entry as our Beam graph would then
    silently route persistent Beam decode through native attention.
    """

    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.cudagraph_dispatcher import CUDAGraphMode

    descriptor = _beam_graph_batch_descriptor(width)
    for wrapper in CUDAGraphWrapper._all_instances:
        if wrapper.runtime_mode != CUDAGraphMode.FULL:
            continue
        entry = wrapper.concrete_cudagraph_entries.get(descriptor)
        if entry is not None and entry.cudagraph is not None:
            return False
    return True


def capture_beam_decode_graph(runner) -> bool:
    """Startup dummy capture of the beam-decode FULL CUDA graph.

    Runs once after ``capture_model()`` when the GRConfig beam graph is active
    (``beam_graph_enabled`` and ``beam_max_width`` set; CLI
    ``--beam-max-width``).  Mirrors
    vllm's own ``_warmup_and_capture`` pattern: N eager warmup forwards (mode
    NONE — primes the Triton JIT of cache_unshared_kv/copy_suffix_kv and FA2
    shape selection), then one FULL-mode forward which the CUDAGraphWrapper
    captures.  The captured graph's shape is
    (num_tokens=width, num_reqs=1, uniform=True);
    every runtime-varying quantity (decode_step, effective KV lengths, prefix
    extent, block-table contents) is device-side data or a fixed-address
    buffer value, so the single graph serves every decode step and every
    prompt length (Prefix FA host-side parameters use global upper bounds).

    The capture forward consumes dummy inputs and its output is discarded —
    exactly the native startup-capture contract, so no post-capture replay is
    needed.  Real requests replay only (``_beam_full_graph_ready`` gates FULL
    in patched_determine); uncaptured shapes fall back to eager.

    Returns True if a graph was captured.
    """
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    gr_cfg = _get_gr_config(runner)
    if not gr_cfg.beam_graph_active:
        return False

    # ── TP / DP ubatching guard ─────────────────────────────────
    # Beam capture runs on a side stream *after* native capture_model()
    # returns, bypassing vllm's graph_capture() context that provides
    # TP rank synchronisation.  With TP>1 the captured graphs would
    # diverge across ranks -> silent wrong results on replay.
    # With DP ubatching, the native _determine result[2:] (should_ubatch,
    # num_tokens_across_dp) is computed from the original token counts and
    # is NOT overridden by the beam patch -- replaying a beam graph with
    # stale ubatch metadata would produce incorrect micro-batch slices.
    # Fail fast rather than silently producing incorrect graphs.
    tp_size = runner.parallel_config.tensor_parallel_size
    if tp_size > 1:
        logger.warning(
            "vllm-gr: beam decode CUDA graph is not supported with "
            "tensor parallelism (TP=%d); beam decode will run eager",
            tp_size,
        )
        return False
    if runner.parallel_config.use_ubatching:
        logger.warning(
            "vllm-gr: beam decode CUDA graph is not supported with "
            "DP ubatching; beam decode will run eager",
        )
        return False
    # ─────────────────────────────────────────────────────────────

    # Re-entry guard: if a previous capture already produced a graph for this
    # width, do NOT rebuild the beam context (a new context moves every
    # buffer's data_ptr, so the existing graph would silently read/write the
    # OLD pool while the runtime refreshes the NEW one).
    width = int(gr_cfg.beam_max_width)
    if _beam_full_graph_ready(runner, width):
        return True
    if not _beam_graph_descriptor_is_free(width):
        logger.warning(
            "vllm-gr: Beam FULL-graph descriptor is already owned by a native "
            "graph (num_tokens=%d, num_reqs=1, uniform=True); Beam decode "
            "will run eager",
            width,
        )
        return False

    steps_upper = int(gr_cfg.beam_max_decode_steps)
    # Dummy-capture constants:
    # - prefix_len: fake prefix length for the dummy session.  Under the
    #   Prefix FA Host-side parameter global-upper-bound design, the captured
    #   graph's shape does not depend on the prefix length (FA2 reads the true
    #   extent from device-side seqused_k), so 1024 is arbitrary.
    # - num_warmups: two eager forwards prime the Triton JIT of
    #   cache_unshared_kv / copy_suffix_kv and FA2 shape selection before
    #   the FULL-mode capture forward.
    prefix_len = 1024
    num_warmups = 2

    # 1. Create the worker-level beam context + one dummy session.
    context, session, beam_info, batch_descriptor = _capture_create_dummy_context(
        runner, gr_cfg, width, steps_upper, prefix_len
    )

    # 2. Warmup + capture.
    captured = False
    try:
        set_cudagraph_capturing_enabled(True)
        # The graph pool's tensors are inference tensors (the native
        # ``_dummy_run`` creates them under ``@torch.inference_mode()``), so
        # ``capture_begin``'s in-place updates are only legal inside
        # inference mode.  Mirror the native decorator here.
        with torch.inference_mode():
            # CUDA graph capture requires a non-default stream (native
            # graph_capture() provides one during capture_model; emulate here).
            side = torch.cuda.Stream()
            side.wait_stream(torch.accelerator.current_stream())
            with torch.cuda.stream(side):
                for _ in range(num_warmups):
                    _dummy_beam_forward(
                        runner,
                        context,
                        beam_info,
                        batch_descriptor,
                        width,
                        prefix_len,
                        capture=False,
                    )
                _dummy_beam_forward(
                    runner,
                    context,
                    beam_info,
                    batch_descriptor,
                    width,
                    prefix_len,
                    capture=True,
                )
            torch.accelerator.current_stream().wait_stream(side)
            torch.accelerator.synchronize()

            # 3. Verify capture succeeded and record the width.
            captured = _verify_and_record_captured_graph(runner, width)
    finally:
        set_cudagraph_capturing_enabled(False)
        # Drop the dummy session even if capture threw. The context and
        # pools stay — a captured graph (if any) references their addresses.
        session.end_session()
        runner._beam_sessions.pop(_BEAM_DUMMY_REQ_ID, None)
        runner._current_beam_info = None

    if captured:
        logger.info(
            "vllm-gr: startup dummy capture of beam decode graph succeeded "
            "(num_tokens=%d, num_reqs=1, uniform=True); runtime is replay-only",
            width,
        )
    else:
        logger.warning(
            "vllm-gr: beam dummy capture did NOT produce a graph for "
            "num_tokens=%d; beam decode will run eager",
            width,
        )
    return captured


def discover_worker_runners() -> tuple[tuple[str, type], ...]:
    """Discover the upstream GPU ModelRunner without importing NPU runtime."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as exc:
        logger.warning(
            "Failed to import vllm.v1.worker.gpu_model_runner.GPUModelRunner; "
            "skipping vllm-gr GPU beam-search runner patches for this attempt. "
            "ImportError: %s",
            exc,
        )
        logger.debug("GPUModelRunner import traceback", exc_info=True)
        return ()
    return (("GPUModelRunner", GPUModelRunner),)


make_patched_prepare_inputs = make_patched_prepare_inputs_gpu
make_patched_execute_model = make_patched_execute_model_gpu
make_patched_bookkeeping_sync = make_patched_bookkeeping_sync_gpu
make_patched_sample = make_patched_sample_gpu
make_patched_sample_tokens = make_patched_sample_tokens_gpu
make_patched_determine_batch_execution_and_padding = (
    make_patched_determine_batch_execution_and_padding_gpu
)
