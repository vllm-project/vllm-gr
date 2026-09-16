# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU V1 adapter for asynchronous scheduling dispatches and persistent Beam state.

Bindings are views of the existing Worker session registry. The only supported
execution is one item on TP=1. Resource rows and physical model rows are explicit
so expanding storage capacity does not implicitly enable concurrent scratch use.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Any, cast

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    DEFAULT_CONSTRAINT_RESOURCE_ID,
    BeamBatchFailure,
    BeamBatchResult,
)
from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspaceKey
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRExecutionCapacity,
    GRRetireMetadata,
    GRRetireResult,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
)
from vllm_gr.v1.worker.beam_search_runner import BeamSearchRunner
from vllm_gr.v1.worker.gr_inputs import (
    GRInputBuffers,
    PreparedGRInputs,
    bind_device_inputs,
)

METADATA_ABI = 1
CONTROL_CAPACITY = 2


@dataclass(frozen=True)
class BeamRequestBinding:
    session_id: str
    item_index: int
    native_request_id: str
    state_row: int
    kv_slot: int
    workspace_row: int


@dataclass(frozen=True)
class BeamDispatchEntry:
    binding: BeamRequestBinding
    metadata: GRStageMetadata
    row_offset: int
    row_count: int


@dataclass(frozen=True)
class BeamDispatchExecution:
    entries: tuple[BeamDispatchEntry, ...]
    num_model_rows: int
    metadata_abi: int = METADATA_ABI


def validate_dispatch(
    scheduler_output: Any,
    *,
    previous: GRStageMetadata | None = None,
    binding: BeamRequestBinding | None = None,
) -> tuple[GRStageMetadata, ...]:
    """Validate the whole metadata map before any Worker resource mutation."""
    mapping = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("GR dispatch requires a nonempty metadata mapping")
    scheduled = scheduler_output.num_scheduled_tokens
    if set(mapping) != set(scheduled) or getattr(scheduler_output, "beam_data", {}):
        raise ValueError("GR dispatch cannot mix ordinary or legacy Beam requests")
    if getattr(scheduler_output, "total_num_scheduled_tokens", sum(scheduled.values())) != sum(
        scheduled.values()
    ):
        raise ValueError("GR dispatch total disagrees with native scheduled counts")
    # Central admission boundary. Other code operates on explicit bindings.
    if len(mapping) != 1:
        raise ValueError("persistent GPU Beam currently requires B=1")
    stages = tuple(mapping.values())
    for key, m in mapping.items():
        if not isinstance(m, GRStageMetadata) or key != m.session_id:
            raise ValueError("invalid GR metadata identity")
        for number in (m.dispatch_id, m.stage_index, m.num_prompt_tokens, m.owner_generation):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(
                    "GR dispatch indices and prompt length must be nonnegative integers"
                )
        if not m.num_prompt_tokens or not isinstance(m.produces_output, bool):
            raise ValueError("GR requires a nonempty prompt and boolean produces_output")
        new_requests = {r.req_id: r for r in getattr(scheduler_output, "scheduled_new_reqs", ())}
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        computed = None
        if key in new_requests:
            native = new_requests[key]
            if (
                m.kind is GRStageKind.DECODE
                or len(native.prompt_token_ids or ()) != m.num_prompt_tokens
            ):
                raise ValueError("GR native request identity or prompt geometry changed")
            computed = native.num_computed_tokens
        elif cached is not None and key in cached.req_ids:
            if key in cached.resumed_req_ids and m.stage_index > 0:
                raise ValueError("GR rejects recompute after generation")
            computed = cached.num_computed_tokens[cached.req_ids.index(key)]
        if computed is None and (new_requests or cached is not None):
            raise ValueError("GR metadata has no matching native request")
        if computed is not None:
            if m.kind is GRStageKind.PREFILL:
                if not (0 <= computed < m.num_prompt_tokens) or m.produces_output != (
                    computed + scheduled[key] == m.num_prompt_tokens
                ):
                    raise ValueError(
                        "GR Prefill chunk output flag disagrees with native token extent"
                    )
            elif computed < m.num_prompt_tokens:
                raise ValueError("GR rejects recompute after generation")
        if (
            not isinstance(m.is_last_stage, bool)
            or (m.decode_step is not None and type(m.decode_step) is not int)
            or type(scheduled[key]) is not int
            or scheduled[key] <= 0
        ):
            raise ValueError("GR dispatch flags, step and scheduled count have invalid types")
        params = m.beam_params
        if m.execution_capacity is not None:
            m.execution_capacity.validate(params)
        if (
            type(params.beam_width) is not int
            or not 1 <= params.beam_width <= 256
            or type(params.max_tokens) is not int
            or params.max_tokens not in (1, 2, 3)
            or not math.isfinite(params.temperature)
            or params.temperature < 0
            or params.constraint_resource_id != DEFAULT_CONSTRAINT_RESOURCE_ID
        ):
            raise ValueError("GR requires fixed W<=256, effective N=1/2/3 and CUDA constraints")
        if not 0 <= m.stage_index < params.max_tokens:
            raise ValueError("GR logical stage exceeds generation budget")
        if m.output_device_state != GRDeviceStateRef(key, m.stage_index):
            raise ValueError("GR output device-state identity mismatch")
        if m.kind is GRStageKind.PREFILL:
            if m.stage_index != 0 or m.decode_step is not None or m.input_device_state is not None:
                raise ValueError("GR rejects recompute after generation")
        elif m.kind is GRStageKind.DECODE:
            if (
                not m.produces_output
                or m.decode_step != m.stage_index - 1
                or m.input_device_state != GRDeviceStateRef(key, m.stage_index - 1)
                or scheduled[key] != 1
            ):
                raise ValueError("invalid GR Decode dependency or logical token count")
        else:
            raise ValueError("unknown GR stage kind")
        if m.is_last_stage != (m.produces_output and m.stage_index == params.max_tokens - 1):
            raise ValueError("GR last-stage flag disagrees with effective budget")
        if m.session_init is not None:
            request = m.session_init
            if (
                m.kind is not GRStageKind.PREFILL
                or request.protocol_version != BEAM_BATCH_PROTOCOL_VERSION
                or len(request.items) != 1
                or request.batch_id != key
                or type(request.items[0].item_index) is not int
                or request.items[0].item_index != 0
                or request.items[0].request.request_id != key
                or len(request.items[0].request.prompt_token_ids or ()) != m.num_prompt_tokens
                or request.beam_params != params
                or request.output_options != m.output_options
            ):
                raise ValueError("GR session_init disagrees with dispatch metadata")
        if binding is not None and key != binding.session_id:
            raise ValueError("a second active GR session cannot use shared execution scratch")
        if previous is None:
            if m.dispatch_id != 0 or m.kind is not GRStageKind.PREFILL:
                raise ValueError("GR session must start with dispatch 0 Prefill")
        elif (
            m.dispatch_id != previous.dispatch_id + 1
            or m.stage_index != previous.stage_index + int(previous.produces_output)
            or m.owner_generation != previous.owner_generation
            or params != previous.beam_params
            or m.output_options != previous.output_options
            or m.num_prompt_tokens != previous.num_prompt_tokens
        ):
            raise ValueError("GR dispatch order, configuration or prompt binding changed")
    return stages


class AsyncGPUBeamOutput(AsyncModelRunnerOutput):  # type: ignore[misc]
    """Only the executor's output consumer waits and constructs the IPC value."""

    def __init__(self, owner: GPUBeamStageRunner, execution: BeamDispatchExecution, slot: int):
        self.owner = owner
        self.execution = execution
        self.slot = slot
        self._output: ModelRunnerOutput | None = None
        self._lock = Lock()

    def get_output(self) -> ModelRunnerOutput:
        with self._lock:
            if self._output is not None:
                return self._output
            # CUDA/D2H failures are fatal and must reach the native Executor
            # failure channel. Never publish a successful release after one.
            try:
                self.owner.copy_done[self.slot].synchronize()
                controls = self.owner.host_control[self.slot].tolist()
                self._output = self._convert(controls)
            except (ValueError, TypeError, IndexError) as error:
                self._output = self._conversion_failure(error)
            finally:
                with self.owner.control_drained:
                    self.owner.control_busy[self.slot] = False
                    self.owner.control_drained.notify_all()
            return self._output

    def _conversion_failure(self, error: Exception) -> ModelRunnerOutput:
        out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
        results = {}
        for entry in self.execution.entries:
            m = entry.metadata
            out.req_id_to_index[m.session_id] = len(out.req_ids)
            out.req_ids.append(m.session_id)
            out.sampled_token_ids.append([])
            terminal = BeamBatchResult(
                BEAM_BATCH_PROTOCOL_VERSION,
                m.session_id,
                None,
                BeamBatchFailure("worker_output_error", str(error), entry.binding.item_index),
            )
            results[m.session_id] = GRWorkerResult(
                m.session_id, m.dispatch_id, m.stage_index, 0, m.output_device_state, True, terminal
            )
        setattr(out, GR_WORKER_RESULT_ATTR, results)
        return out

    def _convert(self, controls) -> ModelRunnerOutput:
        if len(controls) != len(self.execution.entries):
            raise ValueError("Asynchronous scheduling control snapshot has the wrong row count")
        req_ids, sampled, results = [], [], {}
        for entry, (produced, finished, error) in zip(self.execution.entries, controls):
            m = entry.metadata
            terminal = None
            if error:
                terminal = BeamBatchResult(
                    protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
                    batch_id=m.session_id,
                    output=None,
                    failure=BeamBatchFailure(
                        "beam_device_error",
                        f"Beam device error {error}",
                        entry.binding.item_index,
                    ),
                )
            req_ids.append(entry.binding.native_request_id)
            # These are scheduler placeholders. They must never be used as
            # model inputs or published as generated Beam sequences.
            sampled.append([0] if produced else [])
            results[m.session_id] = GRWorkerResult(
                m.session_id,
                m.dispatch_id,
                m.stage_index,
                produced,
                m.output_device_state,
                bool(finished),
                terminal,
            )
        out = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
        )
        setattr(out, GR_WORKER_RESULT_ATTR, results)
        return out


class GPUBeamStageRunner:
    """One exclusive execution owner, using the existing session/state/workspace."""

    def __init__(
        self,
        worker: Any,
        beam: BeamSearchRunner,
        binding: BeamRequestBinding,
        backend: CUDAConstraintBackend,
    ):
        if (
            binding.session_id != beam.session.session_id
            or binding.kv_slot != beam.session.slot
            or binding.state_row != beam.state_index
            or binding.workspace_row != beam.workspace_index
            or binding.item_index != 0
        ):
            raise ValueError("GR binding disagrees with existing session resources")
        context = beam.session.context
        if (
            getattr(context, "_gr_execution_owner", None) is not None
            or beam.workspace.execution_owner is not None
        ):
            raise RuntimeError("Beam context already has an execution owner")
        self.worker, self.beam, self.binding, self.backend = worker, beam, binding, backend
        self.previous: GRStageMetadata | None = None
        self.pending: tuple[BeamDispatchExecution, int] | None = None
        self.control_lock = Lock()
        self.control_drained = Condition(self.control_lock)
        self.control_busy = [False] * CONTROL_CAPACITY
        self.device_control = torch.empty(
            (CONTROL_CAPACITY, 1, 3), device=beam.state.device, dtype=torch.int32
        )
        self.host_control = torch.empty(
            (CONTROL_CAPACITY, 1, 3), pin_memory=True, dtype=torch.int32
        )
        self.produced_ready = [torch.Event() for _ in range(CONTROL_CAPACITY)]
        self.copy_done = [torch.Event() for _ in range(CONTROL_CAPACITY)]
        self.copy_stream = torch.cuda.Stream(device=beam.state.device)
        self.last_work = torch.Event()
        self.closed = False
        self.cancelled = False
        self.execution_error: Exception | None = None
        self.inputs: GRInputBuffers | None = None
        self.prepared: PreparedGRInputs | None = None
        self.metadata: Any = None
        self.graph: Any = None
        self.logits: torch.Tensor | None = None
        self.input_ids = torch.empty(
            beam.state.beam_width, device=beam.state.device, dtype=torch.int32
        )
        self.positions = torch.empty_like(self.input_ids, dtype=torch.int64)
        self.slots = torch.full_like(self.positions, -1)
        self.execution_mask = torch.empty_like(self.input_ids, dtype=torch.bool)
        self.constraint_steps = beam.state.num_generated_tokens[
            binding.state_row : binding.state_row + 1
        ]
        w = beam.workspace
        self.prefixes = w.prefix_token_ids[binding.workspace_row]
        self.temperatures = w.temperatures[binding.workspace_row]
        self.tokens = w.topk_token_ids[binding.workspace_row]
        self.scores = w.topk_scores[binding.workspace_row]
        self.valid = w.topk_valid_mask[binding.workspace_row]
        self.counts = w.topk_valid_counts[binding.workspace_row]
        context._gr_execution_owner = self
        beam.workspace.execution_owner = self

    def admit(self, scheduler_output: Any) -> BeamDispatchExecution:
        if self.closed:
            raise RuntimeError("GR session is closed")
        if (
            not self.backend.is_ready
            or self.beam.workspace.execution_owner is not self
            or self.beam.session.context._gr_execution_owner is not self
        ):
            raise RuntimeError("GR backend or shared execution ownership changed")
        if self.pending is not None:
            raise RuntimeError("sample_tokens must consume the previous GR execute state")
        stages = validate_dispatch(scheduler_output, previous=self.previous, binding=self.binding)
        for m in stages:
            state, params = self.beam.state, m.beam_params
            if (
                state.beam_width != params.beam_width
                or state.max_new_tokens != params.max_tokens
                or state.stop_token_ids_config != params.stop_token_ids
                or state.ignore_stop_tokens_config != params.ignore_stop_tokens
            ):
                raise ValueError("GR metadata configuration does not match persistent Beam state")
        entries, offset = [], 0
        for m in stages:
            count = (
                m.beam_params.beam_width
                if m.kind is GRStageKind.DECODE
                else scheduler_output.num_scheduled_tokens[m.session_id]
            )
            entries.append(BeamDispatchEntry(self.binding, m, offset, count))
            offset += count
        with self.control_lock:
            slot = next((i for i, busy in enumerate(self.control_busy) if not busy), None)
            if slot is None:
                raise RuntimeError("both GR dispatch control slots await output consumption")
            self.control_busy[slot] = True
        execution = BeamDispatchExecution(tuple(entries), offset)
        self.pending = execution, slot
        self.previous = stages[0]  # B=1 admission bridge, not a model/state row.
        torch.cuda.current_stream(self.beam.state.device).wait_event(self.last_work)
        self.device_control[slot, 0, :1].copy_(
            self.beam.state.num_generated_tokens[
                self.binding.state_row : self.binding.state_row + 1
            ]
        )
        return execution

    def prepare_cpu(self, m: GRStageMetadata, slot: int) -> PreparedGRInputs:
        worker = self.worker
        groups = worker.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise RuntimeError("Asynchronous scheduling requires one uniform KV cache group")
        if self.inputs is None:
            self.inputs = GRInputBuffers(
                self.beam.state.beam_width,
                worker.model_config.max_model_len,
                groups[0].kv_cache_spec.block_size,
                self.beam.state.device,
            )
        native = worker.requests[self.binding.native_request_id]
        return self.inputs.prepare_cpu(m, self.binding, block_ids=native.block_ids, slot=slot)

    def _bind_attention(self, m: GRStageMetadata) -> None:
        from vllm_gr.v1.attention.backends.beam_attn_metadata import (
            BeamAttentionMetadataBuilder,
        )

        if self.prepared is None or self.prepared.metadata != m or self.inputs is None:
            raise RuntimeError(
                "Asynchronous scheduling attention requires this dispatch's CPU preparation"
            )
        worker = self.worker
        views = self.inputs.upload(self.prepared)
        self.device_positions = views["positions"]
        self.metadata = BeamAttentionMetadataBuilder.build_async(
            layout=self.prepared.layout,
            views=views,
            session=self.beam.session,
            mask=self.execution_mask,
            slots=self.slots,
            block_size=self.inputs.block_size,
            max_model_len=self.inputs.max_model_len,
            num_heads=worker.model_config.get_num_attention_heads(worker.parallel_config),
            num_kv_heads=worker.model_config.get_num_kv_heads(worker.parallel_config),
        )
        names = tuple(
            name for group in worker.kv_cache_config.kv_cache_groups for name in group.layer_names
        )
        self.attention = dict.fromkeys(names, self.metadata)
        self.slot_mappings = dict.fromkeys(names, self.slots)

    def prepare_decode(self, entry: BeamDispatchEntry) -> None:
        # Device binding has no dependency on Host output consumption.
        prepared = self.prepared
        if (
            prepared is None
            or prepared.metadata != entry.metadata
            or prepared.binding != entry.binding
        ):
            raise RuntimeError(
                "Asynchronous scheduling device binding does not match the prepared dispatch"
            )
        bind_device_inputs(prepared, self)

    def capture_forward(self) -> None:
        """Capture the shared Beam Forward over this interface's fixed inputs."""
        from vllm_gr.v1.worker.beam_decode_graph import get_beam_decode_graph

        graph = get_beam_decode_graph(self.worker)
        graph.capture(self._forward_hidden)
        self.graph = graph

    def _forward(self) -> torch.Tensor:
        return self.worker.model.compute_logits(self._forward_hidden(False))

    def _forward_hidden(self, capture: bool) -> torch.Tensor:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.cudagraph_dispatcher import BatchDescriptor

        worker = self.worker
        with set_forward_context(
            self.attention,
            worker.vllm_config,
            num_tokens=self.input_ids.numel(),
            cudagraph_runtime_mode=CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=self.input_ids.numel(), uniform=True),
            slot_mapping=self.slot_mappings,
        ):
            return worker._model_forward(
                input_ids=self.input_ids,
                positions=self.positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )

    def execute(
        self, scheduler_output: Any, original: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        previous_kind = self.previous.kind if self.previous is not None else None
        execution = self.admit(scheduler_output)
        worker = self.worker
        worker._current_scheduler_output = scheduler_output
        worker._current_beam_data = {}
        worker._current_beam_info = None
        worker._gr_prepared_cuda_prefill_constraint = None
        worker._gr_prelaunched_cuda_prefill = None
        for entry in execution.entries:
            m = entry.metadata
            if m.kind is GRStageKind.PREFILL:
                self.temperatures.fill_(m.beam_params.temperature)
                self.prefixes.fill_(-1)
                out = original(worker, scheduler_output, *args, **kwargs)
                if out is not None or worker.execute_model_state is None:
                    raise RuntimeError(
                        "native GR Prefill must return logits through execute_model_state"
                    )
                self.logits = worker.execute_model_state.logits
            else:
                # Pure CPU work can run before the native Prefill staging fence.
                self.prepared = self.prepare_cpu(m, self.pending[1])
                if previous_kind is GRStageKind.PREFILL:
                    # Native Prefill asynchronously reads pinned InputBatch
                    # buffers that _update_states mutates on the CPU. A stream
                    # wait on last_work cannot protect those Host writes.
                    event = worker.prepare_inputs_event
                    if event is not None:
                        event.synchronize()
                # Decode does not upload these native input buffers again, so
                # subsequent Decode dispatches need no Host input-prep wait.
                worker._update_states(scheduler_output)
                self._bind_attention(m)
                self.prepare_decode(entry)
                from vllm_gr.v1.worker.model_runner_common import _get_gr_config

                graph_enabled = (
                    not worker.model_config.enforce_eager
                    and _get_gr_config(worker).beam_graph_enabled
                )
                if graph_enabled:
                    if self.graph is None:
                        raise RuntimeError(
                            "Asynchronous scheduling graph mode requires successful startup capture"
                        )
                    self.logits = worker.model.compute_logits(self.graph.replay())
                else:
                    self.logits = self._forward()
        return None

    def sample(self, grammar_output: Any = None) -> AsyncGPUBeamOutput:
        if self.execution_error is not None:
            raise RuntimeError(
                "Asynchronous scheduling execute_model failed"
            ) from self.execution_error
        if grammar_output is not None or self.pending is None:
            raise RuntimeError("GR sample requires a pending dispatch and no grammar output")
        if self.backend is None:
            raise RuntimeError("GR sampling requires a bound constraint backend")
        execution, slot = self.pending
        for entry in execution.entries:
            m = entry.metadata
            if m.produces_output and not self.cancelled:
                if self.logits is None:
                    raise RuntimeError("GR output stage has no native model logits")
                rows = 1 if m.kind is GRStageKind.PREFILL else entry.row_count
                candidates = self.tokens[:rows], self.scores[:rows], self.valid[:rows]
                self.backend.select_out_unchecked(
                    self.logits,
                    ConstraintMetadata(
                        self.prefixes[:rows], self.temperatures[:rows], 0, m.beam_params.beam_width
                    ),
                    out_token_ids=candidates[0],
                    out_logprobs=candidates[1],
                    out_valid_mask=candidates[2],
                    out_valid_counts=self.counts[:rows],
                    device_step=self.constraint_steps,
                )
                if m.kind is GRStageKind.PREFILL:
                    self.beam.initialize_from_prefill(*candidates)
                else:
                    self.beam.advance(*candidates)
            row = slice(entry.binding.state_row, entry.binding.state_row + 1)
            control = self.device_control[slot, 0]
            control[:1].copy_(self.beam.state.num_generated_tokens[row] - control[:1])
            control[1:2].copy_(
                self.beam.state.is_finished[row] | (self.beam.state.error_code[row] != 0)
            )
            control[2:3].copy_(self.beam.state.error_code[row])
        stream = torch.cuda.current_stream(self.beam.state.device)
        self.produced_ready[slot].record(stream)
        self.last_work.record(stream)
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.produced_ready[slot])
            self.host_control[slot].copy_(self.device_control[slot], non_blocking=True)
            self.copy_done[slot].record(self.copy_stream)
        self.worker.execute_model_state = None
        self.worker.kv_connector_output = None
        self.pending = None
        return AsyncGPUBeamOutput(self, execution, slot)

    def completion_view(self, binding: BeamRequestBinding) -> dict[str, torch.Tensor]:
        if self.closed or binding != self.binding:
            raise RuntimeError("completion view requires a live matching binding")
        if self.pending is not None:
            raise RuntimeError("completion view requires the current dispatch to be sampled")
        torch.cuda.current_stream(self.beam.state.device).wait_event(self.last_work)
        return {
            name: value[binding.state_row : binding.state_row + 1]
            for name, value in vars(self.beam.state).items()
            if isinstance(value, torch.Tensor)
        }

    def request_release(self, metadata: GRRetireMetadata) -> AsyncGRRetireOutput:
        if (
            self.previous is None
            or metadata.session_id != self.binding.session_id
            or metadata.owner_generation != self.previous.owner_generation
            or self.pending is not None
        ):
            raise RuntimeError(
                "Asynchronous scheduling retirement identity does not match the drained execution"
            )
        self.cancel(self.binding)
        return AsyncGRRetireOutput(self, metadata)

    def cancel(self, binding: BeamRequestBinding) -> None:
        if binding != self.binding:
            raise ValueError("cancel binding mismatch")
        if self.closed or self.cancelled:
            return
        if self.pending is not None:
            raise RuntimeError("sample the dispatched work before cancellation")
        stream = torch.cuda.current_stream(self.beam.state.device)
        stream.wait_event(self.last_work)
        row = binding.state_row
        self.beam.state.is_finished[row : row + 1].fill_(True)
        self.beam.state.active_mask[row : row + 1].zero_()
        self.last_work.record(stream)
        self.cancelled = True

    def release(self, binding: BeamRequestBinding, *, consumer_done: torch.Event) -> None:
        if binding != self.binding:
            raise ValueError("release binding mismatch")
        if self.closed:
            return
        with self.control_lock:
            if self.pending is not None or any(self.control_busy):
                raise RuntimeError("GR release requires dispatch and output-consumer drain")
        torch.cuda.current_stream(self.beam.state.device).wait_event(self.last_work)
        self.beam.end_session(consumer_done=consumer_done)
        # Keep fixed-address graph storage for compatible, fully drained reuse.
        self.closed = True
        self.beam.session.context._gr_execution_owner = None
        self.beam.workspace.execution_owner = None
        self.worker._beam_sessions.pop(binding.session_id, None)


class AsyncGRRetireOutput(AsyncModelRunnerOutput):
    """Release proof is constructed only after GPU work and output consumers drain."""

    def __init__(self, owner, metadata):
        self.owner, self.metadata = owner, metadata
        self._lock = Lock()
        self._output = None

    def get_output(self):
        with self._lock:
            if self._output is not None:
                return self._output
            owner = self.owner
            with owner.control_drained:
                owner.control_drained.wait_for(lambda: not any(owner.control_busy))
            done = torch.Event()
            done.record(torch.cuda.current_stream(owner.beam.state.device))
            owner.release(owner.binding, consumer_done=done)
            if not owner.closed or owner.beam.workspace.execution_owner is owner:
                raise RuntimeError("Asynchronous scheduling Worker release did not complete")
            m = self.metadata
            out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
            setattr(
                out,
                GR_RETIRE_RESULT_ATTR,
                GRRetireResult(m.session_id, m.owner_generation, m.retire_id, True),
            )
            self._output = out
            return out


def execute_gr_retirement(worker, scheduler_output):
    metadata = getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR)
    if (
        not isinstance(metadata, GRRetireMetadata)
        or scheduler_output.total_num_scheduled_tokens
        or scheduler_output.num_scheduled_tokens
        or getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
        or metadata.session_id not in scheduler_output.finished_req_ids
    ):
        raise ValueError("invalid Asynchronous scheduling retirement control frame")
    previous = getattr(worker, "_gr_retirement_output", None)
    if previous is not None and previous.metadata == metadata:
        return previous
    owner = getattr(worker, "_gr_gpu_stage_runner", None)
    if owner is None or owner.closed:
        raise RuntimeError("Asynchronous scheduling retirement has no matching live Worker owner")
    result = owner.request_release(metadata)
    # Preserve the native finished_req_ids cleanup entry, without Forward or sampling.
    event = worker.prepare_inputs_event
    if event is not None:
        event.synchronize()
    worker._update_states(scheduler_output)
    worker._gr_retirement_output = result
    return result


def get_stage_runner(worker: Any, scheduler_output: Any) -> GPUBeamStageRunner:
    """Admit first, then derive the adapter from existing Worker resources."""
    from vllm_gr.config import BeamExecutionMode
    from vllm_gr.config.execution import require_beam_execution_mode

    require_beam_execution_mode(worker.vllm_config, BeamExecutionMode.V1)
    existing = getattr(worker, "_gr_gpu_stage_runner", None)
    if existing is not None and not existing.closed:
        validate_dispatch(scheduler_output, previous=existing.previous, binding=existing.binding)
        if getattr(worker, "constraint_backend", None) is not existing.backend:
            raise ValueError("GR constraint backend binding changed before session release")
        return cast(GPUBeamStageRunner, existing)
    stages = validate_dispatch(scheduler_output)
    (m,) = stages
    from vllm_gr.v1.worker.model_runner_common import (
        _create_beam_context,
        _get_gr_config,
    )

    cfg = _get_gr_config(worker)
    capacity = GRExecutionCapacity(cfg.beam_max_width)
    capacity.validate(m.beam_params)
    if m.execution_capacity is not None and m.execution_capacity != capacity:
        raise ValueError("Scheduler and Worker Asynchronous scheduling execution capacities differ")
    parallel = worker.parallel_config
    if (
        torch.device(worker.device).type != "cuda"
        or not worker.use_async_scheduling
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or getattr(parallel, "decode_context_parallel_size", 1) != 1
        or getattr(parallel, "prefill_context_parallel_size", 1) != 1
        or worker.speculative_config is not None
        or worker.model_config.dtype not in (torch.float16, torch.bfloat16)
        or cfg.beam_max_width != m.beam_params.beam_width
        or cfg.beam_max_decode_steps < m.beam_params.max_tokens
    ):
        raise ValueError(
            "persistent GR requires GPU V1 async, TP/PP/DP=1, FP16/BF16 and matching capacity"
        )
    backend = getattr(worker, "constraint_backend", None)
    if not isinstance(backend, CUDAConstraintBackend):
        raise ValueError("persistent GR requires a loaded canonical CUDA Constraint Table")
    if (
        existing is not None
        and existing.graph is not None
        and existing.backend is not None
        and existing.backend is not backend
    ):
        raise ValueError(
            "Asynchronous scheduling captured constraint/model binding cannot change after startup"
        )
    if (
        not backend.is_ready
        or backend.info.table_metadata.vocab_size != worker.model_config.get_vocab_size()
        or any(table.device != worker.device for table in backend._tables())
    ):
        raise ValueError("constraint backend is not ready or its model/device binding differs")
    if backend.info.max_rows < cfg.beam_max_width or backend.info.max_top_k < cfg.beam_max_width:
        raise ValueError("constraint backend capacity is smaller than fixed Beam width")
    if (
        worker.model_config.is_encoder_decoder
        or worker.model_config.is_hybrid
        or worker.supports_mm_inputs
        or getattr(worker, "lora_config", None) is not None
        or getattr(worker.vllm_config, "kv_transfer_config", None) is not None
    ):
        raise ValueError("persistent GR requires a text decoder without LoRA or KV transfer")
    if len(worker.kv_cache_config.kv_cache_groups) != 1:
        raise ValueError("persistent GR requires one uniform KV group")
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

    layers = worker.compilation_config.static_forward_context
    for group in worker.kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            impl = getattr(layers.get(name), "impl", None)
            if not isinstance(impl, BeamAttentionImpl):
                raise ValueError("persistent GR requires Beam Attention on every KV layer")
            # Packed beam queries do not preserve sliding-window positions
            # across the prefix/suffix split. No window is stored as (-1, -1).
            if impl.sliding_window != (-1, -1):
                raise ValueError(
                    "persistent GR split attention does not support sliding-window attention: "
                    f"{name} (window={impl.sliding_window})"
                )
    head_size = worker.model_config.get_head_size()
    if (
        head_size <= 0
        or head_size & (head_size - 1)
        or getattr(worker.model_config, "quantization", None) is not None
        or worker.cache_config.cache_dtype
        not in ("auto", str(worker.model_config.dtype).removeprefix("torch."))
    ):
        raise ValueError(
            "persistent GR requires unquantized FP16/BF16 KV and a power-of-two head size"
        )
    if getattr(worker, "_beam_postprocess_workspaces", None) is None:
        raise ValueError("persistent GR requires prewarmed Beam postprocess workspace")
    if getattr(worker, "_beam_sessions", {}):
        raise ValueError("another Beam session owns the Worker")
    workspace = worker._beam_postprocess_workspaces.require(
        BeamPostprocessWorkspaceKey(
            cfg.beam_max_width, cfg.beam_max_width, worker.model_config.dtype
        )
    )
    context = getattr(worker, "_beam_context", None)
    if context is None:
        context = _create_beam_context(worker, cfg)
        worker._beam_context = context
    if (
        context.registry.num_active_sessions
        or getattr(context, "_gr_execution_owner", None) is not None
        or workspace.execution_owner is not None
    ):
        raise ValueError("another Beam execution owns the registry or shared workspace")
    if (
        context.max_batch != workspace.max_batch
        or context.max_beams < m.beam_params.beam_width
        or context.max_decode_steps < m.beam_params.max_tokens
    ):
        raise ValueError("existing Beam context and workspace capacity are incompatible")
    BeamSearchRunner.validate_resources(
        context,
        workspace,
        beam_width=m.beam_params.beam_width,
        max_new_tokens=m.beam_params.max_tokens,
        kv_slot=0,
    )
    reuse = (
        existing is not None
        and existing.closed
        and existing.beam.state.beam_width == m.beam_params.beam_width
        and existing.beam.session.context is context
        and existing.beam.workspace is workspace
    )
    if reuse and existing.previous is not None and existing.previous.beam_params == m.beam_params:
        state = existing.beam.state
        state.reset(existing.binding.state_row)
    else:
        state = BeamSearchState(
            context.decision,
            beam_width=m.beam_params.beam_width,
            max_new_tokens=m.beam_params.max_tokens,
            stop_token_ids=m.beam_params.stop_token_ids,
            ignore_stop_tokens=m.beam_params.ignore_stop_tokens,
        )
    slot = context.registry.allocate(m.session_id, m.beam_params.beam_width, m.num_prompt_tokens)
    session = BeamSearchSession(m.session_id, slot, context)
    session.begin_session(m.beam_params.beam_width, m.num_prompt_tokens, decision_index=0)
    binding = BeamRequestBinding(m.session_id, 0, m.session_id, 0, slot, 0)
    beam = BeamSearchRunner(session, state, workspace)
    if reuse:
        assert existing is not None
        adapter = existing
        adapter.beam, adapter.binding, adapter.backend = beam, binding, backend
        adapter.previous, adapter.pending = None, None
        adapter.closed, adapter.cancelled = False, False
        adapter.execution_error = None
        adapter.prepared = None
        adapter.constraint_steps = state.num_generated_tokens[
            binding.state_row : binding.state_row + 1
        ]
        context._gr_execution_owner = adapter
        workspace.execution_owner = adapter
    else:
        adapter = GPUBeamStageRunner(worker, beam, binding, backend)
    sessions = getattr(worker, "_beam_sessions", {})
    sessions[m.session_id] = session
    worker._beam_sessions = sessions
    worker._gr_gpu_stage_runner = adapter
    return cast(GPUBeamStageRunner, adapter)
