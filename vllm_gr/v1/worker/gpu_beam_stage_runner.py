# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU V1 adapter for asynchronous scheduling dispatches and persistent Beam state.

Bindings are views of the existing Worker session registry. The only supported
execution is one item on TP=1. Resource rows and physical model rows are explicit
so expanding storage capacity does not implicitly enable concurrent scratch use.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Condition, Lock
from typing import Any, cast

import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend
from vllm_gr.ops.cuda.unconstrained_topk import (
    UnconstrainedTopKScratch,
    allocate_unconstrained_topk_scratch,
    unconstrained_topk_out,
)
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    SUPPORTED_BEAM_BATCH_SIZES,
    BeamBatchFailure,
    BeamBatchResult,
)
from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspaceKey
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.engine.beam_search_v1 import (
    GRExecutionCapacity,
    GRRetireMetadata,
    GRRetireResult,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
    validate_dispatch,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
)
from vllm_gr.v1.worker.beam_final_output import GPUBeamFinalOutput
from vllm_gr.v1.worker.beam_search_runner import (
    BeamBatchSearchRunner,
    BeamCandidateBatch,
    BeamSearchRunner,
)
from vllm_gr.v1.worker.gr_inputs import (
    GRInputBuffers,
    PreparedGRInputs,
    bind_device_inputs,
)

METADATA_ABI = 1
_UNCONSTRAINED_HEAD_LAYOUT_LOCK = Lock()
CONTROL_CAPACITY = 2
SUPPORTED_BATCH_SIZES = SUPPORTED_BEAM_BATCH_SIZES


def _batch_finished(is_finished: torch.Tensor, error_code: torch.Tensor) -> torch.Tensor:
    """Finish a logical batch only when all members finish or any member errors."""
    return is_finished.all() | (error_code != 0).any()


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


class AsyncGPUBeamOutput(AsyncModelRunnerOutput):  # type: ignore[misc]
    """Only the executor's output consumer waits and constructs the IPC value."""

    def __init__(
        self,
        owner: GPUBeamStageRunner,
        execution: BeamDispatchExecution,
        slot: int,
        terminal_payload: bool,
    ):
        self.owner = owner
        self.execution = execution
        self.slot = slot
        self.terminal_payload = terminal_payload
        self._output: ModelRunnerOutput | None = None
        self._lock = Lock()

    def get_output(self) -> ModelRunnerOutput:
        with self._lock:
            if self._output is not None:
                return self._output
            # CUDA/D2H failures are fatal and must reach the native Executor
            # failure channel. Never publish a successful release after one.
            try:
                if self.terminal_payload:
                    controls = [self.owner.final_output.wait_embedded_control()]
                else:
                    self.owner.copy_done[self.slot].synchronize()
                    controls = self.owner.host_control[self.slot].tolist()
                self._output = self._convert(controls, final_ready=self.terminal_payload)
            except (ValueError, TypeError, IndexError) as error:
                self._output = self._conversion_failure(error)
            finally:
                with self.owner.control_drained:
                    self.owner.control_busy[self.slot] = False
                    self.owner.control_drained.notify_all()
            results = getattr(self._output, GR_WORKER_RESULT_ATTR)
            # The Scheduler looks ahead, so a terminal receipt from an earlier
            # stage can still be followed by an already-issued successor. Only
            # the declared last stage proves that nothing is still in transit;
            # early termination keeps the explicit retirement round trip.
            last_stage = self.execution.entries[-1].metadata.is_last_stage
            if last_stage and any(result.finished for result in results.values()):
                self.owner.request_terminal_release()
            if self.owner.release_terminal_if_drained():
                session_id = self.owner.binding.session_id
                result = results[session_id]
                results[session_id] = replace(result, resources_released=True)
            return self._output

    def _conversion_failure(self, error: Exception) -> ModelRunnerOutput:
        out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
        results = {}
        for entry in self.execution.entries:
            m = entry.metadata
            failure = BeamBatchResult(
                BEAM_BATCH_PROTOCOL_VERSION,
                m.session_id,
                None,
                BeamBatchFailure("worker_output_error", str(error), entry.binding.item_index),
            )
            terminal = self.owner.final_output.failure(self.owner, failure)
            results[m.session_id] = GRWorkerResult(
                m.session_id,
                m.dispatch_id,
                m.stage_index,
                0,
                m.output_device_state,
                True,
                terminal,
                owner_generation=m.owner_generation,
            )
        out.req_ids = [binding.native_request_id for binding in self.owner.bindings]
        out.req_id_to_index = {request_id: i for i, request_id in enumerate(out.req_ids)}
        out.sampled_token_ids = [[] for _ in out.req_ids]
        setattr(out, GR_WORKER_RESULT_ATTR, results)
        return out

    def _convert(self, controls, *, final_ready: bool) -> ModelRunnerOutput:
        if len(controls) != len(self.execution.entries):
            raise ValueError("Asynchronous scheduling control snapshot has the wrong row count")
        req_ids, sampled, results = [], [], {}
        for entry, (produced, finished, error) in zip(self.execution.entries, controls):
            m = entry.metadata
            terminal = None
            final_output = self.owner.final_output
            if error:
                failure = BeamBatchResult(
                    protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
                    batch_id=m.session_id,
                    output=None,
                    failure=BeamBatchFailure(
                        "beam_device_error",
                        f"Beam device error {error}",
                        entry.binding.item_index,
                    ),
                )
                terminal = final_output.failure(self.owner, failure)
            elif finished and not self.owner.cancelled:
                try:
                    terminal = final_output.consume(self.owner, ready=final_ready)
                except (ValueError, TypeError, IndexError) as exc:
                    failure = BeamBatchResult(
                        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
                        batch_id=m.session_id,
                        output=None,
                        failure=BeamBatchFailure(
                            "beam_final_output_error",
                            f"Failed to materialize final Beam output: {exc}",
                            entry.binding.item_index,
                        ),
                    )
                    terminal = final_output.failure(self.owner, failure)
            # These are scheduler placeholders. They must never be used as
            # model inputs or published as generated Beam sequences.
            results[m.session_id] = GRWorkerResult(
                m.session_id,
                m.dispatch_id,
                m.stage_index,
                produced,
                m.output_device_state,
                bool(finished),
                terminal,
                owner_generation=m.owner_generation,
            )
        req_ids = [binding.native_request_id for binding in self.owner.bindings]
        sampled = [([0] if produced else []) for _ in req_ids]
        out = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
        )
        setattr(out, GR_WORKER_RESULT_ATTR, results)
        return out


class _StageExecutionHooks:
    """Per-runner execution surface bridging the persistent GR stage core.

    The stage-runner core (session/state/workspace/constraint/inputs) is
    ModelRunner-agnostic; only the native execution surface differs between
    the V1 and V2 runners.  Each hook maps one such surface.  The default
    implementation below is the V1 surface (``_model_forward`` /
    ``execute_model_state.logits`` / ``prepare_inputs_event`` /
    ``_update_states`` / ``requests[..].block_ids``); the V2 runner installs
    its own via :mod:`vllm_gr.v1.worker.gpu_beam_stage_runner_v2`.
    """

    persistent_graph_install = True
    supports_transposed_head = True

    def __init__(self, worker: Any):
        self.worker = worker

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention: Any,
        slot_mappings: Any,
        capture: bool,
    ) -> torch.Tensor:
        """Run one beam-decode forward, returning hidden states."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.cudagraph_dispatcher import BatchDescriptor

        worker = self.worker
        with set_forward_context(
            attention,
            worker.vllm_config,
            num_tokens=input_ids.numel(),
            cudagraph_runtime_mode=CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=input_ids.numel(), uniform=True),
            slot_mapping=slot_mappings,
        ):
            return worker._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )

    def prefill_logits(
        self, original: Callable[..., Any], scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Run the native prefill execution and extract its logits."""
        worker = self.worker
        out = original(worker, scheduler_output, *args, **kwargs)
        if out is not None or worker.execute_model_state is None:
            raise RuntimeError("native GR Prefill must return logits through execute_model_state")
        return worker.execute_model_state.logits

    def prefill_stage_fence(self) -> None:
        """Wait for any native prefill reads of reused CPU input buffers.

        V1: the pinned InputBatch buffers the Prefill forward reads
        asynchronously are mutated on the CPU by ``_update_states``; a stream
        wait on the decode stream cannot protect those Host writes.  V2
        builds a fresh InputBatch per step and needs no fence.
        """
        event = getattr(self.worker, "prepare_inputs_event", None)
        if event is not None:
            event.synchronize()

    def consume_native_state(self, scheduler_output: Any) -> None:
        """Apply the runner's native request-state bookkeeping."""
        self.worker._update_states(scheduler_output)

    def decode_block_ids(self, session_id: str) -> list[list[int]]:
        """Host KV block ids for the session's prompt, grouped per KV group."""
        native = self.worker.requests[session_id]
        return native.block_ids

    def initialize_inputs(self, owner: GPUBeamStageRunner, rows: int) -> None:
        """V1 model inputs belong to the stage; upload storage stays lazy."""
        owner.input_ids = torch.empty(rows, device=owner.beam.state.device, dtype=torch.int32)
        owner.positions = torch.empty_like(owner.input_ids, dtype=torch.int64)
        owner.slots = torch.full_like(owner.positions, -1)
        owner.execution_mask = torch.empty_like(owner.input_ids, dtype=torch.bool)

    @staticmethod
    def required_constraint_rows(batch_size: int, beam_width: int) -> int:
        return beam_width

    def prepare_new_adapter(self, view: BeamBatchExecutionView) -> None:
        graph = getattr(self.worker, "_beam_decode_graph", None)
        if graph is not None:
            graph.select(view.batch_size, view.beam_width).discard_selected()

    def decode_graph(self, batch_size: int, beam_width: int) -> Any:
        from vllm_gr.v1.worker.beam_decode_graph import get_beam_decode_graph

        return get_beam_decode_graph(self.worker, batch_size=batch_size, beam_width=beam_width)

    def decode_logits(self, owner: GPUBeamStageRunner) -> torch.Tensor:
        """V1 can capture a missing graph against a new stage's buffers."""
        worker = self.worker
        from vllm_gr.v1.worker.model_runner_common import _get_gr_config

        graph_enabled = (
            not worker.model_config.enforce_eager
            and _get_gr_config(worker).beam_graph_enabled
            and not getattr(worker, "_gr_async_decode_graph_capture_failed", False)
        )
        if graph_enabled:
            graph = self.decode_graph(len(owner.bindings), owner.inputs.width)
            if not graph.ready:
                graph.capture(
                    owner._forward_hidden,
                    persistent_install=self.persistent_graph_install,
                )
            owner.graph = graph
        if should_replay_decode_graph(worker, owner.graph):
            return owner._decode_logits(owner.graph.replay())
        else:
            return owner._forward()

    def consume_retirement_state(
        self, scheduler_output: Any, bindings: tuple[BeamRequestBinding, ...]
    ) -> None:
        self.consume_native_state(scheduler_output)


def should_replay_decode_graph(worker: Any, graph: Any) -> bool:
    """Resolve decode graph replay, preserving unexpected-missing invariants."""
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    requested = not worker.model_config.enforce_eager and _get_gr_config(worker).beam_graph_enabled
    if not requested:
        return False
    if getattr(worker, "_gr_async_decode_graph_capture_failed", False):
        if graph is not None:
            raise RuntimeError("Decode capture failed but a graph is still published")
        return False
    if graph is not None:
        if not graph.ready:
            raise RuntimeError("Published Decode graph is not ready")
        return True
    raise RuntimeError("Asynchronous scheduling graph mode requires successful startup capture")


def _get_unconstrained_head_layout(
    worker: Any,
) -> tuple[torch.Tensor, torch.Event] | None:
    """Cache a read-only, vocabulary-contiguous Qwen3 head for Decode."""
    if os.environ.get("VLLM_GR_UNCONSTRAINED_HEAD_LAYOUT") != "transposed":
        return None
    if (
        worker.parallel_config.tensor_parallel_size != 1
        or worker.parallel_config.pipeline_parallel_size != 1
        or getattr(worker, "lora_config", None) is not None
        or getattr(worker.model_config, "quantization", None) is not None
        or getattr(worker.model_config.hf_config, "architectures", None) != ["Qwen3ForCausalLM"]
    ):
        return None
    from vllm import envs
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    model = worker.model
    head = getattr(model, "lm_head", None)
    processor = getattr(model, "logits_processor", None)
    if (
        envs.VLLM_BATCH_INVARIANT
        or not isinstance(head, VocabParallelEmbedding)
        or type(getattr(head, "quant_method", None)) is not UnquantizedEmbeddingMethod
        or type(processor) is not LogitsProcessor
        or processor.logits_as_input
        or processor.soft_cap is not None
        or processor.scale != 1.0
        or getattr(head, "bias", None) is not None
    ):
        return None
    weight = head.weight
    if (
        not weight.is_cuda
        or not weight.is_contiguous()
        or weight.ndim != 2
        or weight.dtype not in (torch.bfloat16, torch.float16)
        or weight.shape[0] != processor.org_vocab_size
    ):
        return None
    source_key = (
        id(weight),
        weight.data_ptr(),
        weight._version,
        tuple(weight.shape),
        weight.dtype,
    )
    with _UNCONSTRAINED_HEAD_LAYOUT_LOCK:
        cached = getattr(worker, "_gr_unconstrained_head_layout", None)
        if cached is not None and cached[0] == source_key:
            return cached[1], cached[2]
        transposed = weight.detach().T.contiguous()
        ready = torch.Event()
        ready.record(torch.cuda.current_stream(weight.device))
        worker._gr_unconstrained_head_layout = (source_key, transposed, ready)
        return transposed, ready


class GPUBeamStageRunner:
    """One exclusive execution owner, using the existing session/state/workspace."""

    def __init__(
        self,
        worker: Any,
        beam: BeamBatchSearchRunner,
        bindings: tuple[BeamRequestBinding, ...],
        backend: CUDAConstraintBackend | None,
        hooks: _StageExecutionHooks | None = None,
    ):
        expected = tuple(range(len(bindings)))
        if not bindings or (
            tuple(binding.item_index for binding in bindings) != expected
            or tuple(binding.kv_slot for binding in bindings) != expected
            or tuple(binding.state_row for binding in bindings) != expected
            or tuple(binding.workspace_row for binding in bindings) != expected
            or tuple(binding.native_request_id for binding in bindings) != beam.layout.request_ids
            or any(binding.session_id != beam.layout.session_id for binding in bindings)
        ):
            raise ValueError("GR binding disagrees with existing session resources")
        context = beam.session.context
        if (
            getattr(context, "_gr_execution_owner", None) is not None
            or beam.workspace.execution_owner is not None
        ):
            raise RuntimeError("Beam context already has an execution owner")
        self.worker, self.beam, self.bindings, self.backend = worker, beam, bindings, backend
        self.binding = bindings[0]
        self.batch_view = beam.layout
        # Device indexing is part of Decode binding and must not upload a CPU
        # index tensor while a CUDA graph is being captured.
        self.state_rows = torch.tensor(
            tuple(binding.state_row for binding in bindings),
            dtype=torch.long,
            device=beam.state.device,
        )
        self.hooks = hooks if hooks is not None else _StageExecutionHooks(worker)
        self.previous: GRStageMetadata | None = None
        self.pending: tuple[BeamDispatchExecution, int] | None = None
        self._validated_dispatch: tuple[Any, tuple[GRStageMetadata, ...]] | None = None
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
        self.terminal_release_requested = False
        self.terminal_release_claimed = False
        self.execution_error: Exception | None = None
        self.dynamic: Any = None
        self.unconstrained_topk_scratch: UnconstrainedTopKScratch | None = None
        self.inputs: GRInputBuffers | None = None
        self.prepared: PreparedGRInputs | None = None
        self.final_output: GPUBeamFinalOutput
        self.metadata: Any = None
        self.graph: Any = None
        self.logits: torch.Tensor | None = None
        batch_size = len(bindings)
        rows = batch_size * beam.state.beam_width
        self._head_layout = (
            _get_unconstrained_head_layout(worker)
            if backend is None and self.hooks.supports_transposed_head
            else None
        )
        self._head_logits = (
            torch.empty(
                (rows, self._head_layout[0].shape[1]),
                dtype=self._head_layout[0].dtype,
                device=beam.state.device,
            )
            if self._head_layout is not None
            else None
        )
        self.input_ids: torch.Tensor
        self.positions: torch.Tensor
        self.slots: torch.Tensor
        self.execution_mask: torch.Tensor
        self.hooks.initialize_inputs(self, rows)
        self.constraint_steps = beam.state.num_generated_tokens[:batch_size]
        w = beam.workspace
        self.prefixes = w.prefix_token_ids[:batch_size].reshape(rows, -1)
        self.temperatures = w.temperatures[:batch_size].reshape(rows)
        self.tokens = w.topk_token_ids[:batch_size].reshape(rows, beam.state.beam_width)
        self.scores = w.topk_scores[:batch_size].reshape(rows, beam.state.beam_width)
        self.valid = w.topk_valid_mask[:batch_size].reshape(rows, beam.state.beam_width)
        self.counts = w.topk_valid_counts[:batch_size].reshape(rows)
        context._gr_execution_owner = self
        beam.workspace.execution_owner = self

    def admit(self, scheduler_output: Any) -> BeamDispatchExecution:
        if self.closed:
            raise RuntimeError("GR session is closed")
        if (
            self.backend is not None
            and not self.backend.is_ready
            or self.beam.workspace.execution_owner is not self
            or self.beam.session.context._gr_execution_owner is not self
        ):
            raise RuntimeError("GR backend or shared execution ownership changed")
        if self.pending is not None:
            raise RuntimeError("sample_tokens must consume the previous GR execute state")
        validated = self._validated_dispatch
        self._validated_dispatch = None
        if validated is not None and validated[0] is scheduler_output:
            stages = validated[1]
        else:
            stages = validate_dispatch(
                scheduler_output,
                previous=self.previous,
                bindings=self.bindings,
            )
        for m in stages:
            state, params = self.beam.state, m.beam_params
            if (
                state.beam_width != params.beam_width
                or state.max_new_tokens != params.max_tokens
                or state.stop_token_ids_config != params.stop_token_ids
                or state.ignore_stop_tokens_config != params.ignore_stop_tokens
            ):
                raise ValueError("GR metadata configuration does not match persistent Beam state")
        if not hasattr(self, "final_output"):
            first = stages[0]
            self.final_output = GPUBeamFinalOutput(
                self, self.bindings, first.beam_params, first.output_options
            )
        elif self.previous is None or self.final_output.bindings != self.bindings:
            first = stages[0]
            if self.final_output.is_compatible(self):
                self.final_output.reset(
                    self, self.bindings, first.beam_params, first.output_options
                )
            else:
                self.final_output = GPUBeamFinalOutput(
                    self, self.bindings, first.beam_params, first.output_options
                )
        entries, offset = [], 0
        for m in stages:
            view = BeamBatchExecutionView.from_stage_metadata(m)
            count = view.num_execution_rows if m.kind is GRStageKind.DECODE else view.batch_size
            entries.append(BeamDispatchEntry(self.binding, m, offset, count))
            offset += count
        with self.control_lock:
            slot = next((i for i, busy in enumerate(self.control_busy) if not busy), None)
            if slot is None:
                raise RuntimeError("both GR dispatch control slots await output consumption")
            self.control_busy[slot] = True
        execution = BeamDispatchExecution(tuple(entries), offset)
        self.pending = execution, slot
        self.previous = stages[0]
        torch.cuda.current_stream(self.beam.state.device).wait_event(self.last_work)
        self.device_control[slot, 0, :1].copy_(
            self.beam.state.num_generated_tokens[: len(self.bindings)].amax().reshape(1)
        )
        return execution

    def prepare_cpu(self, m: GRStageMetadata, slot: int) -> PreparedGRInputs:
        worker = self.worker
        if self.dynamic is not None:
            self.dynamic.activate(BeamBatchExecutionView.from_stage_metadata(m).input_width)
        groups = worker.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise RuntimeError("Asynchronous scheduling requires one uniform KV cache group")
        if self.inputs is None:
            self.inputs = GRInputBuffers(
                self.beam.state.beam_width,
                worker.model_config.max_model_len,
                groups[0].kv_cache_spec.block_size,
                self.beam.state.device,
                max_batch=len(self.bindings),
            )
        block_ids = (
            tuple([list(row)] for row in m.kv_blocks)
            if m.kv_blocks is not None
            else tuple(
                self.hooks.decode_block_ids(binding.native_request_id) for binding in self.bindings
            )
        )
        return self.inputs.prepare_cpu_batch(
            m, self.bindings, block_ids_by_request=block_ids, slot=slot
        )

    def _bind_attention(self, m: GRStageMetadata) -> None:
        from vllm_gr.v1.attention.backends.beam_attn_gpu import (
            GPUBeamAttentionBatch,
        )
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
            session=GPUBeamAttentionBatch(
                self.beam.sessions,
                width=BeamBatchExecutionView.from_stage_metadata(m).input_width,
            ),
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
        if prepared is None:
            raise RuntimeError(
                "Asynchronous scheduling device binding does not match the prepared dispatch"
            )
        prepared_bindings = (
            prepared.binding if isinstance(prepared.binding, tuple) else (prepared.binding,)
        )
        if prepared.metadata != entry.metadata or prepared_bindings != self.bindings:
            raise RuntimeError(
                "Asynchronous scheduling device binding does not match the prepared dispatch"
            )
        bind_device_inputs(prepared, self)

    def capture_forward(self) -> Any:
        """Return a captured candidate; startup publishes it after cleanup."""
        graph = self.hooks.decode_graph(
            len(self.bindings),
            self.inputs.width if self.inputs is not None else self.beam.state.beam_width,
        )
        graph.capture(self._forward_hidden, persistent_install=self.hooks.persistent_graph_install)
        return graph

    def _forward(self) -> torch.Tensor:
        return self._decode_logits(self._forward_hidden(False))

    def _decode_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self._try_transposed_head_logits(hidden)
        return logits if logits is not None else self.worker.model.compute_logits(hidden)

    def _try_transposed_head_logits(self, hidden: torch.Tensor) -> torch.Tensor | None:
        layout = self._head_layout
        output = self._head_logits
        if (
            layout is None
            or output is None
            or hidden.ndim != 2
            or not hidden.is_contiguous()
            or hidden.device != output.device
            or hidden.dtype != output.dtype
            or hidden.shape[1] != layout[0].shape[0]
            or hidden.shape[0] > output.shape[0]
        ):
            return None
        transposed, ready = layout
        torch.cuda.current_stream(hidden.device).wait_event(ready)
        logits = output[: hidden.shape[0]]
        torch.mm(hidden, transposed, out=logits)
        return logits

    def _forward_hidden(self, capture: bool) -> torch.Tensor:
        return self.hooks.forward_hidden(
            self.input_ids, self.positions, self.attention, self.slot_mappings, capture
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
                self.logits = self.hooks.prefill_logits(original, scheduler_output, *args, **kwargs)
            else:
                # Pure CPU work can run before the native Prefill staging fence.
                self.prepared = self.prepare_cpu(m, self.pending[1])
                if previous_kind is GRStageKind.PREFILL:
                    # Native Prefill may asynchronously read reused CPU input
                    # buffers that the native state update mutates.  The
                    # runner-specific fence orders those Host writes.
                    self.hooks.prefill_stage_fence()
                # Decode does not upload these native input buffers again, so
                # subsequent Decode dispatches need no Host input-prep wait.
                self.hooks.consume_native_state(scheduler_output)
                self._bind_attention(m)
                self.prepare_decode(entry)
                self.logits = self.hooks.decode_logits(self)
        return None

    def sample(self, grammar_output: Any = None) -> AsyncGPUBeamOutput:
        if self.execution_error is not None:
            raise RuntimeError(
                "Asynchronous scheduling execute_model failed"
            ) from self.execution_error
        if grammar_output is not None or self.pending is None:
            raise RuntimeError("GR sample requires a pending dispatch and no grammar output")
        execution, slot = self.pending
        terminal_payload = False
        for entry in execution.entries:
            m = entry.metadata
            if m.produces_output and not self.cancelled:
                if self.logits is None:
                    raise RuntimeError("GR output stage has no native model logits")
                layout = BeamBatchExecutionView.from_stage_metadata(m)
                rows = layout.batch_size if m.kind is GRStageKind.PREFILL else entry.row_count
                candidates = (
                    self.dynamic.candidate_buffers(rows, layout.candidate_top_k)
                    if self.dynamic is not None
                    else (self.tokens[:rows], self.scores[:rows], self.valid[:rows])
                )
                if self.backend is not None:
                    self.backend.select_out_unchecked(
                        self.logits,
                        ConstraintMetadata(
                            self.prefixes[:rows],
                            self.temperatures[:rows],
                            0,
                            layout.candidate_top_k,
                        ),
                        out_token_ids=candidates[0],
                        out_logprobs=candidates[1],
                        out_valid_mask=candidates[2],
                        out_valid_counts=self.counts[:rows],
                        device_step=self.constraint_steps,
                    )
                else:
                    logits = self.logits[:rows]
                    if (
                        self.unconstrained_topk_scratch is None
                        or self.unconstrained_topk_scratch.top_k != layout.candidate_top_k
                    ):
                        self.unconstrained_topk_scratch = allocate_unconstrained_topk_scratch(
                            batch_size=len(self.bindings) * self.beam.state.beam_width,
                            vocab_size=int(logits.shape[-1]),
                            top_k=layout.candidate_top_k,
                            device=logits.device,
                        )
                    unconstrained_topk_out(
                        logits,
                        self.temperatures[:rows],
                        layout.candidate_top_k,
                        self.unconstrained_topk_scratch,
                        candidates[0],
                        candidates[1],
                        candidates[2],
                        self.counts[:rows],
                    )
                candidate_batch = BeamCandidateBatch(layout, *candidates)
                if self.dynamic is not None:
                    self.dynamic.advance(layout, candidates, m.is_last_stage)
                elif m.kind is GRStageKind.PREFILL:
                    self.beam.initialize_from_prefill(candidate_batch)
                else:
                    self.beam.advance(candidate_batch)
            state_rows = slice(0, len(self.bindings))
            control = self.device_control[slot, 0]
            control[:1].copy_(
                self.beam.state.num_generated_tokens[state_rows].amax().reshape(1) - control[:1]
            )
            control[1:2].copy_(
                _batch_finished(
                    self.beam.state.is_finished[state_rows],
                    self.beam.state.error_code[state_rows],
                ).reshape(1)
            )
            control[2:3].copy_(self.beam.state.error_code[state_rows].amax().reshape(1))
            # The fixed-length terminal stage returns one packed payload whose
            # header replaces the separate per-stage control D2H.
            if m.is_last_stage:
                terminal_payload = self.final_output.enqueue(self, control)
        stream = torch.cuda.current_stream(self.beam.state.device)
        self.produced_ready[slot].record(stream)
        self.last_work.record(stream)
        if not terminal_payload:
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(self.produced_ready[slot])
                self.host_control[slot].copy_(self.device_control[slot], non_blocking=True)
                self.copy_done[slot].record(self.copy_stream)
        self.worker.execute_model_state = None
        self.worker.kv_connector_output = None
        self.pending = None
        return AsyncGPUBeamOutput(self, execution, slot, terminal_payload)

    def completion_view(self, binding: BeamRequestBinding) -> dict[str, torch.Tensor]:
        if self.closed or binding not in self.bindings:
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
        self.final_output.abort(self)
        self.cancel(self.binding)
        return AsyncGRRetireOutput(self, metadata)

    def request_terminal_release(self) -> None:
        """Stop further work after a terminal receipt; release after all consumers drain."""
        with self.control_lock:
            if self.terminal_release_requested:
                return
            # A terminal control snapshot already proves that device execution
            # stopped. Avoid enqueueing the fallback cancel writes on the hot
            # path; any look-ahead receipt is covered by the latest last_work.
            self.final_output.abort(self)
            self.cancelled = True
            self.terminal_release_requested = True

    def release_terminal_if_drained(self) -> bool:
        """Release on the last terminal output consumer, without a control dispatch."""
        with self.control_lock:
            ready = (
                self.terminal_release_requested
                and not self.terminal_release_claimed
                and not self.closed
                and self.pending is None
                and not any(self.control_busy)
            )
            if ready:
                self.terminal_release_claimed = True
        if not ready:
            return False
        self.release(
            self.binding,
            consumer_done=self.final_output.consumer_done(self),
        )
        # Native Scheduler cleanup normally arrives in the next
        # SchedulerOutput.finished_req_ids. The fused terminal path has no
        # follow-up control dispatch, so retire the Worker's cached request
        # locally after all GPU and output consumers have drained.
        cleanup = SchedulerOutput.make_empty()
        cleanup.finished_req_ids.update(binding.native_request_id for binding in self.bindings)
        self.hooks.consume_native_state(cleanup)
        return True

    @torch.inference_mode()
    def cancel(self, binding: BeamRequestBinding) -> None:
        if binding not in self.bindings:
            raise ValueError("cancel binding mismatch")
        if self.closed or self.cancelled:
            return
        if self.pending is not None:
            raise RuntimeError("sample the dispatched work before cancellation")
        stream = torch.cuda.current_stream(self.beam.state.device)
        stream.wait_event(self.last_work)
        rows = slice(0, len(self.bindings))
        self.beam.state.is_finished[rows].fill_(True)
        self.beam.state.active_mask[rows].zero_()
        self.last_work.record(stream)
        self.cancelled = True

    def release(self, binding: BeamRequestBinding, *, consumer_done: torch.Event) -> None:
        if binding not in self.bindings:
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
        for member in self.bindings:
            self.worker._beam_sessions.pop(member.native_request_id, None)


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
            owner.release(
                owner.binding,
                consumer_done=owner.final_output.consumer_done(owner),
            )
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
            # Startup graph capture releases its dummy owner directly and
            # never reaches this real-retirement path. Schedule only after a
            # serving session has fully drained and released its resources,
            # so the one-shot freeze covers lazily-created persistent state
            # without being consumed by the startup dummy session.
            from vllm_gr.v1.worker.model_runner_common import (
                schedule_warmup_gc_freeze,
            )

            schedule_warmup_gc_freeze()
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
    owner.hooks.prefill_stage_fence()
    owner.hooks.consume_retirement_state(scheduler_output, owner.bindings)
    worker._gr_retirement_output = result
    return result


def get_stage_runner(
    worker: Any,
    scheduler_output: Any,
    *,
    hooks_type: type[_StageExecutionHooks] = _StageExecutionHooks,
) -> GPUBeamStageRunner:
    """Admit first, then derive the adapter from existing Worker resources."""
    existing = getattr(worker, "_gr_gpu_stage_runner", None)
    if existing is not None and type(existing.hooks) is not hooks_type:
        raise ValueError("GR execution hooks cannot change during the runner lifetime")
    if existing is not None and not existing.closed:
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
    # A GRConfig without the switch (or a partial config object) keeps the
    # constrained default: only an explicit ``False`` selects full vocabulary.
    unconstrained = getattr(cfg, "constrained_generation", None) is False
    if not unconstrained and not isinstance(backend, CUDAConstraintBackend):
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
    if backend is not None and (
        not backend.is_ready
        or backend.info.table_metadata.vocab_size != worker.model_config.get_vocab_size()
        or any(table.device != worker.device for table in backend._tables())
    ):
        raise ValueError("constraint backend is not ready or its model/device binding differs")
    if backend is not None and (
        backend.info.max_rows
        < hooks_type.required_constraint_rows(len(m.request_ids), cfg.beam_max_width)
        or backend.info.max_top_k < cfg.beam_max_width
    ):
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
    view = BeamBatchExecutionView.from_stage_metadata(m)
    context = getattr(worker, "_beam_context", None)
    if context is None:
        context = _create_beam_context(worker, cfg, max_batch=max(SUPPORTED_BATCH_SIZES))
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
        or context.max_batch < view.batch_size
    ):
        raise ValueError("existing Beam context and workspace capacity are incompatible")
    # Check live storage before resetting state or allocating new sessions.
    # The key covers every pool and shared buffer; validating the final slot
    # also checks capacity for all contiguous request-major rows.
    # Do this before reuse_adapter so rejection cannot leave partially reset state.
    resource_key = BeamSearchRunner.validate_resources(
        context,
        workspace,
        beam_width=m.beam_params.beam_width,
        max_new_tokens=m.beam_params.max_tokens,
        kv_slot=view.batch_size - 1,
    )
    resources_match = existing is not None and all(
        runner._resource_key == resource_key for runner in existing.beam.runners
    )
    if existing is not None and existing.graph is not None and not resources_match:
        raise ValueError("captured Beam resource addresses or layouts changed")
    # The graph captures the adapter's fixed Forward and attention buffers,
    # not the request-local Beam state or its effective token budget.
    from vllm_gr.v1.worker.gpu_dynamic_beam import (
        GPUDynamicBeam,
        discard_dynamic_graphs,
        uses_dynamic_beam,
    )

    dynamic_requested = uses_dynamic_beam(m.beam_params)
    existing_dynamic = getattr(existing, "dynamic", None)
    reuse_adapter = (
        existing is not None
        and existing.closed
        and resources_match
        and (existing_dynamic is not None) == dynamic_requested
        and existing.beam.layout.dynamic_beam_widths == view.dynamic_beam_widths
        and existing.beam.state.beam_width == m.beam_params.beam_width
        and existing.beam.session.context is context
        and existing.beam.workspace is workspace
        and existing.beam.state.decision is context.decision
        and len(getattr(existing.beam, "runners", ())) == view.batch_size
        and all(
            runner._scratch_owner is workspace.state_scratch for runner in existing.beam.runners
        )
    )
    reuse_beam = (
        reuse_adapter
        and existing.beam.state.max_new_tokens == m.beam_params.max_tokens
        and existing.beam.state.stop_token_ids_config == m.beam_params.stop_token_ids
        and existing.beam.state.ignore_stop_tokens_config == m.beam_params.ignore_stop_tokens
    )
    if reuse_beam:
        assert existing is not None
        existing.beam.reset_for_session()
        state = existing.beam.state
    else:
        state = BeamSearchState(
            context.decision,
            beam_width=m.beam_params.beam_width,
            max_new_tokens=m.beam_params.max_tokens,
            stop_token_ids=m.beam_params.stop_token_ids,
            ignore_stop_tokens=m.beam_params.ignore_stop_tokens,
        )
    if not reuse_adapter and (existing_dynamic is not None or dynamic_requested):
        discard_dynamic_graphs(worker, context)
    sessions = []
    bindings = []
    for item_index, (request_id, prompt_tokens) in enumerate(
        zip(view.request_ids, view.num_prompt_tokens_by_request)
    ):
        slot = context.registry.allocate(request_id, m.beam_params.beam_width, prompt_tokens)
        if slot != item_index:
            raise RuntimeError("GR batch requires contiguous request-major KV slots")
        session = BeamSearchSession(request_id, slot, context)
        session.begin_session(m.beam_params.beam_width, prompt_tokens, decision_index=item_index)
        sessions.append(session)
        bindings.append(
            BeamRequestBinding(m.session_id, item_index, request_id, item_index, slot, item_index)
        )
    session_tuple = tuple(sessions)
    binding_tuple = tuple(bindings)
    if reuse_beam:
        assert existing is not None
        for runner, session in zip(existing.beam.runners, session_tuple):
            runner.rebind_session(session)
        beam = existing.beam
        beam.layout = view
    else:
        runners = tuple(
            BeamSearchRunner(
                session,
                state,
                workspace,
                state_index=index,
                workspace_index=index,
            )
            for index, session in enumerate(session_tuple)
        )
        beam = BeamBatchSearchRunner(runners, view)
    if reuse_adapter:
        assert existing is not None
        adapter = existing
        adapter.beam, adapter.bindings, adapter.backend = beam, binding_tuple, backend
        adapter.binding, adapter.batch_view = binding_tuple[0], view
        adapter.previous, adapter.pending = None, None
        adapter.closed, adapter.cancelled = False, False
        adapter.terminal_release_requested = False
        adapter.terminal_release_claimed = False
        adapter.execution_error = None
        adapter.prepared = None
        adapter.constraint_steps = beam.state.num_generated_tokens[: view.batch_size]
        context._gr_execution_owner = adapter
        workspace.execution_owner = adapter
    else:
        hooks = hooks_type(worker)
        hooks.prepare_new_adapter(view)
        adapter = GPUBeamStageRunner(worker, beam, binding_tuple, backend, hooks=hooks)
    if dynamic_requested:
        if adapter.dynamic is None:
            adapter.dynamic = GPUDynamicBeam(adapter)
        else:
            adapter.dynamic.reset()
        adapter.prefixes = workspace.prefix_token_ids[: view.batch_size].reshape(
            view.batch_size * view.beam_width, -1
        )
    worker._beam_sessions = {
        binding.native_request_id: session for binding, session in zip(binding_tuple, session_tuple)
    }
    worker._gr_gpu_stage_runner = adapter
    adapter._validated_dispatch = (scheduler_output, stages)
    return cast(GPUBeamStageRunner, adapter)
