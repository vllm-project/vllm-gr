# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contracts for EngineCore-owned Asynchronous scheduling Beam Search V1 execution and retirement."""

from __future__ import annotations

import math
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    DEFAULT_CONSTRAINT_RESOURCE_ID,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
    validate_beam_batch_request,
)


class GRStageKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class GRSessionStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHING = "finishing"
    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"


class GRResourceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    DRAINING = "draining"
    RETIRING = "retiring"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class GRExecutionCapacity:
    """Shared Asynchronous scheduling admission contract; storage capacity does not enable concurrency."""

    beam_width: int
    active_sessions: int = 1
    dispatches_per_session: int = 2

    def __post_init__(self) -> None:
        if type(self.beam_width) is not int or not 1 <= self.beam_width <= 256:
            raise ValueError(
                "Asynchronous scheduling requires a fixed physical beam width in [1, 256]"
            )
        if (
            type(self.active_sessions) is not int
            or self.active_sessions != 1
            or type(self.dispatches_per_session) is not int
            or self.dispatches_per_session != 2
        ):
            raise ValueError(
                "Asynchronous scheduling requires one execution owner and two dispatch slots"
            )

    def validate(self, params: BeamBatchParams) -> None:
        if not params.ignore_stop_tokens:
            raise ValueError("beam_search_v1 requires ignore_eos=True")
        if params.catalog_strategy is not None:
            raise ValueError("Constraint Table does not support explicit catalog_strategy")
        if (
            type(params.beam_width) is not int
            or params.beam_width != self.beam_width
            or type(params.max_tokens) is not int
            or params.max_tokens not in (1, 2, 3)
        ):
            raise ValueError(
                "Asynchronous scheduling requires the startup beam width and effective N=1/2/3"
            )


@dataclass(frozen=True, slots=True)
class GRRetireMetadata:
    session_id: str
    owner_generation: int
    retire_id: int


@dataclass(frozen=True, slots=True)
class GRRetireResult:
    session_id: str
    owner_generation: int
    retire_id: int
    released: bool


def validate_gr_execution_config(config: Any) -> None:
    """Reject unsupported Asynchronous scheduling runtimes before native/Worker request allocation."""
    from vllm_gr.config import BeamExecutionMode
    from vllm_gr.config.execution import require_beam_execution_mode

    require_beam_execution_mode(config, BeamExecutionMode.V1)
    if not config.scheduler_config.async_scheduling:
        raise ValueError("Asynchronous scheduling requires async_scheduling=True")
    if getattr(config.scheduler_config, "enable_chunked_prefill", False):
        raise ValueError("beam_search_v1 requires enable_chunked_prefill=False")
    if getattr(config.scheduler_config, "long_prefill_token_threshold", 0):
        raise ValueError("beam_search_v1 does not support partial Prefill scheduling")
    parallel = getattr(config, "parallel_config", None)
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError("Asynchronous scheduling requires TP/PP/DP/DCP/PCP=1")
    if getattr(parallel, "use_ubatching", False):
        raise ValueError("Asynchronous scheduling does not support microbatching")
    for name in ("speculative_config", "lora_config", "kv_transfer_config", "ec_transfer_config"):
        if getattr(config, name, None) is not None:
            raise ValueError(f"Asynchronous scheduling does not support {name}")
    model = getattr(config, "model_config", None)
    if any(
        getattr(model, name, False)
        for name in ("is_encoder_decoder", "is_hybrid", "is_multimodal_model")
    ):
        raise ValueError("Asynchronous scheduling requires a uniform text decoder model")
    dtype = getattr(model, "dtype", None)
    if dtype is not None and str(dtype) not in ("torch.float16", "torch.bfloat16"):
        raise ValueError("Asynchronous scheduling requires FP16 or BF16")
    if getattr(model, "quantization", None) is not None:
        raise ValueError("Asynchronous scheduling does not support quantization")


def validate_gr_device_config(config: Any) -> None:
    """Select an installed device adapter without importing a device runtime."""
    device = getattr(getattr(config, "device_config", None), "device", None)
    if device is not None and str(device).split(":")[0] != "cuda":
        raise ValueError("beam_search_v1 has no execution adapter for this device; GPU V1 required")


def validate_gr_request(request: BeamBatchRequest) -> None:
    """Validate public semantics before native ADD or Worker allocation."""
    validate_beam_batch_request(request)
    if len(request.items) != 1:
        raise ValueError("beam_search_v1 requires batch size B=1")
    native = request.items[0].request
    params = native.sampling_params
    if native.request_id != request.batch_id:
        raise ValueError("batch_id and native request_id must match")
    if native.prompt_embeds is not None or not native.prompt_token_ids:
        raise ValueError("beam_search_v1 requires a nonempty token prompt")
    if native.mm_features or native.lora_request is not None:
        raise ValueError("beam_search_v1 does not support multimodal inputs or LoRA")
    if params is None or not params.ignore_eos:
        raise ValueError("beam_search_v1 requires ignore_eos=True")
    if params.max_tokens != request.beam_params.max_tokens or params.n != 1:
        raise ValueError("native and Beam generation budgets must match and n must be 1")
    GRExecutionCapacity(request.beam_params.beam_width).validate(request.beam_params)


@dataclass(frozen=True, slots=True)
class GRAdmissionRuntime:
    """Engine-owned startup contract shared with its Scheduler."""

    config: Any
    capacity: GRExecutionCapacity


def initialize_gr_admission(core: Any) -> None:
    """Validate static V1 constraints before EngineCoreProc starts receiving input."""
    from vllm_gr.config import BeamExecutionMode, GRConfig

    if hasattr(core, "_gr_admission_runtime"):
        return  # Both EngineCore and EngineCoreProc constructors are wrapped.
    config = core.vllm_config
    gr_config = GRConfig.from_vllm_config(config)
    runtime = None
    if gr_config.beam_execution_mode is BeamExecutionMode.V1:
        validate_gr_execution_config(config)
        validate_gr_device_config(config)
        if core.batch_queue is None or core.batch_queue_size < 2:
            raise RuntimeError("beam_search_v1 requires a non-blocking EngineCore batch queue")
        if gr_config.beam_max_width is None:
            raise ValueError("beam_search_v1 requires beam_max_width to be fixed at startup")
        runtime = GRAdmissionRuntime(config, GRExecutionCapacity(gr_config.beam_max_width))
    core._gr_admission_runtime = runtime
    core.scheduler._gr_admission_runtime = runtime


@dataclass(frozen=True, slots=True)
class GRRequestAdmission:
    """One Core submission's semantic checks; never a verdict cached on the wire request.

    The outer msgspec contracts are frozen, but their native request and
    SamplingParams are mutable. Fence the fields used by validate_gr_request
    again when Scheduler consumes ADD, without repeating static/schema checks.
    """

    runtime: GRAdmissionRuntime
    request: BeamBatchRequest
    sampling_params: Any
    prompt_token_ids: Any
    num_prompt_tokens: int

    def validate(self, scheduler: Any, request: BeamBatchRequest) -> None:
        if (
            request is not self.request
            or getattr(scheduler, "_gr_admission_runtime", None) is not self.runtime
            or scheduler.vllm_config is not self.runtime.config
        ):
            raise ValueError("GR admission does not belong to this request and Scheduler")
        native = request.items[0].request
        params = native.sampling_params
        if (
            native.request_id != request.batch_id
            or native.prompt_token_ids is not self.prompt_token_ids
            or len(self.prompt_token_ids) != self.num_prompt_tokens
            or native.prompt_embeds is not None
            or native.mm_features
            or native.lora_request is not None
            or native.pooling_params is not None
            or params is not self.sampling_params
            or not params.ignore_eos
            or params.max_tokens != request.beam_params.max_tokens
            or params.n != 1
        ):
            raise ValueError("GR request changed after Core admission validation")


@dataclass(frozen=True, slots=True)
class GRDeviceStateRef:
    """Reference to Beam state that remains owned by the Worker device."""

    session_id: str
    producer_stage_index: int


@dataclass(slots=True)
class GRSessionState:
    """Resource lifecycle for one native Request.

    Native Request owns token progress and the effective generation budget.
    Frozen dispatch records identify device dependencies and receipts; they
    remain available after native finish removes the Request.
    """

    session_id: str
    request: BeamBatchRequest
    max_decode_steps: int
    status: GRSessionStatus = GRSessionStatus.WAITING
    terminal_result_pending: bool = False
    resource_status: GRResourceStatus = GRResourceStatus.WAITING
    owner_generation: int = 0
    retirement: GRRetireMetadata | None = None
    worker_resources_allocated: bool = False
    # EngineCore-private lifecycle state. These fields are excluded from the
    # constructor and never cross the Scheduler/Worker boundary.
    _next_dispatch_id: int = field(default=0, init=False, repr=False)
    _dispatch_records: dict[int, GRStageMetadata] = field(
        default_factory=dict, init=False, repr=False
    )
    _in_flight_dispatches: set[int] = field(default_factory=set, init=False, repr=False)
    _pending_terminal_result: BeamBatchResult | None = field(default=None, init=False, repr=False)
    _terminal_emitted: bool = field(default=False, init=False, repr=False)
    _kv_dispatch_leases: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if len(self.request.items) != 1:
            raise ValueError("Beam Search V1 currently requires batch size B=1")
        if self.max_decode_steps < 0:
            raise ValueError("max_decode_steps must be non-negative")
        if self.max_decode_steps != self.request.beam_params.max_tokens - 1:
            raise ValueError("max_decode_steps must equal effective max_tokens minus 1")

    @property
    def total_stages(self) -> int:
        return self.request.beam_params.max_tokens

    @property
    def issued_stages(self) -> int:
        """Diagnostic count of frozen dispatches, never a scheduling budget."""
        return len(self._dispatch_records)

    @property
    def completed_stages(self) -> int:
        return len(self._dispatch_records) - len(self._in_flight_dispatches)

    @property
    def in_flight_stages(self) -> int:
        return len(self._in_flight_dispatches)

    @property
    def latest_device_state(self) -> GRDeviceStateRef | None:
        if not self._dispatch_records:
            return None
        return next(reversed(self._dispatch_records.values())).output_device_state


@dataclass(frozen=True, slots=True)
class GRStageMetadata:
    """GR metadata attached to one native vLLM ``SchedulerOutput``.

    Stage 0 is Prefill. For effective budget N, stages 1..N-1 are Decode
    steps 0..N-2. ``dispatch_id`` identifies each physical invocation;
    ``stage_index`` follows native token progress. Every supported stage
    produces one logical token; partial Prefill is rejected. Worker echoes
    both IDs in GRWorkerResult.

    SchedulerOutput.gr_stage_metadata is keyed by session_id. Asynchronous scheduling admits one
    execution owner; owner_generation distinguishes reuse of a session ID.
    session_init carries the existing B=1 request (Beam parameters, output
    options and beam_params.constraint_resource_id) once, with Prefill only.
    Native SchedulerOutput supplies prompt/KV data. Stage k > 0 references
    (session_id, k - 1) even before host completion; Worker orders producer writes
    and successor reads through its execution stream or device events.
    """

    session_id: str
    dispatch_id: int
    stage_index: int
    kind: GRStageKind
    decode_step: int | None
    is_last_stage: bool
    input_device_state: GRDeviceStateRef | None
    output_device_state: GRDeviceStateRef
    beam_params: BeamBatchParams
    output_options: BeamOutputOptions
    num_prompt_tokens: int
    produces_output: bool = True
    session_init: BeamBatchRequest | None = None
    owner_generation: int = 0
    execution_capacity: GRExecutionCapacity | None = None


@dataclass(frozen=True, slots=True)
class GRWorkerResult:
    """Worker receipt attached as ModelRunnerOutput.gr_worker_results.

    The mapping key matches the dispatched session_id.
    Both GR attachments must survive executor IPC; async output wrappers must
    preserve gr_worker_results on the inner ModelRunnerOutput returned by get_output.
    Intermediate token IDs, parents and scores remain device-resident.
    terminal_result is supplied once for the terminating stage, with batch_id
    matching session_init.batch_id. EngineCore retains it until successors
    are drained and release is proven, then delivers it through wait_final.
    The final output consumer sets resources_released only after device and
    consumer drain; otherwise EngineCore uses a zero-token retirement frame.
    """

    session_id: str
    dispatch_id: int
    stage_index: int
    produced_token_count: int
    device_state: GRDeviceStateRef
    finished: bool
    terminal_result: BeamBatchResult | None = None
    resources_released: bool = False
    owner_generation: int = 0


@dataclass(frozen=True, slots=True)
class GRStageCompletion:
    """GR state derived from one native vLLM ``ModelRunnerOutput``."""

    session_id: str
    stage_index: int
    finished: bool
    device_state: GRDeviceStateRef | None = None
    model_output: ModelRunnerOutput | None = None
    error: str | None = None


class GRAsyncSchedulerInterface(Protocol):
    """Hooks added by patching vLLM Scheduler; this is not a second scheduler."""

    def add_gr_session(self, request: BeamBatchRequest) -> GRSessionState:
        """Create one persistent scheduler state and add it once."""
        ...

    def can_issue_gr_stage(self, session: GRSessionState) -> bool:
        """Whether another stage may be scheduled without consuming a Future.

        Native output placeholders control token progress. The GR hook also
        fences ownership and limits in-flight dispatches to the look-ahead depth.
        """
        ...

    def attach_gr_stage_metadata(
        self,
        scheduler_output: SchedulerOutput,
        session: GRSessionState,
    ) -> GRStageMetadata | None:
        """Attach one GR stage to the output of native ``Scheduler.schedule``."""
        ...

    def update_gr_from_output(
        self,
        session: GRSessionState,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> GRStageCompletion:
        """Consume a matched Worker receipt and update GR resource lifecycle state."""
        ...

    def finish_gr_session(
        self,
        session: GRSessionState,
        status: GRSessionStatus,
    ) -> BeamBatchResult | None:
        """On completion/cancel/error, stop new stages and drain issued work.

        Scheduler and Worker drain or invalidate successors without overwriting
        terminal Beam state; native accounting consumes their placeholders.
        Return None until all device work and terminal copies are complete;
        only then release KV/workspace and return the retained result once.
        Host-observed finished=True alone never authorizes resource release.
        """
        ...


class GRModelExecutorInterface(Protocol):
    """Native vLLM ModelExecutor call extended by the Worker patch."""

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        *,
        non_block: bool,
    ) -> Future[ModelRunnerOutput] | ModelRunnerOutput:
        """Use vLLM's entry; non-blocking execution returns a Future."""
        ...


class GREngineCoreInterface(Protocol):
    """GR hooks installed on vLLM EngineCore by monkey patching."""

    def submit_gr_session(self, request: BeamBatchRequest) -> str:
        """Enqueue one frontend submission and return its session ID.

        Both online and offline adapters call this same entry exactly once.
        """
        ...

    def step_with_batch_queue(self) -> Any:
        """Patched native vLLM asynchronous EngineCore iteration.

        EngineCore adds GR session transitions around the existing schedule,
        ``execute_model(non_block=True)``, queue, and update-from-output flow.
        Consecutive stages of the same session may coexist in the native queue
        when ``can_issue_gr_stage`` permits look-ahead. The upstream return
        contract remains unchanged.
        """
        ...

    def finish_gr_session(self, session_id: str) -> BeamBatchResult:
        """Return the retained terminal result after Scheduler/device drain."""
        ...

    def abort_gr_session(self, session_id: str) -> None: ...


def validate_gr_dispatch(
    scheduler_output: Any,
    *,
    previous: GRStageMetadata | None = None,
    binding: Any | None = None,
) -> tuple[GRStageMetadata, ...]:
    """Validate the whole metadata map before any Worker resource mutation."""
    mapping = getattr(scheduler_output, "gr_stage_metadata", None)
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("GR dispatch requires a nonempty metadata mapping")
    scheduled = scheduler_output.num_scheduled_tokens
    if mapping.keys() != scheduled.keys() or getattr(scheduler_output, "beam_data", {}):
        raise ValueError("GR dispatch cannot mix ordinary or legacy Beam requests")
    scheduled_total = sum(scheduled.values())
    if getattr(scheduler_output, "total_num_scheduled_tokens", scheduled_total) != scheduled_total:
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
        if not m.produces_output:
            raise ValueError("beam_search_v1 requires full Prefill")
        params = m.beam_params
        capacity = m.execution_capacity or GRExecutionCapacity(params.beam_width)
        capacity.validate(params)
        if (
            not math.isfinite(params.temperature)
            or params.temperature < 0
            or params.constraint_resource_id != DEFAULT_CONSTRAINT_RESOURCE_ID
        ):
            raise ValueError("GR requires fixed W<=256, effective N=1/2/3 and Constraint Table")
        if not 0 <= m.stage_index < params.max_tokens:
            raise ValueError("GR logical stage exceeds generation budget")
        if m.output_device_state != GRDeviceStateRef(key, m.stage_index):
            raise ValueError("GR output device-state identity mismatch")
        if m.kind is GRStageKind.PREFILL:
            if m.stage_index != 0 or m.decode_step is not None or m.input_device_state is not None:
                raise ValueError("GR rejects recompute after generation")
        elif m.kind is GRStageKind.DECODE:
            if (
                m.decode_step != m.stage_index - 1
                or m.input_device_state != GRDeviceStateRef(key, m.stage_index - 1)
                or scheduled[key] != 1
            ):
                raise ValueError("invalid GR Decode dependency or logical token count")
        else:
            raise ValueError("unknown GR stage kind")
        if m.is_last_stage != (m.stage_index == params.max_tokens - 1):
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


__all__ = [
    "GRAsyncSchedulerInterface",
    "GRDeviceStateRef",
    "GRExecutionCapacity",
    "GREngineCoreInterface",
    "GRModelExecutorInterface",
    "GRResourceStatus",
    "GRRetireMetadata",
    "GRRetireResult",
    "GRSessionState",
    "GRSessionStatus",
    "GRStageCompletion",
    "GRStageKind",
    "GRStageMetadata",
    "GRWorkerResult",
    "validate_gr_device_config",
    "validate_gr_dispatch",
    "validate_gr_execution_config",
    "validate_gr_request",
]
