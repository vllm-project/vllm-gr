# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contracts for EngineCore-owned Asynchronous scheduling Beam Search V1 execution and retirement."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
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
    device = getattr(getattr(config, "device_config", None), "device", None)
    if device is not None and str(device).split(":")[0] != "cuda":
        raise ValueError("Asynchronous scheduling requires GPU ModelRunner V1")


@dataclass(frozen=True, slots=True)
class GRDeviceStateRef:
    """Reference to Beam state that remains owned by the Worker device."""

    session_id: str
    producer_stage_index: int


@dataclass(slots=True)
class GRSessionState:
    """EngineCore scheduling state for one B=1 GR session.

    ``issued_stages`` and ``completed_stages`` count only output-producing
    logical stages. Physical dispatches, including no-output initial Prefill
    chunks, are tracked separately by private EngineCore state.
    ``output_placeholders`` reserves results that have been scheduled but are
    not yet confirmed on the host.

    beam_params.max_tokens is the effective sampling budget, including the
    first token produced by Prefill. A2 adjusts the public budget once for
    any preinserted begin token or reserved, forced end token; a sampled EOS
    consumes budget normally. max_decode_steps must equal this budget minus 1.
    latest_device_state tracks the previous issued stage, not host completion.
    """

    session_id: str
    request: BeamBatchRequest
    max_decode_steps: int
    status: GRSessionStatus = GRSessionStatus.WAITING
    issued_stages: int = 0
    completed_stages: int = 0
    output_placeholders: int = 0
    confirmed_token_count: int = 0
    pending_device_token_count: int = 0
    latest_device_state: GRDeviceStateRef | None = None
    terminal_result_pending: bool = False
    resource_status: GRResourceStatus = GRResourceStatus.WAITING
    owner_generation: int = 0
    retirement: GRRetireMetadata | None = None
    worker_resources_allocated: bool = False
    # EngineCore-private lifecycle state. These fields are excluded from the
    # constructor and never cross the Scheduler/Worker boundary.
    _next_dispatch_id: int = field(default=0, init=False, repr=False)
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
        if not 0 <= self.completed_stages <= self.issued_stages <= self.total_stages:
            raise ValueError("stage counters must satisfy 0 <= completed <= issued <= total")
        if self.output_placeholders < 0:
            raise ValueError("output_placeholders must be non-negative")
        if self.confirmed_token_count < 0:
            raise ValueError("confirmed_token_count must be non-negative")
        if self.pending_device_token_count < 0:
            raise ValueError("pending_device_token_count must be non-negative")
        if self.pending_device_token_count > self.output_placeholders:
            raise ValueError("pending_device_token_count must not exceed output_placeholders")

    @property
    def total_stages(self) -> int:
        return 1 + self.max_decode_steps

    @property
    def in_flight_stages(self) -> int:
        return self.issued_stages - self.completed_stages

    @property
    def next_stage_kind(self) -> GRStageKind | None:
        """Return the next logical stage; stage 0 is the only Prefill."""
        if self.status not in (GRSessionStatus.WAITING, GRSessionStatus.RUNNING):
            return None
        if self.issued_stages >= self.total_stages:
            return None
        if self.issued_stages == 0:
            return GRStageKind.PREFILL
        return GRStageKind.DECODE


@dataclass(frozen=True, slots=True)
class GRStageMetadata:
    """GR metadata attached to one native vLLM ``SchedulerOutput``.

    Stage 0 is Prefill. For effective budget N, stages 1..N-1 are Decode
    steps 0..N-2. ``dispatch_id`` identifies every physical invocation,
    including no-output initial Prefill chunks; ``stage_index`` advances only
    for output-producing stages. Worker echoes both IDs in GRWorkerResult.

    SchedulerOutput.gr_stage_metadata is keyed by session_id. Asynchronous scheduling admits one
    execution owner; owner_generation distinguishes reuse of a session ID.
    session_init carries the existing B=1 request (Beam parameters, output
    options and beam_params.constraint_resource_id) once, with Prefill only.
    Native SchedulerOutput supplies prompt/KV data. Stage k > 0 references
    (session_id, k - 1) even before host completion; A4 orders producer writes
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
    # Per-KV-group prompt block ids, published by the scheduler with every
    # dispatch.  Runners without host-side request state (V2) build their beam
    # attention block table from this instead of reaching into scheduler
    # structures; V1 may still fall back to its own request state when None.
    kv_blocks: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True, slots=True)
class GRWorkerResult:
    """A4/A5 attach ModelRunnerOutput.gr_v1_results: dict[str, GRWorkerResult].

    Keys match session_id, including multiple sessions in one native output.
    Both GR attachments must survive executor IPC; async output wrappers must
    preserve gr_v1_results on the inner ModelRunnerOutput returned by get_output.
    Intermediate token IDs, parents and scores remain device-resident.
    terminal_result is supplied once for the terminating stage, with batch_id
    matching session_init.batch_id. A3 retains it until successors are drained,
    then delivers it through the frontend's wait_final channel.
    resources_released is supplied once by the last terminal output consumer;
    it replaces the normal-path zero-token retirement round trip.
    """

    session_id: str
    dispatch_id: int
    stage_index: int
    produced_token_count: int
    device_state: GRDeviceStateRef
    finished: bool
    terminal_result: BeamBatchResult | None = None
    resources_released: bool = False


@dataclass(frozen=True, slots=True)
class GRStageCompletion:
    """GR state derived from one native vLLM ``ModelRunnerOutput``."""

    session_id: str
    stage_index: int
    finished: bool
    confirmed_token_count: int
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

        A3 implements the bounded look-ahead rule with output placeholders.
        This is what permits Prefill-to-Decode and Decode-to-Decode overlap.
        """
        ...

    def reserve_gr_output_placeholder(
        self,
        session: GRSessionState,
        num_tokens: int,
    ) -> None:
        """Reserve unconfirmed output before scheduling the next GR stage."""
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
        """Confirm tokens, consume placeholders, and update device-state refs."""
        ...

    def finish_gr_session(
        self,
        session: GRSessionState,
        status: GRSessionStatus,
    ) -> BeamBatchResult | None:
        """On EOS/cancel/error, enter FINISHING and stop new stage issuance.

        A3/A4 must drain or invalidate issued successors without overwriting
        terminal Beam state, retiring their placeholders and completions.
        Return None until all device work and terminal copies are complete;
        only then release KV/workspace and return the retained result once.
        Host-observed finished=True alone never authorizes resource release.
        """
        ...


class GRModelExecutorInterface(Protocol):
    """Native vLLM ModelExecutor call extended by the A4 Worker patch."""

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

        A3 adds GR session transitions around the existing schedule,
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
    "validate_gr_execution_config",
]
