# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contracts for EngineCore-owned Asynchronous scheduling Beam Search V1 execution and retirement."""

from __future__ import annotations

import math
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, cast

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_gr.config import BeamCandidateMode, GRConfig
from vllm_gr.v1.beam.batch_contracts import (
    SUPPORTED_BEAM_BATCH_SIZES,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
    beam_batch_protocol_version,
    validate_beam_batch_request,
    validate_beam_candidate_source,
)
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView


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


def _validate_gr_generation_params(params: BeamBatchParams) -> None:
    """Validate V1 generation semantics without constructing physical capacity."""
    validate_beam_candidate_source(params)
    if params.candidate_mode == BeamCandidateMode.FULL_VOCAB:
        if (
            params.temperature != 0
            or params.ignore_stop_tokens is not False
            or params.length_penalty != 1.0
        ):
            raise ValueError(
                "full_vocab beam_search_v1 requires temperature=0, "
                "ignore_eos=False and length_penalty=1"
            )
        if len(params.stop_token_ids) > 1:
            raise ValueError("full_vocab supports only the tokenizer EOS stop token")
    elif params.ignore_stop_tokens is not True:
        raise ValueError("beam_search_v1 requires ignore_eos=True")
    if params.catalog_strategy is not None:
        raise ValueError("beam_search_v1 does not support catalog_strategy")
    if (
        type(params.beam_width) is not int
        or not 1 <= params.beam_width <= 256
        or type(params.max_tokens) is not int
        or params.max_tokens not in (1, 2, 3)
    ):
        raise ValueError(
            "Asynchronous scheduling requires the startup beam width and effective N=1/2/3"
        )


def validate_gr_candidate_mode(params: BeamBatchParams, config: Any) -> None:
    """Bind request/dispatch candidate semantics to the engine startup choice."""
    validate_beam_candidate_source(params)
    expected = GRConfig.from_vllm_config(config).beam_candidate_mode
    if params.candidate_mode != expected:
        raise ValueError(
            f"Beam candidate_mode={params.candidate_mode!r} does not match "
            f"engine candidate_mode={expected.value!r}"
        )


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
        _validate_gr_generation_params(params)
        if params.beam_width != self.beam_width:
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
    validate_gr_execution_runtime(config)


def validate_gr_execution_runtime(config: Any) -> None:
    """Validate runtime capabilities after the persistent protocol is admitted."""
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
    """Reject unavailable adapters without importing a device runtime."""
    device = getattr(getattr(config, "device_config", None), "device", None)
    if device is not None and str(device).split(":")[0] not in ("cuda", "npu"):
        raise ValueError(
            "Asynchronous scheduling requires a CUDA ModelRunner or NPU ModelRunner V1"
        )


def validate_gr_request(request: BeamBatchRequest, config: Any = None) -> None:
    """Validate shared V1 semantics before native ADD or Worker allocation."""
    validate_beam_batch_request(request)
    if config is not None:
        validate_gr_candidate_mode(request.beam_params, config)
    if len(request.items) not in SUPPORTED_BEAM_BATCH_SIZES:
        raise ValueError(
            "beam_search_v1 supports batch sizes "
            f"{SUPPORTED_BEAM_BATCH_SIZES}, got {len(request.items)}"
        )
    if request.items[0].request.request_id != request.batch_id:
        raise ValueError("batch_id and the first native request_id must match")
    for item in request.items:
        native = item.request
        params = native.sampling_params
        if native.prompt_embeds is not None or not native.prompt_token_ids:
            raise ValueError("beam_search_v1 requires a nonempty token prompt")
        if native.mm_features or native.lora_request is not None:
            raise ValueError("beam_search_v1 does not support multimodal inputs or LoRA")
        if params is None:
            raise ValueError("beam_search_v1 requires sampling parameters")
        if request.beam_params.candidate_mode == BeamCandidateMode.FULL_VOCAB:
            if params.ignore_eos is not True or params.stop_token_ids:
                raise ValueError(
                    "full_vocab native accounting requires ignore_eos=True and no stop tokens; "
                    "Worker Beam state owns EOS termination"
                )
            if params.temperature != request.beam_params.temperature:
                raise ValueError("native and Beam temperature must match")
        elif params.ignore_eos is not True:
            raise ValueError("beam_search_v1 requires ignore_eos=True")
        if params.max_tokens != request.beam_params.max_tokens or params.n != 1:
            raise ValueError("native and Beam generation budgets must match and n must be 1")
    _validate_gr_generation_params(request.beam_params)


@dataclass(frozen=True, slots=True)
class GRDeviceStateRef:
    """Reference to Beam state that remains owned by the Worker device."""

    session_id: str
    producer_stage_index: int


@dataclass(slots=True)
class GRSessionState:
    """EngineCore scheduling state for one fixed-membership GR batch session.

    Native Requests own token progress and the effective budget. Frozen dispatch
    records retain dependencies and receipt identity after native termination.
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
        if self.session_id != self.request.batch_id:
            raise ValueError("session_id must match the Beam batch_id")
        if not self.request.items:
            raise ValueError("Beam Search V1 batch must not be empty")
        request_ids = self.request_ids
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Beam Search V1 batch request IDs must be unique")
        if self.max_decode_steps < 0:
            raise ValueError("max_decode_steps must be non-negative")
        if self.max_decode_steps != self.request.beam_params.max_tokens - 1:
            raise ValueError("max_decode_steps must equal effective max_tokens minus 1")

    @property
    def total_stages(self) -> int:
        return cast(int, self.request.beam_params.max_tokens)

    @property
    def issued_stages(self) -> int:
        """Diagnostic count derived from physical dispatch records."""
        return len(self._dispatch_records)

    @property
    def completed_stages(self) -> int:
        return len(self._dispatch_records) - len(self._in_flight_dispatches)

    @property
    def request_ids(self) -> tuple[str, ...]:
        """Stable caller-defined member order for every batch stage."""
        return tuple(item.request.request_id for item in self.request.items)

    @property
    def batch_size(self) -> int:
        return len(self.request.items)

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
    steps 0..N-2. ``dispatch_id`` identifies the physical invocation separately
    from native logical progress. Worker echoes both identities and generation.

    SchedulerOutput.gr_stage_metadata is keyed by session_id. Asynchronous scheduling admits one
    execution owner; owner_generation distinguishes reuse of a session ID.
    session_init carries the caller-formed batch request (Beam parameters, output
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
    # One row of prompt block IDs per request_id, in the same order. GR
    # execution supports one uniform KV cache group; for B=1 this retains
    # the existing single-row representation used by V1 and V2 runners.
    # None means block IDs were not published.
    kv_blocks: tuple[tuple[int, ...], ...] | None = None
    # The batch is formed by the caller and remains ordered and immutable for
    # the complete session. B=1 keeps the original representation.
    request_ids: tuple[str, ...] = ()
    num_prompt_tokens_by_request: tuple[int, ...] = ()


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

        GR bounds outstanding physical dispatches; native placeholders own progress.
        This is what permits Prefill-to-Decode and Decode-to-Decode overlap.
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
        """Match a receipt and retire its dispatch and KV lease exactly once."""
        ...

    def finish_gr_session(
        self,
        session: GRSessionState,
        status: GRSessionStatus,
    ) -> BeamBatchResult | None:
        """On completion/cancel/error, stop new stage issuance.

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


def validate_dispatch(
    scheduler_output: Any,
    *,
    previous: GRStageMetadata | None = None,
    binding: Any = None,
    bindings: tuple[Any, ...] | None = None,
) -> tuple[GRStageMetadata, ...]:
    """Validate the whole metadata map before any Worker resource mutation."""
    mapping = getattr(scheduler_output, "gr_stage_metadata", None)
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("GR dispatch requires a nonempty metadata mapping")
    scheduled = scheduler_output.num_scheduled_tokens
    if len(mapping) != 1 or getattr(scheduler_output, "beam_data", {}):
        raise ValueError("GR dispatch cannot mix ordinary or legacy Beam requests")
    if any(type(count) is not int or count <= 0 for count in scheduled.values()):
        raise ValueError("GR scheduled counts must be positive integers")
    if getattr(scheduler_output, "total_num_scheduled_tokens", sum(scheduled.values())) != sum(
        scheduled.values()
    ):
        raise ValueError("GR dispatch total disagrees with native scheduled counts")
    stages = tuple(mapping.values())
    for key, m in mapping.items():
        if not isinstance(m, GRStageMetadata) or key != m.session_id:
            raise ValueError("invalid GR metadata identity")
        view = BeamBatchExecutionView.from_stage_metadata(m)
        if view.batch_size not in SUPPORTED_BEAM_BATCH_SIZES or set(scheduled) != set(
            view.request_ids
        ):
            raise ValueError("GR requires one immutable logical batch with B=1,2,4")
        for number in (m.dispatch_id, m.stage_index, m.num_prompt_tokens, m.owner_generation):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(
                    "GR dispatch indices and prompt length must be nonnegative integers"
                )
        if not m.num_prompt_tokens or m.produces_output is not True:
            raise ValueError("GR requires a nonempty prompt and complete, output-producing Prefill")
        new_requests = {r.req_id: r for r in getattr(scheduler_output, "scheduled_new_reqs", ())}
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        for request_id, prompt_tokens in zip(view.request_ids, view.num_prompt_tokens_by_request):
            computed = None
            if request_id in new_requests:
                native = new_requests[request_id]
                if (
                    m.kind is GRStageKind.DECODE
                    or len(native.prompt_token_ids or ()) != prompt_tokens
                ):
                    raise ValueError("GR native request identity or prompt geometry changed")
                computed = native.num_computed_tokens
            elif cached is not None and request_id in cached.req_ids:
                if request_id in cached.resumed_req_ids and m.stage_index > 0:
                    raise ValueError("GR rejects recompute after generation")
                computed = cached.num_computed_tokens[cached.req_ids.index(request_id)]
            if computed is None and (new_requests or cached is not None):
                raise ValueError("GR metadata has no matching native request")
            if computed is not None:
                if type(computed) is not int or computed < 0:
                    raise ValueError("invalid native computed token count")
                if computed + scheduled[request_id] != prompt_tokens + m.stage_index:
                    raise ValueError(
                        "GR stage disagrees with native token extent; partial Prefill/recompute"
                    )
                if m.kind is GRStageKind.PREFILL and computed >= prompt_tokens:
                    raise ValueError("GR Prefill requires remaining prompt tokens")
        if not isinstance(m.is_last_stage, bool) or (
            m.decode_step is not None and type(m.decode_step) is not int
        ):
            raise ValueError("GR dispatch flags and step have invalid types")
        params = m.beam_params
        capacity = m.execution_capacity or GRExecutionCapacity(params.beam_width)
        capacity.validate(params)
        if (
            type(params.beam_width) is not int
            or not 1 <= params.beam_width <= 256
            or type(params.max_tokens) is not int
            or params.max_tokens not in (1, 2, 3)
            or not math.isfinite(params.temperature)
            or params.temperature < 0
        ):
            raise ValueError("GR requires fixed W<=256 and effective N=1/2/3")
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
                or any(scheduled[request_id] != 1 for request_id in view.request_ids)
            ):
                raise ValueError("invalid GR Decode dependency or logical token count")
        else:
            raise ValueError("unknown GR stage kind")
        if m.is_last_stage != (m.produces_output and m.stage_index == params.max_tokens - 1):
            raise ValueError("GR last-stage flag disagrees with effective budget")
        if m.session_init is not None:
            request = m.session_init
            validate_gr_request(request)
            if (
                m.kind is not GRStageKind.PREFILL
                or request.protocol_version != beam_batch_protocol_version(params)
                or len(request.items) != view.batch_size
                or request.batch_id != key
                or tuple(item.item_index for item in request.items) != tuple(range(view.batch_size))
                or tuple(item.request.request_id for item in request.items) != view.request_ids
                or tuple(len(item.request.prompt_token_ids or ()) for item in request.items)
                != view.num_prompt_tokens_by_request
                or request.beam_params != params
                or request.output_options != m.output_options
            ):
                raise ValueError("GR session_init disagrees with dispatch metadata")
        if binding is not None and key != binding.session_id:
            raise ValueError("a second active GR session cannot use shared execution scratch")
        if bindings is not None and (
            tuple(bound.native_request_id for bound in bindings) != view.request_ids
            or any(bound.session_id != key for bound in bindings)
        ):
            raise ValueError("GR batch changed the shared execution binding")
        if previous is None:
            if m.dispatch_id != 0 or m.kind is not GRStageKind.PREFILL:
                raise ValueError("GR session must start with dispatch 0 Prefill")
        elif (
            m.dispatch_id != previous.dispatch_id + 1
            or m.stage_index != previous.stage_index + 1
            or m.owner_generation != previous.owner_generation
            or params != previous.beam_params
            or m.output_options != previous.output_options
            or m.num_prompt_tokens != previous.num_prompt_tokens
        ):
            raise ValueError("GR dispatch order, configuration or prompt binding changed")
        if previous is not None:
            BeamBatchExecutionView.from_stage_metadata(previous).assert_same_logical_batch(view)
    return stages


__all__ = [
    "GRAsyncSchedulerInterface",
    "GRDeviceStateRef",
    "GREngineCoreInterface",
    "GRExecutionCapacity",
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
    "validate_dispatch",
    "validate_gr_candidate_mode",
    "validate_gr_device_config",
    "validate_gr_execution_config",
    "validate_gr_execution_runtime",
    "validate_gr_request",
]
