# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Passive contracts and fail-closed validation for Beam batch execution.

The types in this module describe a future Worker-owned Beam batch run. They
do not register an EngineCore message, reserve Scheduler resources, allocate
KV cache, or invoke a ModelRunner.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Annotated, Literal, NoReturn, TypeGuard

import msgspec
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.config import MAX_BEAM_WIDTH

BEAM_BATCH_PROTOCOL_VERSION = 1
SUPPORTED_BEAM_BATCH_SIZES = (1, 2, 4)
DEFAULT_CONSTRAINT_RESOURCE_ID = "default"
# Terminal failure code for a request the engine refused before the session
# started. It is the one code the frontend reports as a client error; every
# other code means the engine or its Worker failed.
ADMISSION_REJECTED_CODE = "admission_rejected"
# Terminal result key on ``EngineCoreOutput.kv_transfer_params``. The field is
# untyped IPC, so every reader has to tolerate the shapes in
# ``admission_rejection`` below.
GR_BATCH_RESULT_PARAM = "gr_batch_result"


class BeamContractError(ValueError):
    """Raised when a Beam batch value violates the contract."""


class BeamBatchItem(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    item_index: int
    request: EngineCoreRequest


class BeamBatchParams(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    omit_defaults=True,  # type: ignore[call-arg]
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    beam_width: int
    max_tokens: int
    temperature: float
    stop_token_ids: tuple[int, ...]
    ignore_stop_tokens: bool
    constraint_resource_id: str = DEFAULT_CONSTRAINT_RESOURCE_ID
    length_penalty: float = 1.0
    catalog_strategy: Literal["fill", "null"] | None = None


class BeamOutputOptions(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    num_return_sequences: int
    include_stop_token: bool
    include_cumulative_logprob: bool
    include_parent_beam_indices: bool


class BeamBatchRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    protocol_version: int
    batch_id: str
    items: tuple[BeamBatchItem, ...]
    beam_params: BeamBatchParams
    output_options: BeamOutputOptions


class BeamPrefillData(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    item_index: int
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_scheduled_tokens: int


class BeamBatchPlan(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    protocol_version: int
    batch_id: str
    items: tuple[BeamPrefillData, ...]
    beam_params: BeamBatchParams
    output_options: BeamOutputOptions


class BeamOutputSequence(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    token_ids: tuple[Annotated[int, msgspec.Meta(ge=0)], ...]
    cumulative_logprob: (
        Annotated[float, msgspec.Meta(ge=-sys.float_info.max, le=sys.float_info.max)] | None
    )
    parent_beam_indices: tuple[Annotated[int, msgspec.Meta(ge=0)], ...]
    finish_reason: Literal["stop", "length"]
    stop_token_id: int | None


class BeamRequestOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    item_index: int
    request_id: str
    sequences: tuple[BeamOutputSequence, ...]


class BeamBatchOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    items: tuple[BeamRequestOutput, ...]


class BeamBatchFailure(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    code: str
    message: str
    item_index: int | None


class BeamBatchResult(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    protocol_version: int
    batch_id: str
    output: BeamBatchOutput | None
    failure: BeamBatchFailure | None


def admission_rejected(batch_id: str, message: str) -> BeamBatchResult:
    """The terminal result for a session the engine refused before starting.

    Admission rejects a request on its own contents -- the caller's answer to
    act on -- while every other terminal failure describes a broken engine or
    Worker. Both travel as the same payload, so the code is what keeps the two
    apart at the frontend.
    """
    return BeamBatchResult(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=batch_id,
        output=None,
        failure=BeamBatchFailure(ADMISSION_REJECTED_CODE, message, None),
    )


def admission_rejection(output: object) -> str | None:
    """The message of the admission rejection carried by ``output``, if any.

    A serving frontend sees its request end on a bare ``FinishReason.ERROR``
    output, whose reason it would otherwise have to report as a server fault.
    This recovers the reason the engine attached, and returns ``None`` for
    everything else -- outputs without a terminal result, results that failed
    for another reason, and payloads that do not parse.

    ``kv_transfer_params`` is an untyped IPC field, so a multiprocess
    deployment delivers the array-like Struct as the plain list it encodes to
    while an in-process caller holds the Struct itself.
    """
    params = getattr(output, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return None
    result = params.get(GR_BATCH_RESULT_PARAM)
    if isinstance(result, BeamBatchResult):
        failure = result.failure
    elif isinstance(result, (list, tuple)):
        try:
            failure = msgspec.convert(result, type=BeamBatchResult).failure
        except msgspec.ValidationError:
            return None
    else:
        return None
    if failure is None or failure.code != ADMISSION_REJECTED_CODE:
        return None
    return failure.message


@dataclass(frozen=True, slots=True)
class BeamBatchLimits:
    """Non-wire deployment limits for one Beam batch."""

    max_batch_size: int
    max_beam_width: int
    max_tokens: int
    max_num_beam_rows: int
    max_num_batched_tokens: int
    max_model_len: int
    supported_constraint_resource_ids: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise BeamContractError(message)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(name: str, value: object) -> int:
    if not _is_int(value) or value <= 0:
        _fail(f"{name} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if not _is_int(value) or value < 0:
        _fail(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        _fail(f"{name} must be a bool, got {value!r}")
    return value


def _protocol_version(value: object) -> None:
    if not _is_int(value) or value != BEAM_BATCH_PROTOCOL_VERSION:
        _fail(f"protocol_version must equal {BEAM_BATCH_PROTOCOL_VERSION}, got {value!r}")


def _validate_beam_params(params: BeamBatchParams) -> None:
    if not isinstance(params, BeamBatchParams):
        _fail("beam_params must be BeamBatchParams")
    beam_width = _positive_int("beam_params.beam_width", params.beam_width)
    if beam_width > MAX_BEAM_WIDTH:
        _fail(f"beam_params.beam_width must be <= {MAX_BEAM_WIDTH}, got {beam_width}")
    _positive_int("beam_params.max_tokens", params.max_tokens)
    if (
        isinstance(params.temperature, bool)
        or not isinstance(params.temperature, (int, float))
        or not math.isfinite(float(params.temperature))
        or params.temperature < 0
    ):
        _fail(
            "beam_params.temperature must be a finite non-negative number, got "
            f"{params.temperature!r}"
        )
    _bool("beam_params.ignore_stop_tokens", params.ignore_stop_tokens)
    if (
        isinstance(params.length_penalty, bool)
        or not isinstance(params.length_penalty, (int, float))
        or not math.isfinite(float(params.length_penalty))
    ):
        _fail(f"beam_params.length_penalty must be a finite number, got {params.length_penalty!r}")
    if params.catalog_strategy not in (None, "fill", "null"):
        _fail("beam_params.catalog_strategy must be 'fill', 'null', or None")
    seen_stop_ids: set[int] = set()
    for index, token_id in enumerate(params.stop_token_ids):
        normalized = _non_negative_int(f"beam_params.stop_token_ids[{index}]", token_id)
        if normalized in seen_stop_ids:
            _fail("beam_params.stop_token_ids must be unique")
        seen_stop_ids.add(normalized)
    if params.constraint_resource_id != DEFAULT_CONSTRAINT_RESOURCE_ID:
        _fail(
            "Beam batch protocol version 1 only admits "
            f"constraint_resource_id={DEFAULT_CONSTRAINT_RESOURCE_ID!r}"
        )


def _validate_output_options(options: BeamOutputOptions, *, beam_width: int) -> None:
    if not isinstance(options, BeamOutputOptions):
        _fail("output_options must be BeamOutputOptions")
    num_sequences = _positive_int(
        "output_options.num_return_sequences", options.num_return_sequences
    )
    if num_sequences > beam_width:
        _fail(
            "output_options.num_return_sequences must be <= beam_params.beam_width "
            f"({beam_width}), got {num_sequences}"
        )
    _bool("output_options.include_stop_token", options.include_stop_token)
    _bool(
        "output_options.include_cumulative_logprob",
        options.include_cumulative_logprob,
    )
    _bool(
        "output_options.include_parent_beam_indices",
        options.include_parent_beam_indices,
    )


def validate_beam_batch_request(request: BeamBatchRequest) -> None:
    """Validate a backend-neutral Beam batch request."""

    if not isinstance(request, BeamBatchRequest):
        _fail("request must be BeamBatchRequest")
    _protocol_version(request.protocol_version)
    _non_empty_string("batch_id", request.batch_id)
    _validate_beam_params(request.beam_params)
    _validate_output_options(
        request.output_options,
        beam_width=request.beam_params.beam_width,
    )
    if not request.items:
        _fail("items must contain at least one BeamBatchItem")

    request_ids: set[str] = set()
    for expected_index, item in enumerate(request.items):
        if not isinstance(item, BeamBatchItem):
            _fail("items must contain BeamBatchItem values")
        if item.item_index != expected_index:
            _fail(
                "items must be ordered with contiguous item_index values from zero; "
                f"expected {expected_index}, got {item.item_index!r}"
            )
        if not isinstance(item.request, EngineCoreRequest):
            _fail("item.request must be an EngineCoreRequest")
        request_id = _non_empty_string(
            f"items[{expected_index}].request.request_id",
            item.request.request_id,
        )
        if request_id in request_ids:
            _fail("item request_id values must be unique within a batch")
        request_ids.add(request_id)
        if item.request.sampling_params is None:
            _fail("Beam batch items require sampling_params")
        if item.request.pooling_params is not None:
            _fail("Beam batch items cannot carry pooling_params")


def validate_beam_batch_limits(limits: BeamBatchLimits) -> None:
    if not isinstance(limits, BeamBatchLimits):
        _fail("limits must be BeamBatchLimits")
    for name in (
        "max_batch_size",
        "max_beam_width",
        "max_tokens",
        "max_num_beam_rows",
        "max_num_batched_tokens",
        "max_model_len",
    ):
        _positive_int(f"limits.{name}", getattr(limits, name))
    if limits.max_beam_width > MAX_BEAM_WIDTH:
        _fail(f"limits.max_beam_width must be <= {MAX_BEAM_WIDTH}, got {limits.max_beam_width}")
    if not limits.supported_constraint_resource_ids:
        _fail("limits.supported_constraint_resource_ids must not be empty")
    resource_ids: set[str] = set()
    for resource_id in limits.supported_constraint_resource_ids:
        normalized = _non_empty_string("supported_constraint_resource_id", resource_id)
        if normalized in resource_ids:
            _fail("limits.supported_constraint_resource_ids must be unique")
        resource_ids.add(normalized)


def _validate_batch_geometry(
    *,
    batch_size: int,
    params: BeamBatchParams,
    limits: BeamBatchLimits,
    max_num_seqs: int,
) -> None:
    sequence_limit = _positive_int("max_num_seqs", max_num_seqs)
    if batch_size > limits.max_batch_size:
        _fail(f"batch size {batch_size} exceeds max_batch_size {limits.max_batch_size}")
    if batch_size > sequence_limit:
        _fail(f"batch size {batch_size} exceeds max_num_seqs {sequence_limit}")
    if params.beam_width > limits.max_beam_width:
        _fail(f"beam_width {params.beam_width} exceeds max_beam_width {limits.max_beam_width}")
    if params.max_tokens > limits.max_tokens:
        _fail(f"max_tokens {params.max_tokens} exceeds runtime limit {limits.max_tokens}")
    num_beam_rows = batch_size * params.beam_width
    if num_beam_rows > limits.max_num_beam_rows:
        _fail(
            f"B * W beam rows {num_beam_rows} exceeds max_num_beam_rows {limits.max_num_beam_rows}"
        )
    if params.constraint_resource_id not in limits.supported_constraint_resource_ids:
        _fail(
            f"constraint resource {params.constraint_resource_id!r} is not "
            "supported by this runtime"
        )


def validate_beam_batch_request_limits(
    request: BeamBatchRequest,
    limits: BeamBatchLimits,
    *,
    max_num_seqs: int,
) -> None:
    """Validate limits known before native prompt accounting and scheduling."""

    validate_beam_batch_request(request)
    validate_beam_batch_limits(limits)
    _validate_batch_geometry(
        batch_size=len(request.items),
        params=request.beam_params,
        limits=limits,
        max_num_seqs=max_num_seqs,
    )


def validate_beam_batch_plan(
    plan: BeamBatchPlan,
    *,
    expected_request: BeamBatchRequest | None = None,
) -> None:
    """Validate Plan completeness and optional Request correspondence."""

    if not isinstance(plan, BeamBatchPlan):
        _fail("plan must be BeamBatchPlan")
    _protocol_version(plan.protocol_version)
    _non_empty_string("batch_id", plan.batch_id)
    _validate_beam_params(plan.beam_params)
    _validate_output_options(
        plan.output_options,
        beam_width=plan.beam_params.beam_width,
    )
    if not plan.items:
        _fail("plan.items must not be empty")

    request_ids: set[str] = set()
    for expected_index, item in enumerate(plan.items):
        if not isinstance(item, BeamPrefillData):
            _fail("plan.items must contain BeamPrefillData values")
        if item.item_index != expected_index:
            _fail("plan items must be ordered with contiguous item_index values from zero")
        request_id = _non_empty_string(
            f"plan.items[{expected_index}].request_id",
            item.request_id,
        )
        if request_id in request_ids:
            _fail("plan request_id values must be unique")
        request_ids.add(request_id)
        num_prompt_tokens = _positive_int(
            f"plan.items[{expected_index}].num_prompt_tokens",
            item.num_prompt_tokens,
        )
        num_computed_tokens = _non_negative_int(
            f"plan.items[{expected_index}].num_computed_tokens",
            item.num_computed_tokens,
        )
        num_scheduled_tokens = _non_negative_int(
            f"plan.items[{expected_index}].num_scheduled_tokens",
            item.num_scheduled_tokens,
        )
        if num_computed_tokens + num_scheduled_tokens != num_prompt_tokens:
            _fail(
                "full Prefill requires num_computed_tokens + "
                f"num_scheduled_tokens == num_prompt_tokens for item {expected_index}"
            )

    if expected_request is None:
        return
    validate_beam_batch_request(expected_request)
    if plan.protocol_version != expected_request.protocol_version:
        _fail("plan protocol_version must match request")
    if plan.batch_id != expected_request.batch_id:
        _fail("plan batch_id must match request")
    if plan.beam_params != expected_request.beam_params:
        _fail("plan beam_params must match request")
    if plan.output_options != expected_request.output_options:
        _fail("plan output_options must match request")
    if len(plan.items) != len(expected_request.items):
        _fail("plan must contain exactly one item for every request item")
    for prefill, request_item in zip(plan.items, expected_request.items):
        if (
            prefill.item_index != request_item.item_index
            or prefill.request_id != request_item.request.request_id
        ):
            _fail("plan item identity and order must match request")


def validate_beam_batch_plan_scheduler_output(
    plan: BeamBatchPlan,
    scheduler_output: SchedulerOutput,
) -> None:
    """Cross-check the Plan snapshot against native SchedulerOutput."""

    validate_beam_batch_plan(plan)
    if not isinstance(scheduler_output, SchedulerOutput):
        _fail("scheduler_output must be SchedulerOutput")
    if scheduler_output.scheduled_cached_reqs.req_ids:
        _fail("Beam full Prefill Plan cannot contain cached Scheduler request entries")

    new_requests = {data.req_id: data for data in scheduler_output.scheduled_new_reqs}
    planned_ids = {item.request_id for item in plan.items}
    if set(new_requests) != planned_ids:
        _fail("SchedulerOutput new request IDs must exactly match Plan items")
    if set(scheduler_output.num_scheduled_tokens) != planned_ids:
        _fail("SchedulerOutput scheduled-token request IDs must exactly match Plan items")

    for item in plan.items:
        native = new_requests[item.request_id]
        if native.num_computed_tokens != item.num_computed_tokens:
            _fail(f"SchedulerOutput num_computed_tokens must match Plan item {item.item_index}")
        if scheduler_output.num_scheduled_tokens[item.request_id] != item.num_scheduled_tokens:
            _fail(f"SchedulerOutput num_scheduled_tokens must match Plan item {item.item_index}")
    planned_total = sum(item.num_scheduled_tokens for item in plan.items)
    if scheduler_output.total_num_scheduled_tokens != planned_total:
        _fail("SchedulerOutput total_num_scheduled_tokens must match the Plan total")


def validate_beam_batch_plan_limits(
    plan: BeamBatchPlan,
    limits: BeamBatchLimits,
    *,
    max_num_seqs: int,
) -> None:
    """Validate limits that require authoritative Scheduler prompt counts."""

    validate_beam_batch_plan(plan)
    validate_beam_batch_limits(limits)
    _validate_batch_geometry(
        batch_size=len(plan.items),
        params=plan.beam_params,
        limits=limits,
        max_num_seqs=max_num_seqs,
    )
    total_num_scheduled_tokens = sum(item.num_scheduled_tokens for item in plan.items)
    if total_num_scheduled_tokens > limits.max_num_batched_tokens:
        _fail(
            f"scheduled tokens {total_num_scheduled_tokens} exceeds "
            f"max_num_batched_tokens {limits.max_num_batched_tokens}"
        )
    for item in plan.items:
        total_length = item.num_prompt_tokens + plan.beam_params.max_tokens
        if total_length > limits.max_model_len:
            _fail(
                f"item {item.item_index} prompt plus generated length {total_length} "
                f"exceeds max_model_len {limits.max_model_len}"
            )


def _validate_output_sequence(sequence: BeamOutputSequence, *, name: str) -> None:
    if not isinstance(sequence, BeamOutputSequence):
        _fail(f"{name} must be BeamOutputSequence")
    for index, token_id in enumerate(sequence.token_ids):
        _non_negative_int(f"{name}.token_ids[{index}]", token_id)
    if sequence.cumulative_logprob is not None and (
        isinstance(sequence.cumulative_logprob, bool)
        or not isinstance(sequence.cumulative_logprob, (int, float))
        or not math.isfinite(float(sequence.cumulative_logprob))
    ):
        _fail(f"{name}.cumulative_logprob must be finite or None")
    for index, parent_beam_index in enumerate(sequence.parent_beam_indices):
        _non_negative_int(
            f"{name}.parent_beam_indices[{index}]",
            parent_beam_index,
        )
    if sequence.finish_reason not in ("stop", "length"):
        _fail(f"{name}.finish_reason must be 'stop' or 'length'")
    if sequence.finish_reason == "stop":
        _non_negative_int(f"{name}.stop_token_id", sequence.stop_token_id)
    elif sequence.stop_token_id is not None:
        _fail(f"{name}.stop_token_id is only valid for finish_reason='stop'")


def validate_beam_batch_result(
    result: BeamBatchResult,
    *,
    expected_plan: BeamBatchPlan | None = None,
) -> None:
    """Validate the Output/Failure envelope and contextual output invariants."""

    if not isinstance(result, BeamBatchResult):
        _fail("result must be BeamBatchResult")
    _protocol_version(result.protocol_version)
    _non_empty_string("batch_id", result.batch_id)
    has_output = result.output is not None
    has_failure = result.failure is not None
    if has_output == has_failure:
        _fail("exactly one of output and failure must be present")

    if result.failure is not None:
        failure = result.failure
        if not isinstance(failure, BeamBatchFailure):
            _fail("failure must be BeamBatchFailure")
        _non_empty_string("failure.code", failure.code)
        _non_empty_string("failure.message", failure.message)
        if failure.item_index is not None:
            _non_negative_int("failure.item_index", failure.item_index)

    if result.output is not None:
        if not isinstance(result.output, BeamBatchOutput):
            _fail("output must be BeamBatchOutput")
        if not result.output.items:
            _fail("output.items must not be empty")
        request_ids: set[str] = set()
        for expected_index, item in enumerate(result.output.items):
            if not isinstance(item, BeamRequestOutput):
                _fail("output.items must contain BeamRequestOutput values")
            if item.item_index != expected_index:
                _fail("output item_index values must be ordered and contiguous")
            request_id = _non_empty_string(
                f"output.items[{expected_index}].request_id",
                item.request_id,
            )
            if request_id in request_ids:
                _fail("output request_id values must be unique")
            request_ids.add(request_id)
            for sequence_index, sequence in enumerate(item.sequences):
                _validate_output_sequence(
                    sequence,
                    name=f"output.items[{expected_index}].sequences[{sequence_index}]",
                )

    if expected_plan is None:
        return
    validate_beam_batch_plan(expected_plan)
    if result.protocol_version != expected_plan.protocol_version:
        _fail("result protocol_version must match plan")
    if result.batch_id != expected_plan.batch_id:
        _fail("result batch_id must match plan")
    if result.failure is not None and result.failure.item_index is not None:
        if result.failure.item_index >= len(expected_plan.items):
            _fail("failure.item_index is outside the planned batch")
    if result.output is None:
        return

    if len(result.output.items) != len(expected_plan.items):
        _fail("output must contain exactly one item per planned request")
    params = expected_plan.beam_params
    options = expected_plan.output_options
    for item, prefill in zip(result.output.items, expected_plan.items):
        if item.item_index != prefill.item_index or item.request_id != prefill.request_id:
            _fail("output item identity and order must match plan")
        if len(item.sequences) > options.num_return_sequences:
            _fail("output item exceeds num_return_sequences")
        for sequence in item.sequences:
            if options.include_cumulative_logprob != (sequence.cumulative_logprob is not None):
                _fail("cumulative_logprob presence must match include_cumulative_logprob")
            if not options.include_parent_beam_indices and sequence.parent_beam_indices:
                _fail("parent_beam_indices must be empty when they were not requested")
            if options.include_parent_beam_indices:
                if not 1 <= len(sequence.parent_beam_indices) <= params.max_tokens:
                    _fail("parent_beam_indices must contain one entry per generation")
                if any(index >= params.beam_width for index in sequence.parent_beam_indices):
                    _fail("parent_beam_indices values must be less than beam_width")
                expected_token_count = len(sequence.parent_beam_indices)
                if sequence.finish_reason == "stop" and not options.include_stop_token:
                    expected_token_count -= 1
                if len(sequence.token_ids) != expected_token_count:
                    _fail("token_ids length is inconsistent with parent indices/options")
            elif len(sequence.token_ids) > params.max_tokens:
                _fail("output sequence exceeds max_tokens")
            if sequence.finish_reason == "length" and len(sequence.token_ids) != params.max_tokens:
                _fail("length-finished sequence must contain max_tokens token IDs")
            if sequence.finish_reason == "stop":
                if params.ignore_stop_tokens:
                    _fail("stop completion is invalid when ignore_stop_tokens is true")
                if sequence.stop_token_id not in params.stop_token_ids:
                    _fail("stop_token_id must be one of the requested stop tokens")
                if options.include_stop_token:
                    if not sequence.token_ids or sequence.token_ids[-1] != sequence.stop_token_id:
                        _fail("included stop token must be the final token_id")


__all__ = [
    "BEAM_BATCH_PROTOCOL_VERSION",
    "DEFAULT_CONSTRAINT_RESOURCE_ID",
    "SUPPORTED_BEAM_BATCH_SIZES",
    "BeamBatchFailure",
    "BeamBatchItem",
    "BeamBatchLimits",
    "BeamBatchOutput",
    "BeamBatchParams",
    "BeamBatchPlan",
    "BeamBatchRequest",
    "BeamBatchResult",
    "BeamContractError",
    "BeamOutputOptions",
    "BeamOutputSequence",
    "BeamPrefillData",
    "BeamRequestOutput",
    "validate_beam_batch_limits",
    "validate_beam_batch_plan",
    "validate_beam_batch_plan_limits",
    "validate_beam_batch_plan_scheduler_output",
    "validate_beam_batch_request",
    "validate_beam_batch_request_limits",
    "validate_beam_batch_result",
]
