# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Passive contracts and fail-closed validation for Beam batch execution.

Request and Result describe the production SUBMIT_GR_SESSION boundary.
SchedulerOutput carries native scheduling data; no parallel Plan message is
serialized. This module holds data and validation, not runtime ownership.
"""

from __future__ import annotations

import array
import math
import sys
from dataclasses import dataclass
from itertools import repeat
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn, TypeGuard

import msgspec
import numpy as np
from numpy.typing import NDArray
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.config import (
    MAX_BEAM_WIDTH,
    MAX_CANDIDATE_WIDTH,
    SUPPORTED_BEAM_BATCH_SIZES,
)
from vllm_gr.config.beam_widths import (
    SUPPORTED_BEAM_WIDTH_SCHEDULE_STEPS,
    final_retained_width,
    per_parent_width_error,
)

BEAM_BATCH_PROTOCOL_VERSION = 2
DEFAULT_CONSTRAINT_RESOURCE_ID = "default"
# Terminal failure code for a request the engine refused before the session
# started. It is the one code the frontend reports as a client error; every
# other code means the engine or its Worker failed.
ADMISSION_REJECTED_CODE = "admission_rejected"
# Terminal result key on ``EngineCoreOutput.kv_transfer_params``. The field is
# untyped IPC, so every reader has to tolerate the shapes in
# ``admission_rejection`` below.
GR_BATCH_RESULT_PARAM = "gr_batch_result"


class GRAdmissionError(ValueError):
    """An expected request rejection before Scheduler/Worker registration."""


class BeamContractError(ValueError):
    """Raised when a Beam batch value violates the contract."""


class BeamEngineCoreRequest(EngineCoreRequest):
    """GR native request with one canonical ndarray prompt wire field."""

    prompt_token_ids: np.ndarray  # type: ignore[assignment,type-arg]

    if TYPE_CHECKING:
        external_req_id: str | None

    def __post_init__(self) -> None:
        if self.external_req_id is None:
            self.external_req_id = self.request_id


def to_int64_tokens(
    values: object,
    *,
    name: str = "prompt_token_ids",
) -> np.ndarray:
    """Normalize edge token values into the canonical GR token array.

    ndarray inputs must already satisfy the contract because the engine hashes
    raw token bytes: dtype, shape and contiguity are part of the cache-key
    identity. Python sequences unbox once through ``array.array`` (about twice
    as fast as ``np.asarray``) and fail closed on non-integers or int64
    overflow instead of silently truncating.
    """
    if isinstance(values, np.ndarray):
        if (
            values.dtype == np.int64
            and values.ndim == 1
            and values.flags.c_contiguous
            and values.size > 0
        ):
            return values
        raise BeamContractError(f"{name} must be a non-empty 1-D contiguous int64 array")
    if not isinstance(values, (list, tuple, array.array)):
        raise BeamContractError(f"{name} must be an integer sequence or an int64 array")
    try:
        tokens = np.frombuffer(array.array("q", values), dtype=np.int64)
    except (TypeError, OverflowError) as error:
        raise BeamContractError(f"{name} must contain int64 token ids: {error}") from error
    if tokens.size == 0:
        raise BeamContractError(f"{name} must not be empty")
    return tokens


class BeamBatchItem(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    item_index: int
    request: BeamEngineCoreRequest
    trie_id: int = 0

    def __post_init__(self) -> None:
        # Existing frontend/native input processors may still produce lists.
        # Convert exactly once when crossing into the GR request contract.
        if isinstance(self.request, EngineCoreRequest) and not isinstance(
            self.request, BeamEngineCoreRequest
        ):
            values = {
                name: getattr(self.request, name) for name in EngineCoreRequest.__struct_fields__
            }
            if values["prompt_token_ids"] is not None:
                values["prompt_token_ids"] = to_int64_tokens(values["prompt_token_ids"])
            converted = BeamEngineCoreRequest(**values)
            msgspec.structs.force_setattr(self, "request", converted)


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
    # New decoders accept old arrays through the trailing defaults. New arrays
    # still include these fields, so old strict decoders reject them even for
    # fixed-width calls. All participating processes need the same schema.
    dynamic_beam_widths: tuple[int, ...] | None = None
    # Candidates per beam row.  None keeps the pre-decoupling behaviour
    # (K == beam_width).  ``array_like`` structs encode positionally, so this
    # trailing ``None`` is still written as a tenth array element even with
    # ``omit_defaults``: an older strict decoder rejects a new payload outright
    # rather than reading it as "unset".  A new decoder does accept old
    # nine-element arrays through the default.  All participating processes
    # therefore have to run the same schema version; a mixed-version rollout is
    # not supported and needs the protocol version bumped.
    candidate_width: int | None = None
    # Request-level per-stage candidate counts; preserve candidate_width's
    # existing wire position by appending this field after it.
    per_parent_beam_widths: tuple[int, ...] | None = None


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


class BeamSequenceColumns(
    msgspec.Struct, frozen=True, gc=False, forbid_unknown_fields=True, tag="complete_beams_v1"
):
    """Owned little-endian columns for equal-length, non-stopped Beam paths.

    This is an internal transport alternative, independent of a device's final
    buffer layout. Only returned rows and requested columns are transmitted.
    Public clients materialize ordinary sequences unless explicitly requested.
    """

    count: Annotated[int, msgspec.Meta(ge=1)]
    steps: Annotated[int, msgspec.Meta(ge=1)]
    token_ids: bytes
    cumulative_logprobs: bytes | None
    parent_beam_indices: bytes | None


class BeamRequestOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    forbid_unknown_fields=True,
    frozen=True,
    gc=False,
):  # type: ignore[call-arg]
    item_index: int
    request_id: str
    sequences: tuple[BeamOutputSequence, ...] | BeamSequenceColumns


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
    if beam_width > MAX_CANDIDATE_WIDTH and params.candidate_width is None:
        # K follows W when unstated, so an unset K leaves the candidate count
        # implied at the Beam width.  A Beam wider than the candidate ceiling
        # would then describe a K the operators cannot take, and it would be
        # accepted here only to fail at the first workspace lookup.  Internal
        # callers build BeamBatchRequest directly and never pass through
        # GRConfig validation, so the check has to exist here too.
        _fail(
            f"beam_params.beam_width={beam_width} exceeds the candidate width ceiling "
            f"{MAX_CANDIDATE_WIDTH} while candidate_width is unset; K follows the Beam "
            "width in that case. Set beam_params.candidate_width to the per-row "
            "candidate count."
        )
    _positive_int("beam_params.max_tokens", params.max_tokens)
    widths = params.dynamic_beam_widths
    if widths is not None:
        if (
            params.max_tokens not in SUPPORTED_BEAM_WIDTH_SCHEDULE_STEPS
            or not isinstance(widths, tuple)
            or len(widths) != params.max_tokens
        ):
            _fail("beam_params.dynamic_beam_widths must contain one immutable width per step")
        for index, width in enumerate(widths):
            normalized = _positive_int(f"beam_params.dynamic_beam_widths[{index}]", width)
            if normalized > beam_width:
                _fail("beam_params.dynamic_beam_widths entries must not exceed beam_width")
    per_parent_beam_widths = params.per_parent_beam_widths
    if per_parent_beam_widths is not None:
        if (
            params.max_tokens not in SUPPORTED_BEAM_WIDTH_SCHEDULE_STEPS
            or not isinstance(per_parent_beam_widths, tuple)
            or len(per_parent_beam_widths) != params.max_tokens
        ):
            _fail("beam_params.per_parent_beam_widths must contain one immutable value per step")
        for index, per_parent_beam_width in enumerate(per_parent_beam_widths):
            per_parent_beam_width = _positive_int(
                f"beam_params.per_parent_beam_widths[{index}]", per_parent_beam_width
            )
            if per_parent_beam_width > MAX_CANDIDATE_WIDTH:
                _fail(
                    f"beam_params.per_parent_beam_widths entries must be <= {MAX_CANDIDATE_WIDTH}"
                )
        error = per_parent_width_error(beam_width, widths, per_parent_beam_widths)
        if error is not None:
            _fail(error)
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
    if params.candidate_width is not None:
        candidate_width = _positive_int("beam_params.candidate_width", params.candidate_width)
        if candidate_width > MAX_CANDIDATE_WIDTH:
            _fail(
                "beam_params.candidate_width must be <= "
                f"{MAX_CANDIDATE_WIDTH}, got {candidate_width}"
            )
        # K is forwarded to every stage (``BeamBatchExecutionView.candidate_top_k``
        # returns it), so a dynamic schedule -- uniform or not -- samples K
        # candidates per parent and keeps that stage's ``output_width``.  The
        # only unsupported combination is a stage that retains more beams than
        # its parents supply candidates; reject it at admission rather than at
        # the first stage launch.  A per-stage K schedule overrides the scalar
        # and is bounded by its own ``input_width * K >= output_width`` check.
        scalar_widths = () if per_parent_beam_widths is not None else (widths or ())
        for index, width in enumerate(scalar_widths):
            if width > candidate_width:
                _fail(
                    f"beam_params.candidate_width={candidate_width} is below "
                    f"dynamic_beam_widths[{index}]={width}: each parent must supply "
                    "at least as many candidates as the stage retains"
                )
    seen_stop_ids: set[int] = set()
    for index, token_id in enumerate(params.stop_token_ids):
        normalized = _non_negative_int(f"beam_params.stop_token_ids[{index}]", token_id)
        if normalized in seen_stop_ids:
            _fail("beam_params.stop_token_ids must be unique")
        seen_stop_ids.add(normalized)
    if params.constraint_resource_id != DEFAULT_CONSTRAINT_RESOURCE_ID:
        _fail(
            f"Beam batch protocol version {BEAM_BATCH_PROTOCOL_VERSION} only admits "
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
            "output_options.num_return_sequences must be <= the final retained beam width "
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
        beam_width=final_retained_width(
            request.beam_params.beam_width, request.beam_params.dynamic_beam_widths
        ),
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
        if type(item.trie_id) is not int or item.trie_id < 0:
            _fail("item.trie_id must be a non-negative integer")
        request_id = _non_empty_string(
            f"items[{expected_index}].request.request_id",
            item.request.request_id,
        )
        if request_id in request_ids:
            _fail("item request_id values must be unique within a batch")
        request_ids.add(request_id)
        # Direct GR submissions can bypass frontend token validation. Enforce
        # the byte hasher's storage contract and reject negative embedding IDs
        # before enqueue/preprocessing. Embedding-only requests are rejected
        # later with a dedicated message.
        tokens = item.request.prompt_token_ids
        if tokens is not None and (
            not isinstance(tokens, np.ndarray)
            or tokens.dtype != np.int64
            or tokens.ndim != 1
            or not tokens.flags.c_contiguous
            or tokens.size == 0
        ):
            _fail("GR prompt_token_ids must be a nonempty contiguous int64 array")
        if tokens is not None and tokens.min() < 0:
            _fail("GR prompt_token_ids must contain only non-negative token IDs")
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


def sequence_columns(
    sequences: BeamSequenceColumns,
) -> tuple[NDArray[np.int32], NDArray[np.float32] | None, NDArray[np.int32] | None]:
    shape = (sequences.count, sequences.steps)
    tokens = np.frombuffer(sequences.token_ids, dtype="<i4").reshape(shape)
    scores = (
        None
        if sequences.cumulative_logprobs is None
        else np.frombuffer(sequences.cumulative_logprobs, dtype="<f4").reshape(sequences.count)
    )
    parents = (
        None
        if sequences.parent_beam_indices is None
        else np.frombuffer(sequences.parent_beam_indices, dtype="<i4").reshape(shape)
    )
    return tokens, scores, parents


def validate_complete_sequences(
    sequences: BeamSequenceColumns, request: BeamBatchRequest | None = None
) -> None:
    """Validate both the owned column shapes and request-dependent semantics."""
    if type(sequences.count) is not int or type(sequences.steps) is not int:
        raise ValueError("Beam column dimensions must be integers")
    if sequences.count < 1 or sequences.steps < 1:
        raise ValueError("Beam column dimensions must be positive")
    for value in (
        sequences.token_ids,
        sequences.cumulative_logprobs,
        sequences.parent_beam_indices,
    ):
        if value is not None and type(value) is not bytes:
            raise ValueError("Beam columns must be immutable bytes")
    tokens, scores, parents = sequence_columns(sequences)
    if np.any(tokens < 0) or (scores is not None and not np.isfinite(scores).all()):
        raise ValueError("Beam columns contain negative tokens or non-finite scores")
    if parents is not None and np.any(parents < 0):
        raise ValueError("Beam columns contain negative parent indices")
    if request is None:
        return
    options, params = request.output_options, request.beam_params
    if sequences.count > options.num_return_sequences or sequences.steps != params.max_tokens:
        raise ValueError("terminal Beam columns violate the result count or effective budget")
    if (scores is not None) != options.include_cumulative_logprob:
        raise ValueError("terminal Beam score presence differs from requested options")
    if (parents is not None) != options.include_parent_beam_indices:
        raise ValueError("terminal Beam parent presence differs from requested options")
    if parents is not None:
        widths = params.dynamic_beam_widths
        bounds = params.beam_width if widths is None else np.asarray((1, *widths[:-1]))
        if np.any(parents >= bounds):
            raise ValueError("terminal Beam result has invalid parent indices")


def validate_terminal_beam_batch_result(request: BeamBatchRequest, result: BeamBatchResult) -> None:
    """Validate request-dependent invariants after typed terminal decoding."""

    # Typed conversion checks every nested field on both transports,
    # including nonnegative indices and finite cumulative scores.
    # Check the V1 envelope and request-dependent semantics once here.
    if result.protocol_version != request.protocol_version:
        raise ValueError("terminal Beam result protocol version changed")
    if (result.output is None) == (result.failure is None):
        raise ValueError("terminal Beam result requires exactly one output or failure")
    if result.batch_id != request.batch_id:
        raise ValueError("terminal Beam result belongs to a different session")
    if (
        result.failure is not None
        and result.failure.item_index is not None
        and not 0 <= result.failure.item_index < len(request.items)
    ):
        raise ValueError("terminal Beam failure has an invalid item identity")
    if result.failure is not None and (
        not result.failure.code.strip() or not result.failure.message.strip()
    ):
        raise ValueError("terminal Beam failure requires a code and message")
    if result.output is not None:
        if len(result.output.items) != len(request.items):
            raise ValueError("terminal Beam result item count changed")
        options = request.output_options
        budget = request.beam_params.max_tokens
        width = request.beam_params.beam_width
        stage_widths = request.beam_params.dynamic_beam_widths
        parent_widths = None if stage_widths is None else (1, *stage_widths[:-1])
        include_scores = options.include_cumulative_logprob
        include_parents = options.include_parent_beam_indices
        expected_sequence = (
            "length",
            budget,
            None,
            include_scores,
            budget if include_parents else 0,
        )
        for item, source in zip(result.output.items, request.items):
            if item.request_id != source.request.request_id or item.item_index != source.item_index:
                raise ValueError("terminal Beam result item identity changed")
            if isinstance(item.sequences, BeamSequenceColumns):
                validate_complete_sequences(item.sequences, request)
                continue
            if not item.sequences or len(item.sequences) > options.num_return_sequences:
                raise ValueError("terminal Beam result has an invalid sequence count")
            for sequence in item.sequences:
                parents = sequence.parent_beam_indices
                # Compare the request-dependent shape together. Typed
                # conversion above has already checked every field's type.
                if (
                    sequence.finish_reason,
                    len(sequence.token_ids),
                    sequence.stop_token_id,
                    sequence.cumulative_logprob is not None,
                    len(parents),
                ) != expected_sequence:
                    # Keep precise protocol failures on malformed results.
                    if (
                        sequence.finish_reason != "length"
                        or len(sequence.token_ids) != budget
                        or sequence.stop_token_id is not None
                    ):
                        raise ValueError("terminal Beam result violates the effective budget")
                    if include_scores != (sequence.cumulative_logprob is not None):
                        raise ValueError(
                            "terminal Beam score presence differs from requested options"
                        )
                    if not include_parents and parents:
                        raise ValueError("terminal Beam result contains unrequested parent indices")
                    raise ValueError("terminal Beam result has invalid parent indices")
                if include_parents:
                    if parent_widths is None:
                        invalid_parent = max(parents) >= width
                    else:
                        invalid_parent = any(
                            parent >= bound for parent, bound in zip(parents, parent_widths)
                        )
                    if invalid_parent:
                        raise ValueError("terminal Beam result has invalid parent indices")


def materialize_beam_result(result: BeamBatchResult) -> BeamBatchResult:
    """Keep the public tuple-of-sequences API for HTTP and offline consumers."""
    if result.output is None or not any(
        isinstance(item.sequences, BeamSequenceColumns) for item in result.output.items
    ):
        return result
    items = []
    for item in result.output.items:
        sequences = item.sequences
        if isinstance(sequences, BeamSequenceColumns):
            tokens, scores, parents = sequence_columns(sequences)
            sequences = tuple(
                BeamOutputSequence(tuple(t), score, tuple(p), "length", None)
                for t, score, p in zip(
                    tokens.tolist(),
                    repeat(None) if scores is None else scores.tolist(),
                    repeat(()) if parents is None else parents.tolist(),
                )
            )
        items.append(BeamRequestOutput(item.item_index, item.request_id, sequences))
    return BeamBatchResult(
        result.protocol_version, result.batch_id, BeamBatchOutput(tuple(items)), result.failure
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
) -> None:
    """Validate the backend-neutral Output/Failure envelope and value shapes."""

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
            if isinstance(item.sequences, BeamSequenceColumns):
                try:
                    validate_complete_sequences(item.sequences)
                except (TypeError, ValueError) as exc:
                    _fail(str(exc))
                continue
            for sequence_index, sequence in enumerate(item.sequences):
                _validate_output_sequence(
                    sequence,
                    name=f"output.items[{expected_index}].sequences[{sequence_index}]",
                )


__all__ = [
    "ADMISSION_REJECTED_CODE",
    "BEAM_BATCH_PROTOCOL_VERSION",
    "DEFAULT_CONSTRAINT_RESOURCE_ID",
    "SUPPORTED_BEAM_BATCH_SIZES",
    "BeamBatchFailure",
    "BeamBatchItem",
    "BeamBatchLimits",
    "BeamBatchOutput",
    "BeamBatchParams",
    "BeamBatchRequest",
    "BeamBatchResult",
    "BeamSequenceColumns",
    "BeamContractError",
    "BeamEngineCoreRequest",
    "BeamOutputOptions",
    "BeamOutputSequence",
    "BeamRequestOutput",
    "GRAdmissionError",
    "materialize_beam_result",
    "sequence_columns",
    "to_int64_tokens",
    "validate_complete_sequences",
    "validate_beam_batch_limits",
    "validate_beam_batch_request",
    "validate_beam_batch_request_limits",
    "validate_beam_batch_result",
    "validate_terminal_beam_batch_result",
]
