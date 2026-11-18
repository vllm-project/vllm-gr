# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU tests: session-record split of GR dispatch validation."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable

import msgspec
import pytest

from tests.helpers.beam_requests import make_batch_request
from vllm_gr.v1.beam import dispatch as beam_dispatch
from vllm_gr.v1.beam.dispatch import (
    GR_STAGE_OUTPUT_ATTR,
    SESSION_FIELDS,
    STAGE_FIELDS,
    GRDeviceStateRef,
    GRDispatchSession,
    GRExecutionCapacity,
    GRStageKind,
    GRStageMetadata,
    validate_dispatch,
    validate_dispatch_session,
    validate_dispatch_step,
)
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def metadata(
    stage: int = 0,
    *,
    n: int = 3,
    request_ids: tuple[str, ...] = ("session-0",),
    prompt_counts: tuple[int, ...] = (7,),
) -> GRStageMetadata:
    request = make_batch_request(n)
    session_id = request.batch_id
    return GRStageMetadata(
        session_id=session_id,
        dispatch_id=stage,
        stage_index=stage,
        kind=GRStageKind.PREFILL if stage == 0 else GRStageKind.DECODE,
        decode_step=stage - 1 if stage else None,
        is_last_stage=stage == n - 1,
        input_device_state=GRDeviceStateRef(session_id, stage - 1) if stage else None,
        output_device_state=GRDeviceStateRef(session_id, stage),
        beam_params=msgspec.structs.replace(request.beam_params, stop_token_ids=()),
        output_options=request.output_options,
        num_prompt_tokens=sum(prompt_counts),
        produces_output=True,
        request_ids=request_ids,
        num_prompt_tokens_by_request=prompt_counts,
    )


def dispatch(m: GRStageMetadata, count: int | None = None) -> SimpleNamespace:
    scheduled = {m.session_id: count if count is not None else (7 if m.stage_index == 0 else 1)}
    result = SimpleNamespace(
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
    )
    setattr(result, GR_STAGE_OUTPUT_ATTR, {m.session_id: m})
    return result


def test_session_record_matches_the_unrecorded_validation_path() -> None:
    m = metadata()
    output = dispatch(m)
    session = validate_dispatch_session(output)
    assert isinstance(session, GRDispatchSession)
    assert session.session_id == m.session_id
    assert validate_dispatch(output) == (m,)
    assert validate_dispatch(output, session=session) == (m,)
    assert validate_dispatch_step(output, session) == (m,)


def test_decode_stage_validates_against_the_stage_zero_record() -> None:
    first = metadata()
    session = validate_dispatch_session(dispatch(first))
    decode = metadata(1)
    output = dispatch(decode)
    assert validate_dispatch(output, previous=first, session=session) == (decode,)
    assert validate_dispatch(output, previous=first) == (decode,)


def test_step_rejects_a_record_from_another_logical_batch() -> None:
    session = validate_dispatch_session(dispatch(metadata()))
    other = metadata(request_ids=("session-0:1",))
    with pytest.raises(ValueError, match="logical Beam batch identity changed across stages"):
        validate_dispatch_step(dispatch(other), session)


def test_start_of_session_is_still_enforced_with_a_record() -> None:
    session = validate_dispatch_session(dispatch(metadata()))
    with pytest.raises(ValueError, match="dispatch 0 Prefill"):
        validate_dispatch_step(dispatch(metadata(1)), session)


def test_step_does_not_rebuild_the_execution_view(monkeypatch: pytest.MonkeyPatch) -> None:
    first = metadata()
    session = validate_dispatch_session(dispatch(first))

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("step validation rebuilt the execution view")

    monkeypatch.setattr(BeamBatchExecutionView, "from_stage_metadata", classmethod(explode))
    decode = metadata(1)
    assert validate_dispatch_step(dispatch(decode), session, previous=first) == (decode,)


def test_step_still_checks_native_token_extent() -> None:
    first = metadata()
    session = validate_dispatch_session(dispatch(first))
    decode = metadata(1)
    output = dispatch(decode)
    output.scheduled_cached_reqs = SimpleNamespace(req_ids=[], resumed_req_ids=set())
    with pytest.raises(ValueError, match="no matching native request"):
        validate_dispatch_step(output, session, previous=first)


def test_metadata_fields_partition_between_session_and_stage() -> None:
    """Adding a field to ``GRStageMetadata`` has to pick a side."""
    declared = {field.name for field in dataclasses.fields(GRStageMetadata)}
    session = list(SESSION_FIELDS)
    stage = list(STAGE_FIELDS)
    assert len(session) == len(set(session))
    assert len(stage) == len(set(stage))
    assert set(session).isdisjoint(stage)
    assert set(session) | set(stage) == declared


def _with_extra_scheduled(output: SimpleNamespace) -> SimpleNamespace:
    output.num_scheduled_tokens["session-0:1"] = 1
    # Keep the total consistent, or the total check answers first and the
    # logical-batch check this case is about never runs.
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    return output


def _without_scheduled(output: SimpleNamespace) -> SimpleNamespace:
    output.num_scheduled_tokens = {}
    output.total_num_scheduled_tokens = 0
    return output


def _with_foreign_map_key(output: SimpleNamespace) -> SimpleNamespace:
    (_, m) = next(iter(getattr(output, GR_STAGE_OUTPUT_ATTR).items()))
    setattr(output, GR_STAGE_OUTPUT_ATTR, {"session-0:1": m})
    return output


def _rejection(call: Callable[[], Any]) -> str | None:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the assertion is accept/reject, not the class
        return str(exc)
    return None


def _replace_params(metadata: GRStageMetadata, **changes: Any) -> Any:
    return msgspec.structs.replace(metadata.beam_params, **changes)


# Each case tampers with the valid (Prefill, Decode) pair in a way the base
# branch rejects.  The point of the split is to compare the session fields, not
# to stop checking them, so the recorded path has to reject these too.
MUTATIONS: dict[
    str,
    Callable[[GRStageMetadata, GRStageMetadata], tuple[GRStageMetadata | None, SimpleNamespace]],
] = {
    "decode_step_is_not_an_int": lambda first, decode: (
        first,
        dispatch(replace(decode, decode_step=False)),
    ),
    "is_last_stage_is_not_a_bool": lambda first, decode: (
        first,
        dispatch(replace(decode, is_last_stage=0)),
    ),
    "produces_output_is_not_a_bool": lambda first, decode: (
        first,
        dispatch(replace(decode, produces_output=1)),
    ),
    "session_init_on_decode": lambda first, decode: (
        first,
        dispatch(replace(decode, session_init=make_batch_request(3))),
    ),
    "execution_capacity_changed": lambda first, decode: (
        first,
        dispatch(replace(decode, execution_capacity=GRExecutionCapacity(8))),
    ),
    "request_ids_is_not_a_tuple": lambda first, decode: (
        first,
        dispatch(replace(decode, request_ids=["session-0"])),
    ),
    "prompt_counts_is_not_a_tuple": lambda first, decode: (
        first,
        dispatch(replace(decode, num_prompt_tokens_by_request=[1])),
    ),
    "extra_scheduled_request": lambda first, decode: (
        first,
        _with_extra_scheduled(dispatch(decode)),
    ),
    "missing_scheduled_request": lambda first, decode: (
        first,
        _without_scheduled(dispatch(decode)),
    ),
    "map_key_is_not_the_session_id": lambda first, decode: (
        first,
        _with_foreign_map_key(dispatch(decode)),
    ),
    "previous_identity_changed": lambda first, decode: (
        replace(first, request_ids=("session-0:1",)),
        dispatch(decode),
    ),
    # Equal-valued but differently typed: ``msgspec`` equality compares values,
    # so these reach the cheap structural checks instead of the identity check.
    "max_tokens_is_not_an_int": lambda first, decode: (
        first,
        dispatch(replace(decode, beam_params=_replace_params(decode, max_tokens=3.0))),
    ),
    "beam_width_is_not_an_int": lambda first, decode: (
        first,
        dispatch(replace(decode, beam_params=_replace_params(decode, beam_width=4.0))),
    ),
    "ignore_stop_tokens_is_not_a_bool": lambda first, decode: (
        first,
        dispatch(replace(decode, beam_params=_replace_params(decode, ignore_stop_tokens=1))),
    ),
    "prompt_count_is_not_an_int": lambda first, decode: (
        first,
        dispatch(replace(decode, num_prompt_tokens_by_request=(7.0,))),
    ),
}


# The cases that must be answered by the logical-batch check itself: if another
# check happens to catch them, the regression this test exists for is masked.
EXPECTED = {
    "extra_scheduled_request": "immutable logical batch",
    "missing_scheduled_request": "immutable logical batch",
}


@pytest.mark.parametrize("case", sorted(MUTATIONS))  # type: ignore[misc]
def test_recorded_path_rejects_what_the_unrecorded_path_rejects(case: str) -> None:
    first, decode = metadata(), metadata(1)
    session = validate_dispatch_session(dispatch(first))
    previous, output = MUTATIONS[case](first, decode)
    unrecorded = _rejection(lambda: validate_dispatch(output, previous=previous))
    recorded = _rejection(lambda: validate_dispatch(output, previous=previous, session=session))
    assert unrecorded is not None, f"{case} was accepted without the record"
    assert recorded is not None, f"{case} was accepted with the record"
    if case in EXPECTED:
        assert EXPECTED[case] in unrecorded, f"{case} was rejected elsewhere: {unrecorded}"
        assert EXPECTED[case] in recorded, f"{case} was rejected elsewhere: {recorded}"


@pytest.mark.parametrize(  # type: ignore[misc]
    ("session_value", "decode_value", "message"),
    [
        ({"candidate_width": 4}, {"candidate_width": 4.0}, "candidate_width"),
        (
            {"dynamic_beam_widths": (4, 4, 4)},
            {"dynamic_beam_widths": (4.0, 4.0, 4.0)},
            "dynamic_beam_widths",
        ),
    ],
)
def test_cached_path_rejects_equal_valued_wrong_typed_schedules(
    session_value: dict[str, Any], decode_value: dict[str, Any], message: str
) -> None:
    valid = metadata()
    first = replace(valid, beam_params=_replace_params(valid, **session_value))
    session = validate_dispatch_session(dispatch(first))
    decode = replace(metadata(1), beam_params=_replace_params(metadata(1), **decode_value))
    with pytest.raises(ValueError, match=message):
        validate_dispatch_step(dispatch(decode), session, previous=first)


def test_record_is_a_value_snapshot_of_frozen_members() -> None:
    """The record is only sound because its members cannot be mutated in place."""

    m = metadata()
    session = validate_dispatch_session(dispatch(m))
    assert session.fields == tuple(getattr(m, name) for name in SESSION_FIELDS)
    assert isinstance(session.capacity, GRExecutionCapacity)

    # ``FrozenInstanceError`` subclasses ``AttributeError``; ``msgspec`` raises
    # ``AttributeError: immutable type`` for its frozen structs.
    with pytest.raises(AttributeError):
        m.num_prompt_tokens = 9
    with pytest.raises(AttributeError):
        m.beam_params.beam_width = 8
    with pytest.raises(AttributeError):
        m.output_options.num_return_sequences = 4
    with pytest.raises(AttributeError):
        session.capacity.beam_width = 8


def test_step_never_rebuilds_the_session_record(monkeypatch: pytest.MonkeyPatch) -> None:
    first = metadata()
    session = validate_dispatch_session(dispatch(first))

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("step validation rebuilt the session record")

    monkeypatch.setattr(beam_dispatch, "validate_dispatch_session", explode)
    decode = metadata(1)
    assert validate_dispatch_step(dispatch(decode), session, previous=first) == (decode,)
