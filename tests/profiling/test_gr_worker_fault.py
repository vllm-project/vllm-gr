# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the full-model one-shot Worker fault helper."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.profiling.gr_worker_fault import (
    InjectedWorkerError,
    _target,
    injected_failure_delivery,
    inject_first_decode_failure,
)


class _Backend:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.result = SimpleNamespace(device_step=2, control_output=object())

    def advance_beam(
        self,
        current_runner,
        state,
        sampler_output,
        metadata,
        *,
        is_last_step,
    ):
        self.calls.append("advance_leaf")
        return self.result


class _Runner:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self._gr_model_runner_states = {}
        self._gr_device_backend = _Backend(calls)
        self._gr_cuda_output_transport = SimpleNamespace(stats=self._stats)
        self._current_scheduler_output = None
        self.prepare_result = object()

    def _stats(self):
        self.calls.append("control_stats")
        return {"control_slots_occupied": 1}

    def _prepare_inputs(self, scheduled, *args, **kwargs):
        self.calls.append("prepare_leaf")
        return self.prepare_result


def _fixture():
    calls: list[str] = []
    runner = _Runner(calls)
    request_id = "gr-native-0"
    metadata = {
        "persistent_request": True,
        "persistent_runner_state": True,
        "is_beam_decode": True,
        "generation_index": 1,
        "decode_step": 0,
    }
    state = SimpleNamespace(
        stage=SimpleNamespace(name="PREFILL_BOOTSTRAPPED"),
        step=1,
        device_step=1,
        _prepared_step=1,
        beam_width=128,
    )
    scheduled = SimpleNamespace(
        beam_data={request_id: metadata},
        gr_outputs={
            request_id: SimpleNamespace(
                logical_token_count=1,
                physical_row_count=128,
            )
        },
    )
    runner._gr_model_runner_states[request_id] = state
    runner._current_scheduler_output = scheduled
    return runner, scheduled, request_id, metadata, state, calls


def _invoke(runner, scheduled, metadata, state, fault):
    if fault == "prepare":
        return runner._prepare_inputs(scheduled, "arg", keyword="value")
    return runner._gr_device_backend.advance_beam(
        runner,
        state,
        object(),
        metadata,
        is_last_step=False,
    )


def test_target_accepts_only_exact_persistent_first_decode() -> None:
    runner, scheduled, request_id, metadata, state, _ = _fixture()

    assert _target(runner, scheduled) == (request_id, metadata, state)


@pytest.mark.parametrize(
    "metadata_update",
    [
        {"persistent_request": False},
        {"persistent_runner_state": False},
        {"is_beam_decode": False},
        {"generation_index": 0},
        {"generation_index": 2},
    ],
)
def test_target_ignores_ordinary_prefill_and_nonfirst_decode(metadata_update) -> None:
    runner, scheduled, _, _, _, _ = _fixture()
    scheduled.beam_data["gr-native-0"].update(metadata_update)

    assert _target(runner, scheduled) is None


def test_target_rejects_a_mixed_batch_when_target_is_present() -> None:
    runner, scheduled, _, _, _, _ = _fixture()
    scheduled.beam_data["ordinary"] = {}

    with pytest.raises(AssertionError):
        _target(runner, scheduled)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("metadata", "decode_step"), 1),
        (("state", "stage"), SimpleNamespace(name="DECODE_RUNNING")),
        (("state", "step"), 0),
        (("state", "device_step"), 0),
        (("state", "_prepared_step"), 0),
        (("sidecar", "logical_token_count"), 2),
        (("sidecar", "physical_row_count"), 64),
    ],
)
def test_target_rejects_metadata_state_and_geometry_mismatch(path, value) -> None:
    runner, scheduled, request_id, metadata, state, _ = _fixture()
    owner, field = path
    if owner == "metadata":
        metadata[field] = value
    elif owner == "state":
        setattr(state, field, value)
    else:
        setattr(scheduled.gr_outputs[request_id], field, value)

    with pytest.raises(AssertionError):
        _target(runner, scheduled)


@pytest.mark.parametrize("fault", ["prepare", "advance"])
def test_real_leaf_executes_before_the_one_shot_fault(fault) -> None:
    runner, scheduled, request_id, metadata, state, calls = _fixture()
    report = {}

    with inject_first_decode_failure(runner, report, fault) as failure:
        with pytest.raises(InjectedWorkerError) as caught:
            _invoke(runner, scheduled, metadata, state, fault)

    assert caught.value is failure
    leaf = "prepare_leaf" if fault == "prepare" else "advance_leaf"
    assert calls == [leaf, "control_stats"]
    assert report["worker_fault"] == {
        "kind": fault,
        "fired": 1,
        "request_id": request_id,
        "generation_index": 1,
        "decode_step": 0,
        "state_stage": "PREFILL_BOOTSTRAPPED",
        "state_step": 1,
        "device_step": 1,
        "prepared_step": 1,
        "control": {"control_slots_occupied": 1},
        **(
            {"backend_result_device_step": 2}
            if fault == "advance"
            else {}
        ),
    }


@pytest.mark.parametrize("fault", ["prepare", "advance"])
@pytest.mark.parametrize("stage", ["ordinary", "prefill"])
def test_ordinary_and_prefill_calls_return_real_leaf_without_firing(fault, stage) -> None:
    runner, scheduled, _, metadata, state, calls = _fixture()
    if stage == "ordinary":
        scheduled.beam_data = {}
        scheduled.gr_outputs = {}
    else:
        metadata["is_beam_decode"] = False
    report = {}

    with inject_first_decode_failure(runner, report, fault):
        result = _invoke(runner, scheduled, metadata, state, fault)

    expected = (
        runner.prepare_result
        if fault == "prepare"
        else runner._gr_device_backend.result
    )
    assert result is expected
    assert calls == ["prepare_leaf" if fault == "prepare" else "advance_leaf"]
    assert report["worker_fault"] == {"kind": fault, "fired": 0}


def test_advance_rejects_nonidentical_metadata_before_calling_leaf() -> None:
    runner, scheduled, _, metadata, state, calls = _fixture()
    copied_metadata = deepcopy(metadata)

    with inject_first_decode_failure(runner, {}, "advance"):
        with pytest.raises(AssertionError):
            _invoke(runner, scheduled, copied_metadata, state, "advance")

    assert calls == []


@pytest.mark.parametrize("fault", ["prepare", "advance"])
def test_fault_is_one_shot_even_if_caller_catches_it_inside_scope(fault) -> None:
    runner, scheduled, _, metadata, state, calls = _fixture()
    report = {}

    with inject_first_decode_failure(runner, report, fault):
        with pytest.raises(InjectedWorkerError):
            _invoke(runner, scheduled, metadata, state, fault)
        with pytest.raises(AssertionError):
            _invoke(runner, scheduled, metadata, state, fault)

    leaf = "prepare_leaf" if fault == "prepare" else "advance_leaf"
    assert calls.count(leaf) == 2
    assert calls.count("control_stats") == 1
    assert report["worker_fault"]["fired"] == 1


@pytest.mark.parametrize("fault", ["prepare", "advance"])
@pytest.mark.parametrize("had_override", [False, True])
def test_instance_override_is_restored_exactly(fault, had_override) -> None:
    runner, _, _, _, _, _ = _fixture()
    owner = runner if fault == "prepare" else runner._gr_device_backend
    name = "_prepare_inputs" if fault == "prepare" else "advance_beam"
    if had_override:
        override = mock.Mock(name=f"existing_{fault}_override")
        setattr(owner, name, override)
    else:
        override = None
        assert name not in vars(owner)

    with inject_first_decode_failure(runner, {}, fault):
        assert name in vars(owner)
        assert vars(owner)[name] is not override

    if had_override:
        assert vars(owner)[name] is override
    else:
        assert name not in vars(owner)


@pytest.mark.parametrize(
    ("device_step", "control_output"),
    [(1, object()), (2, None)],
)
def test_advance_rejects_invalid_backend_result(device_step, control_output) -> None:
    runner, scheduled, _, metadata, state, calls = _fixture()
    runner._gr_device_backend.result = SimpleNamespace(
        device_step=device_step,
        control_output=control_output,
    )
    report = {}

    with inject_first_decode_failure(runner, report, "advance"):
        with pytest.raises(AssertionError):
            _invoke(runner, scheduled, metadata, state, "advance")

    assert calls == ["advance_leaf", "control_stats"]
    assert report["worker_fault"]["fired"] == 1
    assert "backend_result_device_step" not in report["worker_fault"]


def test_unknown_fault_is_rejected_without_installing_an_override() -> None:
    runner, _, _, _, _, _ = _fixture()
    report = {}

    with pytest.raises(ValueError, match="unknown Worker fault"):
        with inject_first_decode_failure(runner, report, "unknown"):
            pytest.fail("unknown fault entered context")

    assert "_prepare_inputs" not in vars(runner)
    assert "advance_beam" not in vars(runner._gr_device_backend)
    assert report == {}


def _with_cause(error: BaseException, cause: BaseException) -> BaseException:
    try:
        raise error from cause
    except BaseException as delivered:
        return delivered


@pytest.mark.parametrize("fault", ["prepare", "advance"])
def test_failure_delivery_accepts_only_the_identical_injected_object(fault) -> None:
    expected = InjectedWorkerError(f"expected {fault}")

    assert injected_failure_delivery(expected, expected, fault) == "direct"


def test_failure_delivery_accepts_exact_advance_quarantine_direct_cause() -> None:
    expected = InjectedWorkerError("injected advance")
    delivered = _with_cause(
        RuntimeError("control pool quarantined; device drain is required"),
        expected,
    )

    assert delivered.__cause__ is expected
    assert injected_failure_delivery(delivered, expected, "advance") == (
        "quarantined_previous_control"
    )


@pytest.mark.parametrize("fault", ["prepare", "advance"])
def test_failure_delivery_rejects_different_object_with_same_message(fault) -> None:
    expected = InjectedWorkerError("identical text")
    delivered = InjectedWorkerError("identical text")

    with pytest.raises(AssertionError) as caught:
        injected_failure_delivery(delivered, expected, fault)

    assert caught.value.__cause__ is delivered


def test_failure_delivery_rejects_implicit_exception_context() -> None:
    expected = InjectedWorkerError("injected advance")
    delivered = None
    try:
        raise expected
    except InjectedWorkerError:
        try:
            raise RuntimeError("control pool quarantined; device drain is required")
        except RuntimeError as error:
            delivered = error
            assert error.__context__ is expected
            assert error.__cause__ is None

    assert delivered is not None
    with pytest.raises(AssertionError) as caught:
        injected_failure_delivery(delivered, expected, "advance")

    assert caught.value.__cause__ is delivered


def test_failure_delivery_rejects_nested_injected_cause() -> None:
    expected = InjectedWorkerError("injected advance")
    intermediate = _with_cause(RuntimeError("intermediate"), expected)
    delivered = _with_cause(
        RuntimeError("control pool quarantined; device drain is required"),
        intermediate,
    )

    assert delivered.__cause__ is intermediate
    assert intermediate.__cause__ is expected
    with pytest.raises(AssertionError) as caught:
        injected_failure_delivery(delivered, expected, "advance")
    assert caught.value.__cause__ is delivered


@pytest.mark.parametrize(
    "delivered",
    [
        RuntimeError("another control failure"),
        ValueError("control pool quarantined; device drain is required"),
    ],
)
def test_failure_delivery_rejects_other_message_or_exception_type(delivered) -> None:
    expected = InjectedWorkerError("injected advance")
    delivered = _with_cause(delivered, expected)

    with pytest.raises(AssertionError) as caught:
        injected_failure_delivery(delivered, expected, "advance")

    assert caught.value.__cause__ is delivered


def test_failure_delivery_rejects_quarantine_wrapper_for_prepare_fault() -> None:
    expected = InjectedWorkerError("injected prepare")
    delivered = _with_cause(
        RuntimeError("control pool quarantined; device drain is required"),
        expected,
    )

    with pytest.raises(AssertionError) as caught:
        injected_failure_delivery(delivered, expected, "prepare")

    assert caught.value.__cause__ is delivered
