# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the full-model lifecycle probe assertions."""

from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy

import pytest

from tests.profiling.gr_lifecycle_probe import (
    _assert_abort_window,
    _assert_graph_delta,
    _assert_released,
)


def _abort_case(future: Future | None = None):
    if future is None:
        future = Future()
    queued = {
        "future_done": future.done(),
        "request_ids": ["gr-internal-0"],
    }
    return {
        "injected": {
            "request_id": "gr-internal-0",
            "issued": 2,
            "completed": 1,
            "device_step": 2,
        },
        "before_abort": {"batch_queue": [dict(queued)]},
        "after_abort": {
            "batch_queue": [dict(queued)],
            "control": {"control_slots_occupied": 1},
        },
    }


def _graph(*, enabled: bool, widths=(128,), dispatches: int = 0):
    return {
        "beam_graph_enabled": enabled,
        "captured_widths": list(widths),
        "full_dispatches": dispatches,
    }


def _released_snapshot():
    return {
        "state": {"created": 3, "freed": 3, "active": 0},
        "scheduler_requests": 0,
        "frontend_requests": 0,
        "beam_sessions": 0,
        "beam_slots_active": 0,
        "beam_slots_free": 4,
        "native_blocks_free": 96,
        "batch_queue": [],
        "control": {"control_slots_occupied": 0},
    }


def test_abort_window_accepts_one_issued_pending_frame() -> None:
    _assert_abort_window(_abort_case(), issued_target=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("issued", 1),
        ("device_step", 1),
        ("completed", 0),
        ("completed", 2),
    ],
)
def test_abort_window_rejects_invalid_issue_completion_cursor(field, value) -> None:
    case = _abort_case()
    case["injected"][field] = value

    with pytest.raises(AssertionError):
        _assert_abort_window(case, issued_target=2)


@pytest.mark.parametrize("outcome", ["completed", "errored"])
@pytest.mark.parametrize("label", ["before_abort", "after_abort"])
def test_abort_window_rejects_a_done_or_errored_queued_future(
    outcome, label
) -> None:
    future = Future()
    if outcome == "completed":
        future.set_result(object())
    else:
        future.set_exception(RuntimeError("worker response failed"))
    case = _abort_case()
    case[label]["batch_queue"][0]["future_done"] = future.done()

    with pytest.raises(AssertionError):
        _assert_abort_window(case, issued_target=2)


def test_abort_window_rejects_control_ack_before_terminal_queue_drain() -> None:
    case = _abort_case()
    case["after_abort"]["control"]["control_slots_occupied"] = 0

    with pytest.raises(AssertionError):
        _assert_abort_window(case, issued_target=2)


def test_graph_delta_accepts_capture_and_exact_real_dispatch_count() -> None:
    before = _graph(enabled=True, dispatches=7)
    after = _graph(enabled=True, dispatches=9)

    _assert_graph_delta(
        before,
        after,
        execution="graph",
        width=128,
        decode_dispatches=2,
    )


def test_graph_delta_rejects_enabled_configuration_without_capture() -> None:
    before = _graph(enabled=True, widths=(), dispatches=7)
    after = _graph(enabled=True, widths=(), dispatches=9)

    with pytest.raises(AssertionError):
        _assert_graph_delta(
            before,
            after,
            execution="graph",
            width=128,
            decode_dispatches=2,
        )


def test_graph_delta_rejects_capture_without_real_graph_dispatch() -> None:
    before = _graph(enabled=True, dispatches=7)
    after = _graph(enabled=True, dispatches=7)

    with pytest.raises(AssertionError):
        _assert_graph_delta(
            before,
            after,
            execution="graph",
            width=128,
            decode_dispatches=2,
        )


@pytest.mark.parametrize("actual_delta", [1, 3])
def test_graph_delta_rejects_fewer_or_extra_dispatches(actual_delta) -> None:
    before = _graph(enabled=True, dispatches=7)
    after = _graph(enabled=True, dispatches=7 + actual_delta)

    with pytest.raises(AssertionError):
        _assert_graph_delta(
            before,
            after,
            execution="graph",
            width=128,
            decode_dispatches=2,
        )


def test_graph_delta_accepts_actual_eager_without_capture_or_dispatch() -> None:
    before = _graph(enabled=False, widths=(), dispatches=7)
    after = _graph(enabled=False, widths=(), dispatches=7)

    _assert_graph_delta(
        before,
        after,
        execution="eager",
        width=128,
        decode_dispatches=2,
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (_graph(enabled=True), _graph(enabled=True)),
        (_graph(enabled=False), _graph(enabled=True)),
    ],
)
def test_graph_delta_rejects_eager_with_graph_enabled(before, after) -> None:
    with pytest.raises(AssertionError):
        _assert_graph_delta(
            before,
            after,
            execution="eager",
            width=128,
            decode_dispatches=2,
        )


def test_released_accepts_drained_zero_control_and_undrained_owned_control() -> None:
    reference = _released_snapshot()
    drained = deepcopy(reference)
    _assert_released(drained, reference, drained=True)

    undrained = deepcopy(reference)
    undrained["batch_queue"] = [{"future_done": False}]
    undrained["control"]["control_slots_occupied"] = 1
    _assert_released(undrained, reference, drained=False)


@pytest.mark.parametrize("free_key", ["native_blocks_free", "beam_slots_free"])
def test_released_rejects_free_block_or_slot_leak(free_key) -> None:
    reference = _released_snapshot()
    leaked = deepcopy(reference)
    leaked[free_key] -= 1

    with pytest.raises(AssertionError):
        _assert_released(leaked, reference, drained=True)


def test_released_rejects_occupied_control_after_drain() -> None:
    reference = _released_snapshot()
    leaked = deepcopy(reference)
    leaked["control"]["control_slots_occupied"] = 1

    with pytest.raises(AssertionError):
        _assert_released(leaked, reference, drained=True)
