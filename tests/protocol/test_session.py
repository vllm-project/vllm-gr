# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm_gr.protocol.errors import (
    GRErrorCode,
    GRInvalidSequenceError,
    GRInvalidSessionStateError,
    GRProtocolValidationError,
    GRSessionAlreadyExistsError,
    GRSessionNotFoundError,
)
from vllm_gr.v1.engine.session import (
    BeamSessionManager,
    BeamSessionState,
)


def test_full_session_lifecycle() -> None:
    manager = BeamSessionManager[tuple[int, ...]]()

    pending = manager.begin("session-1")
    cached = manager.cache_prefill("session-1", (1, 2, 3))
    first_step = manager.accept_step("session-1", sequence_no=0)
    second_step = manager.accept_step("session-1", sequence_no=1)
    closed = manager.end("session-1", sequence_no=2)

    assert pending.state is BeamSessionState.PENDING
    assert cached.state is BeamSessionState.PREFILL_CACHED
    assert cached.cached_prefill == (1, 2, 3)
    assert first_step.state is BeamSessionState.ACTIVE
    assert first_step.next_sequence_no == 1
    assert second_step.next_sequence_no == 2
    assert closed is not None
    assert closed.state is BeamSessionState.CLOSED
    assert manager.get("session-1") is None
    assert len(manager) == 0


def test_legacy_begin_from_prefill_is_atomic() -> None:
    manager = BeamSessionManager[dict[str, int]]()

    session = manager.begin_from_prefill("session-1", {"prefix_len": 4})

    assert session.state is BeamSessionState.PREFILL_CACHED
    assert session.cached_prefill == {"prefix_len": 4}
    assert manager.get("session-1") == session


def test_duplicate_begin_is_rejected() -> None:
    manager = BeamSessionManager[object]()
    manager.begin("session-1")

    with pytest.raises(GRSessionAlreadyExistsError) as exc_info:
        manager.begin("session-1")

    assert exc_info.value.code is GRErrorCode.SESSION_ALREADY_EXISTS


def test_step_requires_cached_prefill() -> None:
    manager = BeamSessionManager[object]()
    manager.begin("session-1")

    with pytest.raises(GRInvalidSessionStateError) as exc_info:
        manager.accept_step("session-1", 0)

    assert exc_info.value.code is GRErrorCode.INVALID_SESSION_STATE


def test_step_for_unknown_session_is_rejected() -> None:
    manager = BeamSessionManager[object]()

    with pytest.raises(GRSessionNotFoundError) as exc_info:
        manager.accept_step("missing", 0)

    assert exc_info.value.code is GRErrorCode.SESSION_NOT_FOUND


@pytest.mark.parametrize("sequence_no", [1, 3])  # type: ignore[misc]
def test_out_of_order_step_is_rejected(sequence_no: int) -> None:
    manager = BeamSessionManager[object]()
    manager.begin_from_prefill("session-1", object())

    with pytest.raises(GRInvalidSequenceError) as exc_info:
        manager.accept_step("session-1", sequence_no)

    assert exc_info.value.code is GRErrorCode.INVALID_SEQUENCE
    assert exc_info.value.expected == 0


def test_repeated_step_sequence_is_rejected() -> None:
    manager = BeamSessionManager[object]()
    manager.begin_from_prefill("session-1", object())
    manager.accept_step("session-1", 0)

    with pytest.raises(GRInvalidSequenceError) as exc_info:
        manager.accept_step("session-1", 0)

    assert exc_info.value.expected == 1


def test_end_is_idempotent_after_successful_cleanup() -> None:
    manager = BeamSessionManager[object]()
    manager.begin_from_prefill("session-1", object())

    assert manager.end("session-1", 0) is not None
    assert manager.end("session-1", 0) is None


def test_end_validates_sequence_while_session_exists() -> None:
    manager = BeamSessionManager[object]()
    manager.begin_from_prefill("session-1", object())
    manager.accept_step("session-1", 0)

    with pytest.raises(GRInvalidSequenceError):
        manager.end("session-1", 0)

    assert manager.get("session-1") is not None


def test_discard_bypasses_sequence_after_local_failure() -> None:
    manager = BeamSessionManager[object]()
    manager.begin_from_prefill("session-1", object())
    manager.accept_step("session-1", 0)

    closed = manager.discard("session-1")

    assert closed is not None
    assert closed.state is BeamSessionState.CLOSED
    assert manager.discard("session-1") is None


@pytest.mark.parametrize(  # type: ignore[misc]
    ("session_id", "sequence_no"),
    [("", 0), ("session-1", -1), ("session-1", True)],
)
def test_invalid_session_inputs_are_rejected(
    session_id: str,
    sequence_no: int,
) -> None:
    manager = BeamSessionManager[object]()
    if session_id:
        manager.begin_from_prefill(session_id, object())

    with pytest.raises(GRProtocolValidationError):
        manager.end(session_id, sequence_no)


def test_concurrent_sessions_are_isolated() -> None:
    manager = BeamSessionManager[int]()

    def run_session(index: int) -> None:
        session_id = f"session-{index}"
        manager.begin_from_prefill(session_id, index)
        manager.accept_step(session_id, 0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(run_session, range(32)))

    snapshot = manager.snapshot()
    assert len(snapshot) == 32
    assert {session.session_id for session in snapshot} == {
        f"session-{index}" for index in range(32)
    }
    assert all(session.state is BeamSessionState.ACTIVE for session in snapshot)


def test_concurrent_duplicate_begin_has_one_winner() -> None:
    manager = BeamSessionManager[int]()

    def try_begin(index: int) -> bool:
        try:
            manager.begin_from_prefill("shared-session", index)
        except GRSessionAlreadyExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(try_begin, range(32)))

    assert results.count(True) == 1
    assert results.count(False) == 31
    assert len(manager) == 1
