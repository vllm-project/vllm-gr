# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread-safe Beam Session lifecycle independent of EngineCore transport."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import Generic, TypeVar

from vllm_gr.protocol.errors import (
    GRInvalidSequenceError,
    GRInvalidSessionStateError,
    GRProtocolValidationError,
    GRSessionAlreadyExistsError,
    GRSessionNotFoundError,
)

PrefillStateT = TypeVar("PrefillStateT")


class BeamSessionState(str, Enum):
    PENDING = "pending"
    PREFILL_CACHED = "prefill_cached"
    ACTIVE = "active"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class BeamSession(Generic[PrefillStateT]):
    session_id: str
    state: BeamSessionState
    cached_prefill: PrefillStateT | None = None
    next_sequence_no: int = 0


class BeamSessionManager(Generic[PrefillStateT]):
    """Own Beam Session state and enforce monotonic command ordering.

    Sequence numbers start at zero. Ending a missing session is deliberately
    idempotent, while executing a step for a missing session is an error.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, BeamSession[PrefillStateT]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_session_id(session_id: object) -> str:
        if not isinstance(session_id, str) or not session_id:
            raise GRProtocolValidationError("session_id must be a non-empty string")
        return session_id

    @staticmethod
    def _validate_sequence_no(sequence_no: object) -> int:
        if isinstance(sequence_no, bool) or not isinstance(sequence_no, int) or sequence_no < 0:
            raise GRProtocolValidationError(
                f"sequence_no must be a non-negative integer, got {sequence_no!r}"
            )
        return sequence_no

    def begin(self, session_id: str) -> BeamSession[PrefillStateT]:
        session_id = self._validate_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                raise GRSessionAlreadyExistsError(session_id)
            session: BeamSession[PrefillStateT] = BeamSession(
                session_id=session_id,
                state=BeamSessionState.PENDING,
            )
            self._sessions[session_id] = session
            return session

    def cache_prefill(
        self,
        session_id: str,
        cached_prefill: PrefillStateT,
    ) -> BeamSession[PrefillStateT]:
        session_id = self._validate_session_id(session_id)
        with self._lock:
            session = self._get_required(session_id)
            if session.state is not BeamSessionState.PENDING:
                raise GRInvalidSessionStateError(
                    session_id,
                    session.state.value,
                    "cache prefill for",
                )
            updated = replace(
                session,
                state=BeamSessionState.PREFILL_CACHED,
                cached_prefill=cached_prefill,
            )
            self._sessions[session_id] = updated
            return updated

    def begin_from_prefill(
        self,
        session_id: str,
        cached_prefill: PrefillStateT,
    ) -> BeamSession[PrefillStateT]:
        """Legacy ADD_BATCH convenience: begin and cache in one lock scope."""
        session_id = self._validate_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                raise GRSessionAlreadyExistsError(session_id)
            session = BeamSession(
                session_id=session_id,
                state=BeamSessionState.PREFILL_CACHED,
                cached_prefill=cached_prefill,
            )
            self._sessions[session_id] = session
            return session

    def accept_step(
        self,
        session_id: str,
        sequence_no: int,
    ) -> BeamSession[PrefillStateT]:
        session_id = self._validate_session_id(session_id)
        sequence_no = self._validate_sequence_no(sequence_no)
        with self._lock:
            session = self._get_required(session_id)
            if session.state not in (
                BeamSessionState.PREFILL_CACHED,
                BeamSessionState.ACTIVE,
            ):
                raise GRInvalidSessionStateError(
                    session_id,
                    session.state.value,
                    "execute a step for",
                )
            self._ensure_expected_sequence(session, sequence_no)
            updated = replace(
                session,
                state=BeamSessionState.ACTIVE,
                next_sequence_no=sequence_no + 1,
            )
            self._sessions[session_id] = updated
            return updated

    def end(
        self,
        session_id: str,
        sequence_no: int,
    ) -> BeamSession[PrefillStateT] | None:
        """Close and remove a session; repeated cleanup returns ``None``."""
        session_id = self._validate_session_id(session_id)
        sequence_no = self._validate_sequence_no(sequence_no)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            self._ensure_expected_sequence(session, sequence_no)
            del self._sessions[session_id]
            return replace(session, state=BeamSessionState.CLOSED)

    def discard(self, session_id: str) -> BeamSession[PrefillStateT] | None:
        """Remove state after an unrecoverable local failure without sequence checks."""
        session_id = self._validate_session_id(session_id)
        with self._lock:
            session = self._sessions.pop(session_id, None)
            return None if session is None else replace(session, state=BeamSessionState.CLOSED)

    def get(self, session_id: str) -> BeamSession[PrefillStateT] | None:
        session_id = self._validate_session_id(session_id)
        with self._lock:
            return self._sessions.get(session_id)

    def snapshot(self) -> tuple[BeamSession[PrefillStateT], ...]:
        with self._lock:
            return tuple(self._sessions.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _get_required(self, session_id: str) -> BeamSession[PrefillStateT]:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise GRSessionNotFoundError(session_id) from exc

    @staticmethod
    def _ensure_expected_sequence(
        session: BeamSession[PrefillStateT],
        sequence_no: int,
    ) -> None:
        if sequence_no != session.next_sequence_no:
            raise GRInvalidSequenceError(
                session.session_id,
                received=sequence_no,
                expected=session.next_sequence_no,
            )
