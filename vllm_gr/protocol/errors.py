# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured errors for the GR EngineCore protocol."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_gr.protocol.version import GRProtocolVersion


class GRErrorCode(str, Enum):
    DECODE_ERROR = "decode_error"
    INCOMPATIBLE_VERSION = "incompatible_version"
    INVALID_MESSAGE = "invalid_message"
    SESSION_ALREADY_EXISTS = "session_already_exists"
    SESSION_NOT_FOUND = "session_not_found"
    INVALID_SESSION_STATE = "invalid_session_state"
    INVALID_SEQUENCE = "invalid_sequence"


class GRProtocolError(Exception):
    """Base exception with a stable machine-readable error code."""

    code: GRErrorCode

    def __init__(self, code: GRErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


class GRProtocolDecodeError(GRProtocolError):
    def __init__(self, message: str) -> None:
        super().__init__(GRErrorCode.DECODE_ERROR, message)


class GRProtocolValidationError(GRProtocolError):
    def __init__(self, message: str) -> None:
        super().__init__(GRErrorCode.INVALID_MESSAGE, message)


class GRIncompatibleVersionError(GRProtocolError):
    def __init__(
        self,
        *,
        received: GRProtocolVersion,
        supported: GRProtocolVersion,
    ) -> None:
        self.received = received
        self.supported = supported
        super().__init__(
            GRErrorCode.INCOMPATIBLE_VERSION,
            "incompatible GR protocol version: "
            f"received={received.major}.{received.minor}, "
            f"supported={supported.major}.{supported.minor}",
        )


class GRSessionError(GRProtocolError):
    """Base exception for Beam Session lifecycle violations."""


class GRSessionAlreadyExistsError(GRSessionError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            GRErrorCode.SESSION_ALREADY_EXISTS,
            f"beam session {session_id!r} already exists",
        )


class GRSessionNotFoundError(GRSessionError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            GRErrorCode.SESSION_NOT_FOUND,
            f"beam session {session_id!r} was not found",
        )


class GRInvalidSessionStateError(GRSessionError):
    def __init__(self, session_id: str, state: str, operation: str) -> None:
        self.session_id = session_id
        self.state = state
        self.operation = operation
        super().__init__(
            GRErrorCode.INVALID_SESSION_STATE,
            f"cannot {operation} beam session {session_id!r} while state={state}",
        )


class GRInvalidSequenceError(GRSessionError):
    def __init__(self, session_id: str, received: int, expected: int) -> None:
        self.session_id = session_id
        self.received = received
        self.expected = expected
        super().__init__(
            GRErrorCode.INVALID_SEQUENCE,
            f"beam session {session_id!r} received sequence_no={received}, expected={expected}",
        )
