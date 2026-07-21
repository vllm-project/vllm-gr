# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import msgspec
import pytest
from vllm.sampling_params import SamplingParams

from vllm_gr.protocol.codec import GRProtocolCodec
from vllm_gr.protocol.errors import (
    GRErrorCode,
    GRIncompatibleVersionError,
    GRProtocolDecodeError,
    GRProtocolValidationError,
)
from vllm_gr.protocol.messages import (
    GRBeamStepRequest,
    GRCommandEnvelope,
    GREndBeamSessionRequest,
)


def _sampling_params() -> SamplingParams:
    return SamplingParams(max_tokens=1, temperature=1.0, logprobs=5)


def _step(**overrides: object) -> GRBeamStepRequest:
    values: dict[str, object] = {
        "session_id": "session-1",
        "sequence_no": 0,
        "beam_width": 2,
        "beam_token_ids": [[10, 11], [20, 21]],
        "prefix_len": 4,
        "sampling_params": _sampling_params(),
        "eos_token_id": 2,
        "priority": 0,
        "trace_headers": {"traceparent": "00-test"},
    }
    values.update(overrides)
    return GRBeamStepRequest(**values)  # type: ignore[arg-type]


def _envelope(command: object | None = None, **overrides: object) -> GRCommandEnvelope:
    values: dict[str, object] = {
        "command_id": "command-1",
        "command": _step() if command is None else command,
        "client_index": 3,
        "current_wave": 7,
    }
    values.update(overrides)
    return GRCommandEnvelope(**values)  # type: ignore[arg-type]


def test_beam_step_round_trip() -> None:
    codec = GRProtocolCodec()

    decoded = codec.decode(codec.encode(_envelope()))

    assert decoded.command_id == "command-1"
    assert decoded.client_index == 3
    assert decoded.current_wave == 7
    assert isinstance(decoded.command, GRBeamStepRequest)
    assert decoded.command.session_id == "session-1"
    assert decoded.command.sequence_no == 0
    assert decoded.command.beam_width == 2
    assert decoded.command.beam_token_ids == [[10, 11], [20, 21]]
    assert decoded.command.sampling_params.max_tokens == 1
    assert decoded.command.sampling_params.n == 1
    assert decoded.command.trace_headers == {"traceparent": "00-test"}


def test_end_session_round_trip_is_a_distinct_command() -> None:
    codec = GRProtocolCodec()
    command = GREndBeamSessionRequest(session_id="session-1", sequence_no=2)

    decoded = codec.decode(codec.encode(_envelope(command)))

    assert isinstance(decoded.command, GREndBeamSessionRequest)
    assert decoded.command.session_id == "session-1"
    assert decoded.command.sequence_no == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"beam_width": 0}, "positive integer"),
        ({"beam_width": True}, "positive integer"),
        ({"beam_width": 3}, "length must equal"),
        ({"beam_token_ids": [[], []]}, "at least one decode token"),
        ({"beam_token_ids": [[1], [2, 3]]}, "same decode length"),
        ({"beam_token_ids": [[1], [True]]}, "non-negative integers"),
        ({"beam_token_ids": [[1], [-2]]}, "non-negative integers"),
        ({"prefix_len": -1}, "prefix_len"),
        ({"sequence_no": -1}, "sequence_no"),
        ({"session_id": ""}, "session_id"),
    ],
)
def test_invalid_beam_step_is_rejected_before_encode(
    overrides: dict[str, object],
    message: str,
) -> None:
    codec = GRProtocolCodec()

    with pytest.raises(GRProtocolValidationError, match=message) as exc_info:
        codec.encode(_envelope(_step(**overrides)))

    assert exc_info.value.code is GRErrorCode.INVALID_MESSAGE


def test_newer_major_version_is_rejected() -> None:
    codec = GRProtocolCodec()

    with pytest.raises(GRIncompatibleVersionError) as exc_info:
        codec.encode(_envelope(protocol_major=2))

    assert exc_info.value.code is GRErrorCode.INCOMPATIBLE_VERSION
    assert exc_info.value.received.major == 2


def test_newer_minor_version_is_rejected() -> None:
    codec = GRProtocolCodec()

    with pytest.raises(GRIncompatibleVersionError):
        codec.encode(_envelope(protocol_minor=1))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"protocol_major": 0}, "protocol_major must be positive"),
        ({"protocol_major": True}, "protocol_major must be a non-negative integer"),
        ({"protocol_minor": -1}, "protocol_minor must be a non-negative integer"),
    ],
)
def test_invalid_protocol_version_is_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    codec = GRProtocolCodec()

    with pytest.raises(GRProtocolValidationError, match=message):
        codec.encode(_envelope(**overrides))


def test_older_minor_version_is_accepted() -> None:
    codec = GRProtocolCodec()
    envelope = _envelope(protocol_minor=0)

    assert codec.decode(codec.encode(envelope)) == envelope


def test_malformed_payload_has_structured_decode_error() -> None:
    codec = GRProtocolCodec()

    with pytest.raises(GRProtocolDecodeError) as exc_info:
        codec.decode(b"not-msgpack")

    assert exc_info.value.code is GRErrorCode.DECODE_ERROR


def test_unknown_command_tag_has_structured_decode_error() -> None:
    codec = GRProtocolCodec()
    payload = msgspec.msgpack.encode(
        {
            "command_id": "command-1",
            "command": {"type": "unknown", "session_id": "session-1"},
        }
    )

    with pytest.raises(GRProtocolDecodeError, match="invalid GR protocol payload"):
        codec.decode(payload)


def test_unknown_envelope_field_is_rejected() -> None:
    codec = GRProtocolCodec()
    payload = msgspec.msgpack.encode(
        {
            "command_id": "command-1",
            "command": {
                "type": "end_session",
                "session_id": "session-1",
                "sequence_no": 0,
            },
            "unexpected": True,
        }
    )

    with pytest.raises(GRProtocolDecodeError):
        codec.decode(payload)
