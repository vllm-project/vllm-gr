# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoding, decoding, and boundary validation for GR commands."""

from __future__ import annotations

from collections.abc import Mapping

import msgspec
from vllm.sampling_params import SamplingParams

from vllm_gr.protocol.errors import (
    GRProtocolDecodeError,
    GRProtocolValidationError,
)
from vllm_gr.protocol.messages import (
    GRBeamStepRequest,
    GRCommandEnvelope,
    GREndBeamSessionRequest,
)
from vllm_gr.protocol.version import CURRENT_GR_PROTOCOL_VERSION, GRProtocolVersion


def _require_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise GRProtocolValidationError(f"{field_name} must be a non-empty string")


def _require_non_negative_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GRProtocolValidationError(
            f"{field_name} must be a non-negative integer, got {value!r}"
        )


def _require_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GRProtocolValidationError(f"{field_name} must be an integer, got {value!r}")


def _validate_beam_step(command: GRBeamStepRequest) -> None:
    _require_non_empty_string(command.session_id, "session_id")
    _require_non_negative_integer(command.sequence_no, "sequence_no")
    _require_non_negative_integer(command.prefix_len, "prefix_len")
    _require_integer(command.priority, "priority")

    beam_width = command.beam_width
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise GRProtocolValidationError(
            f"beam_width must be a positive integer, got {beam_width!r}"
        )
    if len(command.beam_token_ids) != beam_width:
        raise GRProtocolValidationError(
            "beam_token_ids length must equal "
            f"beam_width={beam_width}, got {len(command.beam_token_ids)}"
        )
    if not command.beam_token_ids or not command.beam_token_ids[0]:
        raise GRProtocolValidationError(
            "beam_token_ids entries must contain at least one decode token"
        )

    decode_steps = len(command.beam_token_ids[0])
    if any(len(tokens) != decode_steps for tokens in command.beam_token_ids):
        raise GRProtocolValidationError(
            "all beam_token_ids entries must have the same decode length"
        )
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for tokens in command.beam_token_ids
        for token_id in tokens
    ):
        raise GRProtocolValidationError("beam_token_ids must contain only non-negative integers")
    if not isinstance(command.sampling_params, SamplingParams):
        raise GRProtocolValidationError("sampling_params must be a SamplingParams instance")
    if command.eos_token_id is not None and (
        isinstance(command.eos_token_id, bool)
        or not isinstance(command.eos_token_id, int)
        or command.eos_token_id < 0
    ):
        raise GRProtocolValidationError("eos_token_id must be a non-negative integer or None")
    if command.trace_headers is not None and (
        not isinstance(command.trace_headers, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in command.trace_headers.items()
        )
    ):
        raise GRProtocolValidationError("trace_headers must be a string-to-string mapping or None")


def _validate_end_session(command: GREndBeamSessionRequest) -> None:
    _require_non_empty_string(command.session_id, "session_id")
    _require_non_negative_integer(command.sequence_no, "sequence_no")


def validate_envelope(envelope: GRCommandEnvelope) -> None:
    """Validate an envelope constructed locally or decoded from the wire."""
    if not isinstance(envelope, GRCommandEnvelope):
        raise GRProtocolValidationError("message must be a GRCommandEnvelope")

    _require_non_negative_integer(envelope.protocol_major, "protocol_major")
    _require_non_negative_integer(envelope.protocol_minor, "protocol_minor")
    if envelope.protocol_major == 0:
        raise GRProtocolValidationError("protocol_major must be positive")

    received_version = GRProtocolVersion(
        major=envelope.protocol_major,
        minor=envelope.protocol_minor,
    )
    received_version.ensure_supported_by(CURRENT_GR_PROTOCOL_VERSION)

    _require_non_empty_string(envelope.command_id, "command_id")
    _require_non_negative_integer(envelope.client_index, "client_index")
    _require_non_negative_integer(envelope.current_wave, "current_wave")

    match envelope.command:
        case GRBeamStepRequest():
            _validate_beam_step(envelope.command)
        case GREndBeamSessionRequest():
            _validate_end_session(envelope.command)
        case _:
            raise GRProtocolValidationError(
                f"unsupported GR command type: {type(envelope.command).__name__}"
            )


class GRProtocolCodec:
    """Reusable msgpack codec for one protocol endpoint.

    ``msgspec`` encoder/decoder instances are not generally thread-safe. A
    codec instance must therefore be owned by one IO thread, matching vLLM's
    existing EngineCore transport model.
    """

    def __init__(self) -> None:
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(GRCommandEnvelope)

    def encode(self, envelope: GRCommandEnvelope) -> bytes:
        validate_envelope(envelope)
        return self._encoder.encode(envelope)

    def decode(self, payload: bytes | bytearray | memoryview) -> GRCommandEnvelope:
        try:
            envelope = self._decoder.decode(payload)
        except (msgspec.DecodeError, msgspec.ValidationError) as exc:
            raise GRProtocolDecodeError(f"invalid GR protocol payload: {exc}") from exc
        validate_envelope(envelope)
        return envelope
