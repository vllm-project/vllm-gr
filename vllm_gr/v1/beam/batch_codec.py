# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Typed MsgPack codec for Beam batch transport contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import msgspec
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchPlan,
    BeamBatchRequest,
    BeamBatchResult,
    validate_beam_batch_plan,
    validate_beam_batch_request,
    validate_beam_batch_result,
)

BeamBatchFrames: TypeAlias = Sequence[bytestr]
BeamBatchEncoded: TypeAlias = bytestr | Sequence[bytestr]


class BeamBatchCodecError(ValueError):
    """Raised when Beam batch frames cannot be encoded or decoded."""


class BeamBatchCodec:
    """Typed encoders and decoders for the three top-level contracts.

    The underlying vLLM tensor-aware codec is not thread-safe. Use one
    instance per transport thread.
    """

    def __init__(
        self,
        *,
        oob_tensor_consumer: Any = None,
        oob_tensor_provider: Any = None,
    ) -> None:
        self._encoder = (
            MsgpackEncoder()
            if oob_tensor_consumer is None
            else MsgpackEncoder(oob_tensor_consumer=oob_tensor_consumer)
        )
        decoder_kwargs = (
            {} if oob_tensor_provider is None else {"oob_tensor_provider": oob_tensor_provider}
        )
        self._request_decoder = MsgpackDecoder(BeamBatchRequest, **decoder_kwargs)
        self._plan_decoder = MsgpackDecoder(BeamBatchPlan, **decoder_kwargs)
        self._result_decoder = MsgpackDecoder(BeamBatchResult, **decoder_kwargs)

    def encode_request(self, request: BeamBatchRequest) -> BeamBatchFrames:
        validate_beam_batch_request(request)
        return self._encode(request, "BeamBatchRequest")

    def decode_request(self, frames: BeamBatchEncoded) -> BeamBatchRequest:
        request = self._decode(self._request_decoder, frames, "BeamBatchRequest")
        validate_beam_batch_request(request)
        return request

    def encode_plan(self, plan: BeamBatchPlan) -> BeamBatchFrames:
        validate_beam_batch_plan(plan)
        return self._encode(plan, "BeamBatchPlan")

    def decode_plan(self, frames: BeamBatchEncoded) -> BeamBatchPlan:
        plan = self._decode(self._plan_decoder, frames, "BeamBatchPlan")
        validate_beam_batch_plan(plan)
        return plan

    def encode_result(self, result: BeamBatchResult) -> BeamBatchFrames:
        validate_beam_batch_result(result)
        return self._encode(result, "BeamBatchResult")

    def decode_result(self, frames: BeamBatchEncoded) -> BeamBatchResult:
        result = self._decode(self._result_decoder, frames, "BeamBatchResult")
        validate_beam_batch_result(result)
        return result

    def _encode(self, value: Any, type_name: str) -> BeamBatchFrames:
        try:
            return self._encoder.encode(value)
        except (TypeError, ValueError, msgspec.EncodeError) as exc:
            raise BeamBatchCodecError(f"failed to encode {type_name}: {exc}") from exc

    @staticmethod
    def _decode(
        decoder: MsgpackDecoder,
        frames: BeamBatchEncoded,
        type_name: str,
    ) -> Any:
        try:
            return decoder.decode(frames)
        except (TypeError, ValueError, msgspec.DecodeError, IndexError) as exc:
            raise BeamBatchCodecError(f"failed to decode {type_name}: {exc}") from exc


__all__ = ["BeamBatchCodec", "BeamBatchCodecError", "BeamBatchFrames"]
