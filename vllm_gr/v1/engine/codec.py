# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Payload decoders for the vLLM-GR EngineCore wire protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder

from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


@dataclass(frozen=True)
class EngineCoreDecoders:
    add_request: MsgpackDecoder
    submit_gr_session: MsgpackDecoder
    generic: MsgpackDecoder


def build_engine_core_decoders(oob_tensor_provider: Any) -> EngineCoreDecoders:
    """Build one decoder set for an EngineCore input socket thread."""
    decoder_kwargs = {"oob_tensor_provider": oob_tensor_provider}
    return EngineCoreDecoders(
        add_request=MsgpackDecoder(EngineCoreRequest, **decoder_kwargs),
        submit_gr_session=MsgpackDecoder(BeamBatchRequest, **decoder_kwargs),
        generic=MsgpackDecoder(**decoder_kwargs),
    )


def decode_engine_core_payload(
    decoders: EngineCoreDecoders,
    request_type: EngineCoreRequestType,
    data_frames: Any,
) -> Any:
    """Decode payload frames using the schema selected by request type."""
    if request_type == EngineCoreRequestType.ADD:
        decoder = decoders.add_request
    elif request_type == EngineCoreRequestType.SUBMIT_GR_SESSION:
        decoder = decoders.submit_gr_session
    else:
        decoder = decoders.generic
    return decoder.decode(data_frames)
