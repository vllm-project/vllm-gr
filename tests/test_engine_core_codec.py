# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed V1 payloads and native EngineCore messages share the socket decoder."""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np
import pytest
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackEncoder

from tests.test_gr_async_scheduler import _batch_request
from vllm_gr.v1.beam.batch_contracts import BeamBatchItem, BeamBatchRequest
from vllm_gr.v1.engine.codec import (
    build_engine_core_decoders,
    decode_engine_core_payload,
)
from vllm_gr.v1.engine.wire import register_engine_core_request_types

register_engine_core_request_types()


def _frames(payload: Any) -> list[Any]:
    return list(MsgpackEncoder().encode(payload))


def test_decoders_share_oob_tensor_provider() -> None:
    provider = object()
    decoders = build_engine_core_decoders(provider)
    assert decoders.add_request.oob_tensor_provider is provider
    assert decoders.submit_gr_session.oob_tensor_provider is provider
    assert decoders.generic.oob_tensor_provider is provider


@pytest.mark.parametrize("batch_size", [1, 2, 4])  # type: ignore[misc]
def test_decode_whole_gr_batch(batch_size: int) -> None:
    source = _batch_request()
    items = tuple(
        BeamBatchItem(
            item_index=index,
            request=msgspec.structs.replace(source.items[0].request, request_id=f"session-{index}"),
        )
        for index in range(batch_size)
    )
    source = msgspec.structs.replace(source, items=items)
    decoded = decode_engine_core_payload(
        build_engine_core_decoders(None), EngineCoreRequestType.SUBMIT_GR_SESSION, _frames(source)
    )
    assert isinstance(decoded, BeamBatchRequest)
    assert [bytes(frame) for frame in _frames(decoded)] == [
        bytes(frame) for frame in _frames(source)
    ]
    for original, item in zip(source.items, decoded.items, strict=True):
        assert isinstance(item.request, EngineCoreRequest)
        tokens = item.request.prompt_token_ids
        assert isinstance(tokens, list)
        assert all(type(token) is int for token in tokens)
        assert tokens == list(original.request.prompt_token_ids)


def test_native_add_payload() -> None:
    source = _batch_request().items[0].request
    values = {name: getattr(source, name) for name in EngineCoreRequest.__struct_fields__}
    values["prompt_token_ids"] = list(source.prompt_token_ids)
    request = EngineCoreRequest(**values)
    decoded = decode_engine_core_payload(
        build_engine_core_decoders(None), EngineCoreRequestType.ADD, _frames(request)
    )
    assert isinstance(decoded, EngineCoreRequest)
    assert decoded == request


def test_native_abort_payload() -> None:
    request_ids = ["request-0", "request-1"]
    assert (
        decode_engine_core_payload(
            build_engine_core_decoders(None), EngineCoreRequestType.ABORT, _frames(request_ids)
        )
        == request_ids
    )


def test_invalid_v1_payload_rejected_at_decode() -> None:
    with pytest.raises(msgspec.ValidationError):
        decode_engine_core_payload(
            build_engine_core_decoders(None),
            EngineCoreRequestType.SUBMIT_GR_SESSION,
            _frames({"not": "a batch"}),
        )
