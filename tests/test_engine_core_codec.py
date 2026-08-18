# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for EngineCore payload decoder selection."""

from __future__ import annotations

from typing import Any

import msgspec
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType

from vllm_gr.v1.beam.controller_types import BeamControllerRequest
from vllm_gr.v1.engine.codec import (
    EngineCoreDecoders,
    build_engine_core_decoders,
    decode_engine_core_payload,
)
from vllm_gr.v1.engine.types import (
    BeamControllerCancelRequest,
    BeamControllerStartRequest,
    BeamRequestStepUpdate,
)
from vllm_gr.v1.engine.wire import register_engine_core_request_types

register_engine_core_request_types()


def _sampling_params() -> SamplingParams:
    return SamplingParams(max_tokens=4, temperature=1.0)


def _engine_core_request(request_id: str) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=_sampling_params(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _beam_step_update() -> BeamRequestStepUpdate:
    return BeamRequestStepUpdate(
        session_id="session-0",
        parent_beam_ids=["parent-0", "parent-1"],
        child_beam_ids=["child-0", "child-1"],
        beam_tokens=[[11], [12]],
        pruned_ids=[],
        prefix_len=8,
        beam_width=2,
        sampling_params=_sampling_params(),
    )


def _frames(payload: Any) -> list[bytes]:
    return [msgspec.msgpack.encode(payload)]


def test_build_engine_core_decoders_shares_oob_tensor_provider() -> None:
    provider = object()

    decoders = build_engine_core_decoders(provider)

    assert isinstance(decoders, EngineCoreDecoders)
    assert decoders.add_request.oob_tensor_provider is provider
    assert decoders.add_batch.oob_tensor_provider is provider
    assert decoders.beam_step_update.oob_tensor_provider is provider
    assert decoders.beam_controller_start.oob_tensor_provider is provider
    assert decoders.beam_controller_cancel.oob_tensor_provider is provider
    assert decoders.generic.oob_tensor_provider is provider


def test_decode_add_batch_payload() -> None:
    decoders = build_engine_core_decoders(None)
    requests = [
        _engine_core_request("request-0"),
        _engine_core_request("request-1"),
    ]

    decoded = decode_engine_core_payload(
        decoders,
        EngineCoreRequestType.ADD_BATCH,
        _frames(requests),
    )

    assert [request.request_id for request in decoded] == [
        "request-0",
        "request-1",
    ]
    assert all(isinstance(request, EngineCoreRequest) for request in decoded)


def test_decode_beam_step_update_payload() -> None:
    decoders = build_engine_core_decoders(None)

    decoded = decode_engine_core_payload(
        decoders,
        EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE,
        _frames(_beam_step_update()),
    )

    assert isinstance(decoded, BeamRequestStepUpdate)
    assert decoded.session_id == "session-0"
    assert decoded.beam_width == 2


def test_decode_beam_controller_start_and_cancel_payloads() -> None:
    decoders = build_engine_core_decoders(None)
    controller = BeamControllerRequest(
        beam_session_id="session-0",
        external_request_id="request-0",
        beam_width=2,
        max_tokens=4,
        prompt_token_ids=[1, 2],
        initial_tokens=[1, 2],
        prompt_length=2,
        eos_token_id=3,
    )
    start = BeamControllerStartRequest(
        controller=controller,
        engine_request=_engine_core_request("session-0"),
        call_id=7,
        constraint_resource={
            "format_version": 1,
            "backend": "trie",
            "arrays": {"node_offsets": {"data": b"\x00", "dtype": "int64"}},
            "digest": "digest-0",
        },
    )
    decoded_start = decode_engine_core_payload(
        decoders,
        EngineCoreRequestType.BEAM_CONTROLLER_START,
        _frames(start),
    )
    assert isinstance(decoded_start, BeamControllerStartRequest)
    assert isinstance(decoded_start.engine_request, EngineCoreRequest)
    assert decoded_start.controller.beam_session_id == "session-0"
    assert decoded_start.constraint_resource == start.constraint_resource

    cancel = BeamControllerCancelRequest("session-0")
    decoded_cancel = decode_engine_core_payload(
        decoders,
        EngineCoreRequestType.BEAM_CONTROLLER_CANCEL,
        _frames(cancel),
    )
    assert isinstance(decoded_cancel, BeamControllerCancelRequest)
    assert decoded_cancel.beam_session_id == "session-0"
