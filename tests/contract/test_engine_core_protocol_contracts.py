# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors

"""Inventory tests for the vLLM-GR EngineCore protocol boundary."""

from __future__ import annotations

from types import SimpleNamespace

import msgspec
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder

from vllm_gr.v1.engine.scheduler_metadata import build_beam_request_metadata
from vllm_gr.v1.engine.types import BeamRequestStepUpdate
from vllm_gr.v1.engine.wire import (
    ADD_BATCH_WIRE_VALUE,
    BEAM_REQUEST_STEP_UPDATE_WIRE_VALUE,
    register_engine_core_request_types,
)

UPSTREAM_REQUEST_WIRE_VALUES = {
    "ADD": b"\x00",
    "ABORT": b"\x01",
    "START_DP_WAVE": b"\x02",
    "UTILITY": b"\x03",
    "EXECUTOR_FAILED": b"\x04",
    "WAKEUP": b"\x05",
}

GR_REQUEST_WIRE_VALUES = {
    "ADD_BATCH": b"\x06",
    "BEAM_REQUEST_STEP_UPDATE": b"\x07",
}

ENGINE_CORE_REQUEST_FIELDS = (
    "request_id",
    "prompt_token_ids",
    "mm_features",
    "sampling_params",
    "pooling_params",
    "arrival_time",
    "lora_request",
    "cache_salt",
    "data_parallel_rank",
    "prompt_embeds",
    "prompt_is_token_ids",
    "client_index",
    "current_wave",
    "priority",
    "trace_headers",
    "resumable",
    "external_req_id",
    "reasoning_ended",
    "reasoning_parser_kwargs",
    "abort_immediately",
)

BEAM_STEP_UPDATE_FIELDS = (
    "session_id",
    "parent_beam_ids",
    "child_beam_ids",
    "beam_tokens",
    "pruned_ids",
    "prefix_len",
    "beam_width",
    "sampling_params",
    "eos_token_id",
    "client_index",
    "current_wave",
    "priority",
    "data_parallel_rank",
    "lora_request",
    "cache_salt",
    "trace_headers",
)


def _sampling_params() -> SamplingParams:
    return SamplingParams(max_tokens=4, temperature=1.0)


def _minimal_beam_step_update(**overrides: object) -> BeamRequestStepUpdate:
    values = {
        "session_id": "session-0",
        "parent_beam_ids": ["parent-0", "parent-1"],
        "child_beam_ids": ["child-0", "child-1"],
        "beam_tokens": [[11], [12]],
        "pruned_ids": [],
        "prefix_len": 8,
        "beam_width": 2,
        "sampling_params": _sampling_params(),
    }
    values.update(overrides)
    return BeamRequestStepUpdate(**values)


def test_engine_core_wire_value_inventory() -> None:
    assert ADD_BATCH_WIRE_VALUE == b"\x06"
    assert BEAM_REQUEST_STEP_UPDATE_WIRE_VALUE == b"\x07"
    register_engine_core_request_types()
    register_engine_core_request_types()

    actual = {name: member.value for name, member in EngineCoreRequestType.__members__.items()}
    expected = UPSTREAM_REQUEST_WIRE_VALUES | GR_REQUEST_WIRE_VALUES

    assert actual == expected
    assert len(set(actual.values())) == len(actual)
    for name, wire_value in expected.items():
        assert EngineCoreRequestType(wire_value) is getattr(EngineCoreRequestType, name)


def test_add_batch_payload_schema() -> None:
    # Guard against upstream vLLM EngineCoreRequest field schema changes.
    assert EngineCoreRequest.__struct_fields__ == ENGINE_CORE_REQUEST_FIELDS

    request = EngineCoreRequest(
        request_id="request-0",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=_sampling_params(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    encoded = msgspec.msgpack.encode([request])

    raw_payload = msgspec.msgpack.decode(encoded)
    assert isinstance(raw_payload, list)
    assert len(raw_payload) == 1
    assert isinstance(raw_payload[0], list)
    assert raw_payload[0][0] == "request-0"

    decoded = MsgpackDecoder(list[EngineCoreRequest]).decode(encoded)
    assert len(decoded) == 1
    assert isinstance(decoded[0], EngineCoreRequest)
    assert decoded[0].request_id == "request-0"


def test_beam_step_update_field_inventory() -> None:
    assert BeamRequestStepUpdate.__struct_fields__ == BEAM_STEP_UPDATE_FIELDS


def test_beam_step_update_minimal_raw_wire_shape() -> None:
    encoded = msgspec.msgpack.encode(_minimal_beam_step_update())
    raw_payload = msgspec.msgpack.decode(encoded)

    assert isinstance(raw_payload, list)
    assert len(raw_payload) == 16
    assert raw_payload[:7] == [
        "session-0",
        ["parent-0", "parent-1"],
        ["child-0", "child-1"],
        [[11], [12]],
        [],
        8,
        2,
    ]
    assert raw_payload[8:] == [None, 0, 0, 0, None, None, None, None]

    decoded = MsgpackDecoder(BeamRequestStepUpdate).decode(encoded)
    assert decoded.session_id == "session-0"
    assert decoded.beam_width == 2
    assert decoded.eos_token_id is None
    assert decoded.trace_headers is None


def test_beam_step_update_full_raw_wire_shape() -> None:
    update = _minimal_beam_step_update(
        eos_token_id=2,
        client_index=3,
        current_wave=4,
        priority=5,
        data_parallel_rank=6,
        cache_salt="cache-salt",
        trace_headers={"traceparent": "trace-0"},
    )
    raw_payload = msgspec.msgpack.decode(msgspec.msgpack.encode(update))

    assert len(raw_payload) == len(BEAM_STEP_UPDATE_FIELDS)
    assert raw_payload[8:13] == [2, 3, 4, 5, 6]
    assert raw_payload[13] is None
    assert raw_payload[14:] == ["cache-salt", {"traceparent": "trace-0"}]


def test_scheduler_beam_metadata_shape() -> None:
    request = SimpleNamespace(
        request_id="session-0",
        is_beam_decode=True,
        beam_width=2,
        beam_decode_steps=3,
        prefix_len=8,
        num_computed_tokens=13,
    )

    metadata = build_beam_request_metadata(request, num_scheduled_tokens=5)

    assert metadata == {
        "is_beam_decode": True,
        "beam_width": 2,
        "beam_decode_steps": 3,
        "prefix_len": 8,
        "cache_len": 8,
    }


def test_non_beam_request_has_no_scheduler_metadata() -> None:
    request = SimpleNamespace(request_id="request-0", is_beam_decode=False)

    assert build_beam_request_metadata(request, num_scheduled_tokens=1) is None


def test_scheduler_output_accepts_beam_data_shape() -> None:
    output = SchedulerOutput.make_empty()
    metadata = {
        "is_beam_decode": True,
        "beam_width": 2,
        "beam_decode_steps": 3,
        "prefix_len": 8,
        "cache_len": 8,
    }

    output.beam_data = {"session-0": metadata}

    assert output.beam_data == {"session-0": metadata}
