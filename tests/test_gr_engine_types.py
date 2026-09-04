# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import msgspec
import pytest

from vllm_gr.v1.engine.types import (
    GRModelRunnerOutput,
    GRRequestMetadata,
    GRRequestOutput,
    GRRequestStage,
    GRSchedulerOutput,
    GRSequenceOutput,
    canonicalize_constraint_digest,
)

_DIGEST = "sha256:" + "a" * 64


def test_gr_request_stage_transitions() -> None:
    assert GRRequestStage.NEW.can_transition_to(GRRequestStage.PREFILL_RUNNING)
    assert GRRequestStage.PREFILL_RUNNING.can_transition_to(
        GRRequestStage.PREFILL_BOOTSTRAPPED
    )
    assert GRRequestStage.PREFILL_RUNNING.can_transition_to(
        GRRequestStage.FINAL_READY
    )
    assert GRRequestStage.PREFILL_BOOTSTRAPPED.can_transition_to(
        GRRequestStage.DECODE_RUNNING
    )
    assert GRRequestStage.DECODE_RUNNING.can_transition_to(
        GRRequestStage.DECODE_RUNNING
    )
    assert GRRequestStage.DECODE_RUNNING.can_transition_to(
        GRRequestStage.FINAL_READY
    )
    assert GRRequestStage.FINAL_READY.can_transition_to(GRRequestStage.RELEASED)
    assert GRRequestStage.RELEASED.can_transition_to(GRRequestStage.RELEASED)
    assert not GRRequestStage.NEW.can_transition_to(GRRequestStage.DECODE_RUNNING)
    assert not GRRequestStage.FINAL_READY.can_transition_to(
        GRRequestStage.DECODE_RUNNING
    )


def test_gr_request_metadata_msgpack_round_trip() -> None:
    metadata = GRRequestMetadata(
        request_id="request-0",
        beam_width=64,
        generation_steps=5,
        prefix_len=128,
        result_width=4,
        constraint_digest=_DIGEST,
    )
    payload = msgspec.msgpack.encode(metadata)
    assert msgspec.msgpack.decode(payload, type=GRRequestMetadata) == metadata
    assert set(metadata.__struct_fields__) == {
        "request_id",
        "beam_width",
        "generation_steps",
        "prefix_len",
        "result_width",
        "constraint_digest",
        "protocol_version",
    }


def test_gr_request_metadata_is_fixed_and_validated() -> None:
    metadata = GRRequestMetadata("request-0", 4, 3, 16, 2, _DIGEST)
    with pytest.raises(AttributeError):
        metadata.beam_width = 8  # type: ignore[misc]
    with pytest.raises(ValueError, match="result_width"):
        GRRequestMetadata("request-0", 4, 3, 16, 5, _DIGEST)


def test_constraint_digest_is_required_and_canonical() -> None:
    assert canonicalize_constraint_digest("a" * 64) == _DIGEST
    assert canonicalize_constraint_digest(_DIGEST) == _DIGEST
    for invalid in (None, "", "sha256:test", "A" * 64):
        with pytest.raises(ValueError, match="constraint_digest"):
            canonicalize_constraint_digest(invalid)
        with pytest.raises(ValueError, match="constraint_digest"):
            GRRequestMetadata("request-0", 4, 3, 16, 2, invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GRRequestMetadata(
            "request-0", 4, 3, 16, constraint_digest=_DIGEST, protocol_version=99
        ),
        lambda: GRSchedulerOutput(
            "request-0",
            GRRequestStage.DECODE_RUNNING,
            1,
            1,
            False,
            protocol_version=99,
        ),
        lambda: GRRequestOutput("request-0", [], 1, 1, protocol_version=99),
        lambda: GRModelRunnerOutput(
            "request-0",
            0,
            False,
            1,
            1,
            GRRequestStage.DECODE_RUNNING,
            protocol_version=99,
        ),
    ],
)
def test_protocol_version_mismatch_fails_fast(factory) -> None:
    with pytest.raises(ValueError, match="protocol_version"):
        factory()


@pytest.mark.parametrize(
    ("wire_type", "raw"),
    [
        (GRRequestMetadata, ["request-0", 4, 3, 16, 1, _DIGEST, 99]),
        (
            GRSchedulerOutput,
            ["request-0", "decode_running", 1, 1, False, False, False, None, 99],
        ),
        (GRRequestOutput, ["request-0", [], 1, 1, "final_ready", None, 99]),
        (
            GRModelRunnerOutput,
            ["request-0", 0, False, 1, 1, "decode_running", None, None, 99],
        ),
    ],
)
def test_decoded_protocol_version_mismatch_fails_fast(wire_type, raw) -> None:
    payload = msgspec.msgpack.encode(raw)
    with pytest.raises(ValueError, match="protocol_version"):
        msgspec.msgpack.decode(payload, type=wire_type)


def test_scheduler_output_contains_no_beam_state() -> None:
    output = GRSchedulerOutput(
        request_id="request-0",
        stage=GRRequestStage.DECODE_RUNNING,
        step=1,
        device_step=1,
        is_prefill=False,
        prefix_block_ids=[[1, 2]],
    )
    assert set(output.__struct_fields__) == {
        "request_id",
        "stage",
        "step",
        "device_step",
        "is_prefill",
        "is_last_prefill",
        "is_final",
        "prefix_block_ids",
        "protocol_version",
    }


def test_model_runner_output_is_lightweight_until_final() -> None:
    intermediate = GRModelRunnerOutput(
        request_id="request-0",
        token_id=1,
        finished=False,
        step=1,
        device_step=1,
        stage=GRRequestStage.DECODE_RUNNING,
    )
    assert intermediate.final_output is None

    final = GRRequestOutput(
        request_id="request-0",
        outputs=[GRSequenceOutput([10, 11], [0, 0], -0.25, [-0.1, -0.15])],
        step=2,
        device_step=2,
    )
    output = GRModelRunnerOutput(
        request_id="request-0",
        token_id=2,
        finished=True,
        step=2,
        device_step=2,
        stage=GRRequestStage.FINAL_READY,
        final_output=final,
    )
    payload = msgspec.msgpack.encode(output)
    assert msgspec.msgpack.decode(payload, type=GRModelRunnerOutput) == output


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GRSchedulerOutput(
            "request-0", GRRequestStage.DECODE_RUNNING, -1, 0, False
        ),
        lambda: GRSchedulerOutput(
            "request-0", GRRequestStage.DECODE_RUNNING, 0, -1, False
        ),
        lambda: GRRequestOutput("request-0", [], -1, 0),
        lambda: GRRequestOutput("request-0", [], 0, -1),
        lambda: GRModelRunnerOutput(
            "request-0", 0, False, -1, 0, GRRequestStage.DECODE_RUNNING
        ),
        lambda: GRModelRunnerOutput(
            "request-0", 0, False, 0, -1, GRRequestStage.DECODE_RUNNING
        ),
    ],
)
def test_wire_steps_must_be_non_negative(factory) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        factory()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"request_id": "other"}, "request_id"),
        ({"step": 3}, "step"),
        ({"device_step": 3}, "device_step"),
        ({"stage": GRRequestStage.DECODE_RUNNING}, "stage"),
        ({"finished": False}, "finished"),
    ],
)
def test_final_envelope_fields_must_match(override, message) -> None:
    final = GRRequestOutput("request-0", [], 2, 2)
    fields = {
        "request_id": "request-0",
        "token_id": 0,
        "finished": True,
        "step": 2,
        "device_step": 2,
        "stage": GRRequestStage.FINAL_READY,
        "final_output": final,
    }
    fields.update(override)
    with pytest.raises(ValueError, match=message):
        GRModelRunnerOutput(**fields)


def test_final_request_output_rejects_non_final_stage() -> None:
    with pytest.raises(ValueError, match="final_ready"):
        GRRequestOutput(
            "request-0", [], 2, 2, stage=GRRequestStage.DECODE_RUNNING
        )
