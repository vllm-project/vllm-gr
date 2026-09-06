# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import msgspec
import pytest

from vllm_gr.v1.engine.types import (
    GRRequestMetadata,
    GRRequestStage,
    GRSchedulerOutput,
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
            [
                "request-0",
                "decode_running",
                1,
                1,
                False,
                False,
                False,
                1,
                1,
                None,
                99,
            ],
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
        "logical_token_count",
        "physical_row_count",
        "prefix_block_ids",
        "protocol_version",
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GRSchedulerOutput(
            "request-0", GRRequestStage.DECODE_RUNNING, -1, 0, False
        ),
        lambda: GRSchedulerOutput(
            "request-0", GRRequestStage.DECODE_RUNNING, 0, -1, False
        ),
    ],
)
def test_wire_steps_must_be_non_negative(factory) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        factory()
