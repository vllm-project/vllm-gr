# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import msgspec

from vllm_gr.v1.engine.types import (
    GRModelRunnerOutput,
    GRRequestMetadata,
    GRRequestOutput,
    GRRequestStage,
    GRSchedulerOutput,
    GRSequenceOutput,
)


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
        max_tokens=5,
        prefix_len=128,
        continue_token_id=1,
        finish_token_id=2,
        num_return_sequences=4,
        return_logprobs=True,
        constraint_digest="sha256:" + "a" * 64,
    )
    payload = msgspec.msgpack.encode(metadata)
    assert msgspec.msgpack.decode(payload, type=GRRequestMetadata) == metadata


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
