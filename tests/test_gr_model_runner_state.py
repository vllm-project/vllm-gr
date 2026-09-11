# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.persistent_contracts import (
    GRBeamSessionSpec,
    GRStageKind,
    GRWorkerStageSpec,
)
from vllm_gr.v1.worker.gr_model_runner_state import (
    GRModelRunnerStateRegistry,
    GRModelRunnerStatus,
)


def _context(max_batch: int = 2) -> BeamAttentionContext:
    return BeamAttentionContext(
        max_batch=max_batch,
        max_beams=4,
        num_kv_heads=1,
        max_decode_steps=5,
        head_dim=8,
        num_layers=1,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )


def _spec(session_id: str = "session-0", **overrides: object) -> GRBeamSessionSpec:
    values: dict[str, object] = {
        "session_id": session_id,
        "request_id": f"request-{session_id}",
        "beam_width": 2,
        "max_tokens": 3,
        "constraint_handle": "default",
        "temperature": 1.0,
        "ignore_eos": True,
        "early_stopping": False,
        "ignore_stop_tokens": True,
    }
    values.update(overrides)
    return GRBeamSessionSpec(**values)  # type: ignore[arg-type]


def _stage(index: int, max_tokens: int = 3) -> GRWorkerStageSpec:
    return GRWorkerStageSpec(
        session_id="session-0",
        stage_index=index,
        stage_kind=GRStageKind.PREFILL if index == 0 else GRStageKind.DECODE,
        is_last_stage=index == max_tokens - 1,
    )


def test_registry_allocates_stable_slot_views_and_tombstones_ids() -> None:
    registry = GRModelRunnerStateRegistry(_context(max_batch=1))
    state = registry.create(_spec())
    pointers = state.buffers.data_ptrs()

    assert state.status is GRModelRunnerStatus.NEW
    assert state.slot_id == 0
    assert state.buffers.generated_token_ids.shape == (1, 2, 5)
    assert state.buffers.parent_beam_indices.shape == (1, 2, 5)
    assert state.buffers.data_ptrs() == pointers

    registry.release("session-0")
    registry.release("session-0")
    assert "session-0" not in registry._states
    with pytest.raises(ValueError, match="already created"):
        registry.create(_spec())


def test_registry_two_slots_keep_device_state_isolated() -> None:
    registry = GRModelRunnerStateRegistry(_context())
    first = registry.create(_spec("first"))
    first.buffers.generated_token_ids.fill_(11)
    first.buffers.cumulative_logprobs.fill_(-1.25)

    second = registry.create(_spec("second"))
    second.buffers.generated_token_ids.fill_(22)
    second.buffers.cumulative_logprobs.fill_(-2.5)

    torch.testing.assert_close(
        first.buffers.generated_token_ids,
        torch.full_like(first.buffers.generated_token_ids, 11),
    )
    torch.testing.assert_close(
        first.buffers.cumulative_logprobs,
        torch.full_like(first.buffers.cumulative_logprobs, -1.25),
    )
    assert first.slot_id != second.slot_id
    assert first.buffers.generated_token_ids.data_ptr() != (
        second.buffers.generated_token_ids.data_ptr()
    )


def test_registry_exhaustion_unknown_abort_and_idempotent_abort() -> None:
    registry = GRModelRunnerStateRegistry(_context(max_batch=1))
    registry.create(_spec())
    with pytest.raises(RuntimeError, match="No free beam session slots"):
        registry.create(_spec("session-1"))
    with pytest.raises(KeyError, match="unknown"):
        registry.abort("missing")

    registry.abort("session-0")
    version = registry.get("session-0").state_version
    registry.abort("session-0")
    assert registry.get("session-0").state_version == version


def test_slot_is_not_reused_before_terminal_event_completion() -> None:
    registry = GRModelRunnerStateRegistry(_context(max_batch=1))
    state = registry.create(_spec())
    event = SimpleNamespace(done=False, query=lambda: event.done)
    state.terminal_event = event
    state.terminal_event_recorded = True

    registry.release("session-0")
    with pytest.raises(RuntimeError, match="No free beam session slots"):
        registry.create(_spec("session-1"))

    event.done = True
    registry.reap_completed()
    assert "session-0" not in registry._states
    reused = registry.create(_spec("session-1"))
    assert reused.slot_id == 0


def test_completed_session_state_and_old_tombstones_are_reclaimed() -> None:
    registry = GRModelRunnerStateRegistry(_context(max_batch=1))
    for index in range(100):
        session_id = f"session-{index}"
        registry.create(_spec(session_id))
        registry.release(session_id)

    assert registry.active_session_count == 0
    assert not registry._states
    assert not registry._pending_release
    assert len(registry._released_session_ids) <= (
        registry._max_released_session_tombstones
    )


def test_stage_validation_rejects_duplicate_out_of_order_and_bad_terminal() -> None:
    state = GRModelRunnerStateRegistry(_context()).create(_spec())
    state.validate_stage(_stage(0))
    state.next_expected_stage = 1

    with pytest.raises(RuntimeError, match="expected stage 1, got 0"):
        state.validate_stage(_stage(0))
    with pytest.raises(RuntimeError, match="expected stage 1, got 2"):
        state.validate_stage(_stage(2))
    with pytest.raises(ValueError, match="is_last_stage must be False"):
        state.validate_stage(GRWorkerStageSpec("session-0", 1, GRStageKind.DECODE, True))


def test_five_stage_capacity_is_a_lightweight_lifecycle_contract() -> None:
    state = GRModelRunnerStateRegistry(_context()).create(_spec(max_tokens=5))
    pointers = state.buffers.data_ptrs()
    for index in range(5):
        state.validate_stage(_stage(index, max_tokens=5))
        state.next_expected_stage += 1
    assert state.next_expected_stage == 5
    assert state.buffers.data_ptrs() == pointers
