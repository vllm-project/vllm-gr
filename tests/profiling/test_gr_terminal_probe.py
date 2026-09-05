# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for terminal-probe Beam slot restoration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.profiling.gr_terminal_probe import _beam_slots_restored
from vllm_gr.v1.beam.beam_session_registry import BeamSessionRegistry


@pytest.mark.parametrize("capacity", [1, 3])
def test_fresh_registry_accepts_lazy_and_initialized_free_baselines(capacity) -> None:
    registry = BeamSessionRegistry(max_batch=capacity)

    assert _beam_slots_restored(registry, initial_free=None) is True
    assert _beam_slots_restored(registry, initial_free=capacity) is True


@pytest.mark.parametrize("capacity", [1, 3])
def test_real_allocate_release_restores_the_whole_registry(capacity) -> None:
    registry = BeamSessionRegistry(max_batch=capacity)
    initial_free = len(registry._free_slots)

    slot = registry.allocate("session-0", beam_width=128, prefix_len=1024)
    assert slot == 0
    assert _beam_slots_restored(registry, initial_free) is False

    registry.release("session-0")
    assert _beam_slots_restored(registry, initial_free) is True


@pytest.mark.parametrize("capacity", [1, 3])
def test_active_real_session_is_not_reported_as_restored(capacity) -> None:
    registry = BeamSessionRegistry(max_batch=capacity)
    registry.allocate("active", beam_width=64, prefix_len=128)

    assert _beam_slots_restored(registry, initial_free=capacity) is False


@pytest.mark.parametrize(
    "corrupt_free",
    [
        [0, 1, 2, 0],  # duplicate
        [0, 1],  # missing
        [0, 1, 3],  # out of range while retaining the expected length
    ],
)
def test_duplicate_missing_and_out_of_range_free_lists_are_rejected(
    corrupt_free,
) -> None:
    registry = BeamSessionRegistry(max_batch=3)
    registry._free_slots = corrupt_free

    assert _beam_slots_restored(registry, initial_free=3) is False


@pytest.mark.parametrize("capacity", [1, 3])
def test_leftover_slot_state_is_rejected_even_with_a_complete_free_list(
    capacity,
) -> None:
    registry = BeamSessionRegistry(max_batch=capacity)
    registry._slots[0] = SimpleNamespace(session_id="stale", slot=0)

    assert _beam_slots_restored(registry, initial_free=capacity) is False


@pytest.mark.parametrize("capacity", [1, 3])
def test_restored_registry_must_match_initialized_free_count(capacity) -> None:
    registry = BeamSessionRegistry(max_batch=capacity)
    mismatched_initial = capacity + 1

    assert _beam_slots_restored(registry, mismatched_initial) is False
