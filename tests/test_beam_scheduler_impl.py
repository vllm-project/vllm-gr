# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the issue #39 scheduler orchestration data structures."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_gr.v1.engine.beam_group import (
    BeamBatchGroup,
    BeamGroupingConfig,
    BeamGroupState,
    BeamPhase,
    UnitReservation,
)
from vllm_gr.v1.engine.beam_scheduler import (
    build_decode_output,
    build_prefill_output,
    release_unit_resources,
    reserve_unit_resources,
    schedule,
    try_admit_group,
)


def _child(request_id: str, num_tokens: int, num_prompt_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=0,
        prompt_token_ids=[0] * num_prompt_tokens,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
    )


def _member(batch_id: str, beam_width: int, children: tuple) -> SimpleNamespace:
    return SimpleNamespace(
        batch_id=batch_id,
        request=SimpleNamespace(beam_params=SimpleNamespace(beam_width=beam_width)),
        children=children,
    )


class _FakeBlocks:
    def __init__(self, block_ids: tuple[list[int], ...]):
        self._block_ids = block_ids

    def get_block_ids(self) -> tuple[list[int], ...]:
        return self._block_ids


class _FakeKV:
    def __init__(self, *, fail_at: str | None = None):
        self.fail_at = fail_at
        self.allocated: list[str] = []
        self.freed: list[str] = []

    def get_computed_blocks(self, child):
        return _FakeBlocks(([],)), 0

    def allocate_slots(self, child, *args, **kwargs):
        if child.request_id == self.fail_at:
            return None
        self.allocated.append(child.request_id)
        return _FakeBlocks(([len(self.allocated)],))

    def free(self, child) -> None:
        self.freed.append(child.request_id)


def _scheduler(kv: _FakeKV, beam_pool: object | None = None) -> SimpleNamespace:
    sched = SimpleNamespace(kv_cache_manager=kv)
    if beam_pool is not None:
        sched.beam_pool = beam_pool
    return sched


def test_group_children_and_batch_ids_are_flattened() -> None:
    child0 = _child("r0", 4, 4)
    child1 = _child("r1", 2, 2)
    group = BeamBatchGroup(
        group_id="g0",
        members=(_member("b0", 2, (child0,)), _member("b1", 3, (child1,))),
    )

    assert group.children == (child0, child1)
    assert group.batch_ids == ("b0", "b1")


def test_reservation_requires_per_child_entries() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))

    with pytest.raises(ValueError, match="blocks_by_child"):
        UnitReservation(
            unit=group,
            blocks_by_child={},
            num_computed_by_child={"r0": 0},
        )


def test_grouping_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="group_target_sizes"):
        BeamGroupingConfig(group_target_sizes=(0,))


def test_reserve_unit_resources_success_and_release() -> None:
    child0 = _child("r0", 4, 4)
    child1 = _child("r1", 2, 2)
    group = BeamBatchGroup(
        group_id="g0",
        members=(_member("b0", 2, (child0,)), _member("b1", 3, (child1,))),
    )
    kv = _FakeKV()
    reservation = reserve_unit_resources(_scheduler(kv), group)

    assert reservation is not None
    assert set(reservation.blocks_by_child) == {"r0", "r1"}
    assert reservation.hold_blocks is True

    release_unit_resources(_scheduler(kv), reservation)
    assert set(kv.freed) == {"r0", "r1"}


def test_reserve_unit_resources_rolls_back_kv_on_failure() -> None:
    child0 = _child("r0", 4, 4)
    child1 = _child("r1", 2, 2)
    group = BeamBatchGroup(
        group_id="g0",
        members=(_member("b0", 2, (child0,)), _member("b1", 3, (child1,))),
    )
    kv = _FakeKV(fail_at="r1")
    reservation = reserve_unit_resources(_scheduler(kv), group)

    assert reservation is None
    assert kv.freed == ["r0"]


def test_build_prefill_output_has_full_prompt_schedule() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    reservation = UnitReservation(
        unit=group,
        blocks_by_child={"r0": _FakeBlocks(([7],))},
        num_computed_by_child={"r0": 0},
    )
    output = build_prefill_output(reservation)

    assert output.total_num_scheduled_tokens == 4
    assert output.scheduled_new_reqs[0].req_id == "r0"


def test_build_decode_output_uses_prefix_as_computed() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    reservation = UnitReservation(
        unit=group,
        blocks_by_child={"r0": _FakeBlocks(([7],))},
        num_computed_by_child={"r0": 0},
    )
    state = BeamGroupState(unit=group, reservation=reservation, phase=BeamPhase.PREFILL)
    output = build_decode_output(state)

    assert output.scheduled_cached_reqs.req_ids == ["r0"]
    assert output.scheduled_cached_reqs.num_computed_tokens == [4]
    assert output.scheduled_cached_reqs.num_output_tokens == [1]
    assert output.beam_data["r0"]["is_beam_decode_run"] is True


def test_try_admit_group_reserves_and_builds_prefill() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    admitted = try_admit_group(_scheduler(_FakeKV()), group)

    assert admitted is not None
    output, state = admitted
    assert state.phase == BeamPhase.PREFILL
    assert output.scheduled_new_reqs[0].req_id == "r0"


# ---------------------------------------------------------------------------
# v7 phase machine: admission -> prefill chunks -> decode steps -> finalize
# ---------------------------------------------------------------------------


class _FakePool:
    """Minimal decode-buffer pool stub implementing the v7 §4.2 reservation API."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.reserved = 0
        self.reserve_calls: list[int] = []
        self.release_calls: list[int] = []

    def reserve(self, num_rows: int) -> bool:
        self.reserve_calls.append(num_rows)
        if self.reserved + num_rows > self.capacity:
            return False
        self.reserved += num_rows
        return True

    def release_reserved(self, num_rows: int) -> None:
        self.release_calls.append(num_rows)
        self.reserved -= num_rows


def test_reserve_unit_resources_reserves_decode_buffer() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    pool = _FakePool(capacity=100)
    cfg = BeamGroupingConfig(max_decode_steps=5)

    reservation = reserve_unit_resources(_scheduler(_FakeKV(), pool), group, cfg)

    assert reservation is not None
    # B=1 child x W=2 beams x 5 steps = 10 rows
    assert reservation.decode_rows == 10
    assert pool.reserved == 10


def test_reserve_unit_resources_rolls_back_pool_on_failure() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    kv = _FakeKV()
    pool = _FakePool(capacity=5)  # smaller than BxWxsteps = 10
    cfg = BeamGroupingConfig(max_decode_steps=5)

    reservation = reserve_unit_resources(_scheduler(kv, pool), group, cfg)

    assert reservation is None
    assert kv.freed == ["r0"]  # prompt block rolled back
    assert pool.reserved == 0  # pool never reserved on failure


def test_release_unit_resources_releases_decode_buffer() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    kv = _FakeKV()
    pool = _FakePool(capacity=100)
    cfg = BeamGroupingConfig(max_decode_steps=5)

    reservation = reserve_unit_resources(_scheduler(kv, pool), group, cfg)
    assert reservation is not None

    release_unit_resources(_scheduler(kv, pool), reservation)

    assert set(kv.freed) == {"r0"}
    assert pool.reserved == 0
    assert pool.release_calls == [10]


def test_schedule_admission_returns_single_prefill() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    scheduler = _scheduler(_FakeKV())
    cfg = BeamGroupingConfig()

    output, state = schedule(scheduler, None, config=cfg, dispatch_unit=group)

    assert state is not None
    assert state.phase == BeamPhase.PREFILL
    assert output.total_num_scheduled_tokens == 4


def test_schedule_prefill_then_decode_single_dispatch() -> None:
    """The two-dispatch model: prefill once, then decode once."""
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    reservation = UnitReservation(
        unit=group,
        blocks_by_child={"r0": _FakeBlocks(([7],))},
        num_computed_by_child={"r0": 0},
    )
    state = BeamGroupState(unit=group, reservation=reservation, phase=BeamPhase.PREFILL)
    scheduler = _scheduler(_FakeKV())
    cfg = BeamGroupingConfig()

    # PREFILL → single decode dispatch.
    output, new_state = schedule(scheduler, state, config=cfg)
    assert new_state is not None
    assert new_state.phase == BeamPhase.DECODE
    assert output.total_num_scheduled_tokens == 1  # one decode dispatch


def test_schedule_decode_dispatched_waits_for_final_result() -> None:
    """After the single decode dispatch, the scheduler emits empty output."""
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    reservation = UnitReservation(
        unit=group,
        blocks_by_child={"r0": _FakeBlocks(([7],))},
        num_computed_by_child={"r0": 0},
    )
    state = BeamGroupState(unit=group, reservation=reservation, phase=BeamPhase.DECODE)
    scheduler = _scheduler(_FakeKV())
    cfg = BeamGroupingConfig()

    output, new_state = schedule(scheduler, state, config=cfg)

    assert new_state is state
    assert new_state.phase == BeamPhase.DECODE
    assert output.total_num_scheduled_tokens == 0


def test_schedule_finalize_returns_empty() -> None:
    child = _child("r0", 4, 4)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    reservation = UnitReservation(
        unit=group,
        blocks_by_child={"r0": _FakeBlocks(([7],))},
        num_computed_by_child={"r0": 0},
    )
    state = BeamGroupState(unit=group, reservation=reservation, phase=BeamPhase.FINALIZE)
    scheduler = _scheduler(_FakeKV())
    cfg = BeamGroupingConfig()

    output, new_state = schedule(scheduler, state, config=cfg)

    assert new_state is state
    assert new_state.phase == BeamPhase.FINALIZE
    assert output.total_num_scheduled_tokens == 0


def test_schedule_no_dispatch_unit_returns_empty() -> None:
    scheduler = _scheduler(_FakeKV())
    cfg = BeamGroupingConfig()

    output, state = schedule(scheduler, None, config=cfg, dispatch_unit=None)

    assert state is None
    assert output.total_num_scheduled_tokens == 0


def test_schedule_full_two_dispatch_flow() -> None:
    """End-to-end two-dispatch flow: admit → prefill → decode → wait."""
    child = _child("r0", 8, 8)
    group = BeamBatchGroup(group_id="g0", members=(_member("b0", 2, (child,)),))
    scheduler = _scheduler(_FakeKV())
    cfg = BeamGroupingConfig()

    # 1. Admission: single full prefill (8 tokens, no chunking).
    output0, state = schedule(scheduler, None, config=cfg, dispatch_unit=group)
    assert state is not None
    assert state.phase == BeamPhase.PREFILL
    assert output0.total_num_scheduled_tokens == 8

    # 2. Single decode dispatch.
    output1, state = schedule(scheduler, state, config=cfg)
    assert state is not None
    assert state.phase == BeamPhase.DECODE
    assert output1.total_num_scheduled_tokens == 1

    # 3. Wait for the Worker's final result (empty output).
    output2, state = schedule(scheduler, state, config=cfg)
    assert state is not None
    assert state.phase == BeamPhase.DECODE
    assert output2.total_num_scheduled_tokens == 0
