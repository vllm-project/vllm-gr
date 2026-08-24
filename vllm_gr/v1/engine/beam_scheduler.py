# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler-side Beam Batch orchestration helpers.

The functions in this module are intentionally dependency-light: they take the
owning ``Scheduler`` (or the small pieces of it they need) and the group state,
then reuse vLLM native KV/request/plan helpers.  They are the issue #39
state-machine implementation points and can be unit-tested without starting an
EngineCore.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from vllm_gr.v1.engine.beam_group import (
    BeamBatchGroup,
    BeamGroupingConfig,
    BeamGroupState,
    BeamPhase,
    UnitReservation,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


def _child_beam_width(group: BeamBatchGroup, child: Request) -> int:
    """Return the beam width associated with one child Request."""
    for member in group.members:
        if child in member.children:
            return member.request.beam_params.beam_width
    raise ValueError(f"child {child.request_id!r} is not a member of group {group.group_id!r}")


def _child_prefix_len(child: Request) -> int:
    """Prompt prefix length used to size one decode slot."""
    return getattr(child, "num_prompt_tokens", getattr(child, "num_tokens", 0))


def reserve_unit_resources(
    scheduler,
    unit: BeamBatchGroup,
    config: BeamGroupingConfig | None = None,
) -> UnitReservation | None:
    """Reserve full prompt KV blocks + decode buffer rows for a group, all-or-none.

    This is the thin, rollback-safe wrapper around native ``get_computed_blocks``,
    ``allocate_slots``, and ``free`` (issue #39 §8).  On success the reservation
    covers BOTH prompt blocks and ``decode_rows`` == B × W × max_decode_steps
    decode buffer rows from ``scheduler.beam_pool`` (when present).  Any failure
    rolls back everything already allocated and returns ``None``.
    """
    kv = scheduler.kv_cache_manager

    blocks: dict[str, KVCacheBlocks] = {}
    num_computed: dict[str, int] = {}
    for child in unit.children:
        computed, n = kv.get_computed_blocks(child)
        allocated = kv.allocate_slots(
            child,
            child.num_tokens - n,
            num_new_computed_tokens=n,
            new_computed_blocks=computed,
            full_sequence_must_fit=True,
        )
        if allocated is None:
            for prev_child in unit.children:
                if prev_child.request_id in blocks:
                    kv.free(prev_child)
            return None
        blocks[child.request_id] = allocated
        num_computed[child.request_id] = n

    # Decode buffer reservation: B × W × max_steps rows (issue #39 §4.2).
    cfg = config or BeamGroupingConfig()
    beam_pool = getattr(scheduler, "beam_pool", None)
    decode_rows = 0
    if beam_pool is not None and hasattr(beam_pool, "reserve"):
        beam_width = max(_child_beam_width(unit, c) for c in unit.children)
        decode_rows = len(unit.children) * beam_width * cfg.max_decode_steps
        if not beam_pool.reserve(decode_rows):
            for prev_child in unit.children:
                if prev_child.request_id in blocks:
                    kv.free(prev_child)
            return None

    return UnitReservation(
        unit=unit,
        blocks_by_child=blocks,
        num_computed_by_child=num_computed,
        decode_rows=decode_rows,
        hold_blocks=True,
    )


def release_unit_resources(scheduler, reservation: UnitReservation) -> None:
    """Release prompt KV blocks + decode buffer rows for one reservation (issue #39 §8)."""
    kv = scheduler.kv_cache_manager
    for child in reservation.unit.children:
        kv.free(child)
    if reservation.decode_rows > 0:
        beam_pool = getattr(scheduler, "beam_pool", None)
        if beam_pool is not None and hasattr(beam_pool, "release_reserved"):
            beam_pool.release_reserved(reservation.decode_rows)


def build_prefill_output(reservation: UnitReservation) -> SchedulerOutput:
    """Build the single full-prefill SchedulerOutput from a reservation.

    Prefill is dispatched once over the whole remaining prompt (no chunking).
    """
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

    output = SchedulerOutput.make_empty()
    scheduled_new_reqs = []
    num_scheduled_tokens: dict[str, int] = {}
    for child in reservation.unit.children:
        block_ids = reservation.blocks_by_child[child.request_id].get_block_ids()
        scheduled_new_reqs.append(NewRequestData.from_request(child, block_ids))
        num_scheduled_tokens[child.request_id] = (
            child.num_tokens - reservation.num_computed_by_child[child.request_id]
        )

    output.scheduled_new_reqs = scheduled_new_reqs
    output.num_scheduled_tokens = num_scheduled_tokens
    output.total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    return output


def build_decode_output(
    state: BeamGroupState,
    *,
    max_decode_steps: int | None = None,
) -> SchedulerOutput:
    """Build the single decode dispatch for a group.

    Decode is dispatched exactly once; the Worker owns the full decode loop
    (select / top-k / valid-token filtering are executed inside the Worker) and
    returns the final result.  The paged prompt blocks stay occupied until
    EngineCore releases the reservation at terminal time.
    """
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    children = state.unit.children
    req_ids = [child.request_id for child in children]

    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in children],
        all_token_ids={},
        new_block_ids=[None for _ in children],
        num_computed_tokens=[_child_prefix_len(child) for child in children],
        num_output_tokens=[1 for _ in children],
    )
    output.num_scheduled_tokens = dict(zip(req_ids, [1] * len(children)))
    output.total_num_scheduled_tokens = len(children)
    output.beam_data = build_beam_group_decode_metadata(
        state, max_decode_steps=max_decode_steps
    )
    return output


def build_beam_group_decode_metadata(
    state: BeamGroupState,
    *,
    max_decode_steps: int | None = None,
) -> dict[str, dict]:
    """Build the per-child ``beam_data`` payload for the single decode dispatch.

    The payload carries the flag that tells the native scheduler/Worker not to
    reclaim the paged blocks for these requests until EngineCore explicitly
    releases the reservation at terminal time.  ``max_decode_steps`` is the
    authoritative decode-loop bound (from ``GRConfig.beam_max_decode_steps``);
    when absent the Worker falls back to its own GR config value.
    """
    metadata: dict[str, dict] = {}
    for child in state.unit.children:
        entry: dict = {
            "is_beam_decode": True,
            "is_beam_decode_run": True,  # hold blocks until EngineCore terminal release
            "beam_width": _child_beam_width(state.unit, child),
            "prefix_len": _child_prefix_len(child),
            "parent_beam_ids": None,
        }
        if max_decode_steps is not None:
            entry["max_decode_steps"] = max_decode_steps
        metadata[child.request_id] = entry
    return metadata


def try_admit_group(
    scheduler,
    group: BeamBatchGroup,
    config: BeamGroupingConfig | None = None,
) -> tuple[SchedulerOutput, BeamGroupState] | None:
    """Reserve resources and build the single prefill dispatch for a group.

    Returns ``None`` when the group cannot be admitted all-or-none and the
    reservation has been rolled back.  On success the returned state has phase
    ``PREFILL``.
    """
    reservation = reserve_unit_resources(scheduler, group, config)
    if reservation is None:
        return None
    state = BeamGroupState(unit=group, reservation=reservation, phase=BeamPhase.PREFILL)
    return build_prefill_output(reservation), state


def _beam_grouping_lock(scheduler):
    """Return the scheduler-scoped lock protecting beam queues."""
    import threading

    lock = getattr(scheduler, "beam_grouping_lock", None)
    if lock is None:
        lock = threading.Lock()
        scheduler.beam_grouping_lock = lock
    return lock


def _ensure_beam_queues(scheduler) -> None:
    """Initialize the bucket waiting queue and the prepared dispatch queue."""
    if getattr(scheduler, "beam_waiting", None) is None:
        scheduler.beam_waiting = OrderedDict()
    if getattr(scheduler, "_prepared_dispatch_queue", None) is None:
        scheduler._prepared_dispatch_queue = []


def batch_child_count(batch) -> int:
    """Number of native child requests in a business batch."""
    return len(batch.children)


def batch_max_prompt_len(batch) -> int:
    """Max prompt length across a business batch's children."""
    return max(
        getattr(child, "num_prompt_tokens", getattr(child, "num_tokens", 0))
        for child in batch.children
    )


def _bucket_for_batch(batch, config: BeamGroupingConfig) -> int:
    """Return the first length bucket that can hold this batch."""
    length = batch_max_prompt_len(batch)
    for bucket in config.length_buckets:
        if length <= bucket:
            return bucket
    raise ValueError(
        f"batch prompt length {length} exceeds max bucket {config.length_buckets[-1]}"
    )


def _make_group(members: list, target_size: int) -> BeamBatchGroup:
    from uuid import uuid4

    if len(members) == 1:
        group_id = members[0].batch_id
    else:
        group_id = f"beam-group-{uuid4().hex}"
    return BeamBatchGroup(
        group_id=group_id,
        members=tuple(members),
        target_size=target_size,
    )


def _form_ready_groups_locked(scheduler, config: BeamGroupingConfig) -> None:
    """Form full or timed-out groups into the prepared dispatch queue.

    Caller must hold ``scheduler.beam_grouping_lock``.
    """
    beam_waiting: OrderedDict = scheduler.beam_waiting
    queue: list = scheduler._prepared_dispatch_queue
    primary = config.group_target_sizes[0]
    secondary = config.group_target_sizes[1] if len(config.group_target_sizes) > 1 else primary
    now = time.monotonic()

    for bucket in list(beam_waiting.keys()):
        entries = beam_waiting[bucket]
        while entries:
            # Greedy first-fit: walk the chain in FIFO order and take every
            # batch that still fits; skip (but keep) ones that would overflow.
            total = 0
            members: list = []
            kept: list = []
            for entry in entries:
                size = batch_child_count(entry[0])
                if total + size <= primary:
                    total += size
                    members.append(entry[0])
                else:
                    kept.append(entry)

            if total == primary:
                beam_waiting[bucket] = kept
                queue.append(_make_group(members, primary))
                entries = kept
                continue

            # Not full.  Force-dispatch only when the head has waited long enough.
            if now - entries[0][1] >= config.group_max_wait_ms:
                beam_waiting[bucket] = kept
                target = primary if total > secondary else secondary
                queue.append(_make_group(members, target))
                entries = kept
                continue

            break

        if not entries:
            del beam_waiting[bucket]


def refresh_prepared_groups(
    scheduler,
    config: BeamGroupingConfig | None = None,
) -> None:
    """Re-evaluate waiting buckets and move ready groups to the dispatch queue."""
    cfg = config or BeamGroupingConfig()
    lock = _beam_grouping_lock(scheduler)
    with lock:
        _ensure_beam_queues(scheduler)
        _form_ready_groups_locked(scheduler, cfg)


def enqueue_beam_batch_and_prepare(
    scheduler,
    batch,
    config: BeamGroupingConfig | None = None,
) -> None:
    """Enqueue one batch into a length bucket (or direct) and form ready groups."""
    cfg = config or BeamGroupingConfig()
    lock = _beam_grouping_lock(scheduler)
    with lock:
        _ensure_beam_queues(scheduler)
        if not cfg.group_multi_batches:
            scheduler._prepared_dispatch_queue.append(
                _make_group([batch], batch_child_count(batch))
            )
            return
        bucket = _bucket_for_batch(batch, cfg)
        scheduler.beam_waiting.setdefault(bucket, []).append(
            (batch, time.monotonic())
        )
        _form_ready_groups_locked(scheduler, cfg)


def peek_prepared_dispatch_group(scheduler) -> BeamBatchGroup | None:
    """Return the head of the prepared dispatch queue without removing it."""
    lock = _beam_grouping_lock(scheduler)
    with lock:
        queue = getattr(scheduler, "_prepared_dispatch_queue", None)
        return queue[0] if queue else None


def pop_prepared_dispatch_group(
    scheduler,
    unit: BeamBatchGroup,
) -> None:
    """Remove the head group after admission succeeded."""
    lock = _beam_grouping_lock(scheduler)
    with lock:
        queue = getattr(scheduler, "_prepared_dispatch_queue", None)
        if queue and queue[0].group_id == unit.group_id:
            queue.pop(0)


def remove_beam_batch(
    scheduler,
    batch_id: str,
) -> None:
    """Remove one batch from bucket waiting and any prepared dispatch group."""
    lock = _beam_grouping_lock(scheduler)
    with lock:
        beam_waiting = getattr(scheduler, "beam_waiting", None)
        if beam_waiting is not None:
            for bucket in list(beam_waiting.keys()):
                entries = beam_waiting[bucket]
                filtered = [(b, t) for (b, t) in entries if b.batch_id != batch_id]
                if filtered:
                    beam_waiting[bucket] = filtered
                else:
                    del beam_waiting[bucket]
        queue = getattr(scheduler, "_prepared_dispatch_queue", None)
        if queue is not None:
            scheduler._prepared_dispatch_queue = [
                group for group in queue if batch_id not in group.batch_ids
            ]


def schedule(
    scheduler,
    active: BeamGroupState | None,
    *,
    config: BeamGroupingConfig | None = None,
    dispatch_unit: BeamBatchGroup | None = None,
) -> tuple[SchedulerOutput, BeamGroupState | None]:
    """Two-dispatch phase machine (issue #39 §8): prefill once, then decode once.

    ``dispatch_unit`` is the next waiting group to admit (``None`` when no
    waiting group is available).  Phase transitions:

    - ``active is None``: admit ``dispatch_unit`` (if any) → single prefill;
    - ``phase == PREFILL``: prefill done → single decode dispatch;
    - ``phase == DECODE``: decode dispatched; wait for the Worker's final
      result (empty output);
    - ``phase == FINALIZE``: empty output (resource release is driven by
      EngineCore, not by the scheduler).
    """
    from vllm.v1.core.sched.output import SchedulerOutput

    if active is None:
        if dispatch_unit is None:
            return SchedulerOutput.make_empty(), None
        return try_admit_group(scheduler, dispatch_unit, config) or (
            SchedulerOutput.make_empty(),
            None,
        )
    if active.phase == BeamPhase.PREFILL:
        active.phase = BeamPhase.DECODE
        max_decode_steps = (config or BeamGroupingConfig()).max_decode_steps
        return build_decode_output(active, max_decode_steps=max_decode_steps), active
    # DECODE / FINALIZE: nothing to dispatch; EngineCore performs the release.
    return SchedulerOutput.make_empty(), active


__all__ = [
    "batch_child_count",
    "batch_max_prompt_len",
    "build_beam_group_decode_metadata",
    "build_decode_output",
    "build_prefill_output",
    "enqueue_beam_batch_and_prepare",
    "peek_prepared_dispatch_group",
    "pop_prepared_dispatch_group",
    "refresh_prepared_groups",
    "release_unit_resources",
    "remove_beam_batch",
    "reserve_unit_resources",
    "schedule",
    "try_admit_group",
]
