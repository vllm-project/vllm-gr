# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler-side Beam Batch grouping and reservation data structures.

These are the thin orchestration objects described by the issue #39 Scheduler
framework.  They intentionally contain no vLLM scheduling logic: they describe
what a group is, what resources it reserved, and what phase it is in.  The
state-machine and native-function reuse live in :mod:`vllm_gr.v1.engine.beam_scheduler`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

    from vllm_gr.v1.beam.batch_contracts import BeamBatchPlan, BeamPrefillData
    from vllm_gr.v1.engine.beam_batch import BeamBatch


class BeamPhase(str, Enum):
    """Execution phase for one active beam group.

    ``PREFILL`` means the group has been admitted and its full-prefill
    scheduler output has been dispatched; on the next scheduler call it should
    advance to decode.  ``DECODE`` means the single decode dispatch has been
    sent; the Worker runs the full decode loop internally and the group awaits
    the final result.  ``FINALIZE`` marks the group as awaiting resource
    release.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    FINALIZE = "finalize"


@dataclass
class BeamBatchGroup:
    """One scheduler dispatch unit: a collection of business ``BeamBatch``."""

    group_id: str
    members: tuple[BeamBatch, ...]
    target_size: int | None = None

    def __post_init__(self) -> None:
        if not self.group_id or not self.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if not self.members:
            raise ValueError("members must contain at least one BeamBatch")
        if self.target_size is not None and self.target_size < 1:
            raise ValueError("target_size must be >= 1")

    @property
    def children(self) -> tuple[Request, ...]:
        """Flattened native child ``Request`` objects from all members."""
        return tuple(child for member in self.members for child in member.children)

    @property
    def batch_ids(self) -> tuple[str, ...]:
        """Stable business batch IDs, in member order."""
        return tuple(member.batch_id for member in self.members)


@dataclass
class UnitReservation:
    """Resource reservation result for one dispatch unit.

    Per the issue #39 design (§4.2 / §7.2) a reservation covers BOTH prompt KV
    paged blocks (``blocks_by_child``) and the decode buffer rows
    (``decode_rows`` == B × W × max_decode_steps).  ``hold_blocks`` marks the
    all-or-none reservation; prompt blocks stay occupied until the terminal
    release in FINALIZE.
    """

    unit: BeamBatchGroup
    blocks_by_child: dict[str, KVCacheBlocks]
    num_computed_by_child: dict[str, int]
    decode_rows: int = 0
    hold_blocks: bool = True

    def __post_init__(self) -> None:
        child_ids = {child.request_id for child in self.unit.children}
        if set(self.blocks_by_child) != child_ids:
            raise ValueError("blocks_by_child must contain exactly one entry per child")
        if set(self.num_computed_by_child) != child_ids:
            raise ValueError("num_computed_by_child must contain exactly one entry per child")
        if self.decode_rows < 0:
            raise ValueError("decode_rows must be >= 0")


@dataclass(frozen=True)
class BeamGroupingConfig:
    """Beam grouping and execution knobs."""

    group_target_sizes: tuple[int, ...] = (8, 4)
    length_buckets: tuple[int, ...] = (512, 1024, 2048, 4096)
    group_max_wait_ms: float = 10.0
    group_multi_batches: bool = False
    max_decode_steps: int = 3

    def __post_init__(self) -> None:
        if not self.group_target_sizes:
            raise ValueError("group_target_sizes must not be empty")
        if any(size < 1 for size in self.group_target_sizes):
            raise ValueError("group_target_sizes must be >= 1")
        if tuple(self.group_target_sizes) != tuple(sorted(self.group_target_sizes, reverse=True)):
            raise ValueError("group_target_sizes must be sorted descending")
        if not self.length_buckets:
            raise ValueError("length_buckets must not be empty")
        if any(bucket < 1 for bucket in self.length_buckets):
            raise ValueError("length_buckets must be >= 1")
        if tuple(self.length_buckets) != tuple(sorted(self.length_buckets)):
            raise ValueError("length_buckets must be sorted ascending")
        if self.group_max_wait_ms < 0:
            raise ValueError("group_max_wait_ms must be >= 0")
        if self.max_decode_steps < 1:
            raise ValueError("max_decode_steps must be >= 1")


@dataclass
class BeamGroupState:
    """Persistent EngineCore-side state for one active group (issue #39 §3 / §7).

    Carries only what the two-dispatch phase machine needs: the unit, its
    resource reservation, and the current phase.  The Worker owns the decode
    loop (select / top-k / filtering), so no per-step beam state lives here.
    """

    unit: BeamBatchGroup
    reservation: UnitReservation
    phase: BeamPhase = BeamPhase.PREFILL


@dataclass(frozen=True)
class BeamBatchGroupPlan:
    """Group-level worker contract assembled from existing member plans."""

    group_id: str
    members: tuple[BeamBatchPlan, ...]

    def __post_init__(self) -> None:
        if not self.group_id or not self.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if not self.members:
            raise ValueError("members must contain at least one BeamBatchPlan")

    @property
    def children(self) -> tuple[BeamPrefillData, ...]:
        """Flattened per-child prefill data from all member plans."""
        return tuple(item for plan in self.members for item in plan.items)


__all__ = [
    "BeamBatchGroup",
    "BeamBatchGroupPlan",
    "BeamGroupState",
    "BeamGroupingConfig",
    "BeamPhase",
    "UnitReservation",
]
