# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BeamSessionRegistry — per-request session registry with slot allocation.

Manages the mapping from session_id to pool slot, and holds per-request
state (decode_step, beam_width, prefix_len, is_last_step, is_active).

Slot allocation uses a free-list strategy: released slots are recycled,
enabling multiple requests to rotate through a fixed pool of slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SessionSlotState:
    """Per-request state stored in the registry.

    All fields are host-side values. decode_step is a plain int (not a device
    tensor) because every consumer already calls .item() to transfer to host.
    """

    session_id: str
    slot: int
    decode_step: int = 0  # 0-indexed, host-side counter
    beam_width: int = 0
    prefix_len: int = 0
    is_last_step: bool = False
    is_active: bool = False
    # Worker-side same-step KV reorder record.
    # ``kv_reorder`` holds the request-local parent mapping the worker's beam
    # decision already applied to the KV pool inside the *same* execute_model
    # step (``beam_decision.run_beam_decision`` → ``select_unshared_kv``).
    # It is derived in ``_maybe_run_worker_beam_decision`` from the return
    # channel's host copy of ``out_token_index`` (its request-0 row IS the
    # request-local mapping), so no extra device→host sync is spent on it.
    # The next step's ``_get_or_create_beam_session`` consumes it: an exact
    # match against the frontend-returned ``parent_beam_ids`` skips the
    # duplicate reorder; a mismatch triggers the composite reorder instead.
    # ``None`` means no same-step reorder happened (frontend-driven path).
    kv_reorder: list[int] | None = None


class BeamSessionRegistry:
    """Per-request session registry with slot allocation.

    Responsibilities:
    - Allocate/release pool slots for beam search sessions.
    - Map session_id → slot (for routing KV reads/writes to the correct pool row).
    - Store per-request state (decode_step, beam_width, etc.) in SessionSlotState.

    The registry is owned by BeamAttentionContext and shared across all
    BeamSearchSession instances.
    """

    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self._slot_map: Dict[str, int] = {}  # session_id → slot
        self._free_slots: List[int] = list(range(max_batch))  # free-list
        self._slots: Dict[int, SessionSlotState] = {}  # slot → state

    @property
    def num_active_sessions(self) -> int:
        return len(self._slot_map)

    def allocate(self, session_id: str, beam_width: int, prefix_len: int) -> int:
        """Allocate a slot for a new session.

        Returns the slot index. Raises RuntimeError if no free slots.
        """
        if not self._free_slots:
            raise RuntimeError(
                f"No free beam session slots (max_batch={self.max_batch}). "
                f"Active sessions: {list(self._slot_map.keys())}"
            )
        slot = self._free_slots.pop(0)
        self._slot_map[session_id] = slot
        self._slots[slot] = SessionSlotState(
            session_id=session_id,
            slot=slot,
            decode_step=0,
            beam_width=beam_width,
            prefix_len=prefix_len,
            is_last_step=False,
            is_active=True,
        )
        return slot

    def release(self, session_id: str) -> None:
        """Release a session's slot back to the free list."""
        slot = self._slot_map.pop(session_id, None)
        if slot is not None:
            del self._slots[slot]
            self._free_slots.append(slot)

    def get_slot(self, session_id: str) -> int:
        """Get the slot index for a session. Raises KeyError if not found."""
        return self._slot_map[session_id]

    def get_state(self, session_id: str) -> SessionSlotState:
        """Get the per-request state for a session."""
        return self._slots[self._slot_map[session_id]]

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists in the registry."""
        return session_id in self._slot_map
