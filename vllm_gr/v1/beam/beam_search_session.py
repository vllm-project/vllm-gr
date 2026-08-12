# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BeamSearchSession — per-request facade for incremental beam decode.

Manages the full lifecycle of a beam search request.  References a shared
BeamAttentionContext and routes all KV operations through its own slot.

All per-request state (decode_step, beam_width, prefix_len, is_last_step)
is stored in BeamSessionRegistry.SessionSlotState, accessed via
context.registry.get_state(session_id).

Proxy properties (pools, decode_step, beam_width, etc.) forward to the
shared context + registry state, keeping the external interface unchanged
for callers (cascade_attention, beam_attn, etc.).
"""

from __future__ import annotations

from typing import List

import torch

from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool
from vllm_gr.v1.beam.beam_session_registry import SessionSlotState


class BeamSearchSession:
    """Per-request facade for incremental beam decode.

    Responsibilities:
    - Manage the full lifecycle of a beam search request.
    - Route KV operations through its own slot in the shared context.
    - Encapsulate step_complete (select_unshared_kv + advance_step).
    """

    def __init__(
        self,
        session_id: str,
        slot: int,
        context: BeamAttentionContext,
    ):
        self.session_id = session_id
        self.slot = slot
        self.context = context  # shared ref, not owned

    def begin_session(self, beam_width: int, prefix_len: int):
        """Start a beam search session. Resets slot state + pool KV."""
        state = self._state
        state.decode_step = 0
        state.beam_width = beam_width
        state.prefix_len = prefix_len
        state.is_last_step = False
        state.is_active = True
        # Reset pool KV for this slot (clear stale data from previous sessions)
        for pool in self.context.pools:
            pool.reset_slot(self.slot)

    def end_session(self):
        """End a beam search session. Releases slot from registry."""
        state = self._state
        state.is_active = False
        self.context.registry.release(self.session_id)

    def step_complete(self, parent: torch.Tensor):
        """Called after beam decision: select_unshared_kv + advance_step.

        On the last step, skip select_unshared_kv (no need to reorder KV
        because there is no next step).
        """
        state = self._state
        if not state.is_last_step:
            from vllm_gr.v1.attention.backends.beam_kernels import select_unshared_kv

            select_unshared_kv(
                parent,
                self.slot,
                [p.unshared_key for p in self.context.pools],
                [p.unshared_value for p in self.context.pools],
                state.decode_step,
            )
        state.decode_step += 1

    # --- Internal helpers ---

    @property
    def _state(self) -> SessionSlotState:
        """Get the per-request state from the registry."""
        return self.context.registry.get_state(self.session_id)

    # --- Proxy properties for BeamAttentionContext ---
    # Forward accesses session.pools / session.decode_step etc.,
    # internally forwarded to context + registry state.

    @property
    def is_active(self) -> bool:
        return self._state.is_active

    @property
    def beam_width(self) -> int:
        return self._state.beam_width

    @property
    def decode_step(self) -> int:
        """Host-side int (not a device tensor)."""
        return self._state.decode_step

    @property
    def pools(self) -> List[BeamAttentionPool]:
        return self.context.pools

    @property
    def suffix_k_buf(self) -> torch.Tensor:
        k_buf, _ = self.context.get_suffix_buf(self.slot, self.beam_width)
        return k_buf

    @property
    def suffix_v_buf(self) -> torch.Tensor:
        _, v_buf = self.context.get_suffix_buf(self.slot, self.beam_width)
        return v_buf
