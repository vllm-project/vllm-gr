# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
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

    def begin_session(
        self,
        beam_width: int,
        prefix_len: int,
        *,
        decision_index: int | None = None,
        reset_decision: bool = True,
    ):
        """Start a beam search session. Resets slot state (lazy KV reset).

        ``decision_index`` selects a physical decision row for the independent
        CUDA runner; it need not equal the KV slot. Omission preserves the
        legacy full-buffer reset. Persistent state must be reset separately,
        after final consumers, before its next Prefill initialization.

        ``reset_decision=False`` is for callers that just reset this row through
        ``BeamSearchRunner.reset_for_session`` (or its batched form): that
        launch already writes the same zeros to ``sequence``, ``beam_scores``
        and ``group_token_num``, so repeating it per member only adds launches.
        It needs an explicit ``decision_index``; the full-buffer reset also
        clears the legacy host counter, which a row reset leaves alone.

        Pool KV is NOT zeroed here. ``cache_unshared_kv_kernel`` is an exact
        overwrite (writes only ``pool[slot, beam, head, decode_step, :]`` per
        step), and cascade attention limits suffix reads to
        ``[:decode_step+1]`` via ``suffix_seqlens_kv`` (built in
        ``beam_attn_metadata.py``). A new session starts at ``decode_step=0``
        and overwrites forward; stale rows beyond ``decode_step`` are never
        read, so zeroing is redundant. ``select_unshared_kv`` also restricts
        its gather→scatter to ``[:decode_step+1]``, so aux buffers stay clean.
        """
        state = self._state
        if not reset_decision and decision_index is None:
            raise ValueError("reset_decision=False requires an explicit decision_index")
        state.decode_step = 0
        state.beam_width = beam_width
        state.prefix_len = prefix_len
        state.is_last_step = False
        state.is_active = True
        state.kv_reorder = None
        # Reset the device-resident Beam decision state. Otherwise a second
        # session reusing the slot inherits the previous session's
        # ``sequence`` / ``beam_scores`` / ``worker_decode_steps``, and
        # ``init_beam_decision_state`` is silently skipped because
        # ``worker_decode_steps != 0`` — every subsequent decision then reads
        # a corrupted cumulative state.
        decision = getattr(self.context, "decision", None)
        if decision is not None and reset_decision:
            if decision_index is None:
                decision.reset()
            else:
                decision.reset(decision_index)

    def end_session(self):
        """End a beam search session. Releases slot from registry."""
        state = self._state
        state.is_active = False
        self.context.registry.release(self.session_id)

    def step_complete(self, parent: torch.Tensor, *, layout_prevalidated: bool = False):
        """Called after beam decision: select_unshared_kv + advance_step.

        On the last step, skip select_unshared_kv (no need to reorder KV
        because there is no next step).
        """
        state = self._state
        if not state.is_last_step:
            from vllm_gr.v1.beam.reorder import reorder_unshared_kv

            # Hand the context's PER-LAYER [num_layers, ...] aux buffers to the
            # reorder dispatcher.  In cascade mode the fused Triton path uses
            # the whole tensor for its single all-layer launch; the per-layer
            # reference slices it with [layer]; xllm reorders in place and
            # ignores aux.
            aux_key = getattr(self.context, "aux_key_buf", None)
            aux_value = getattr(self.context, "aux_value_buf", None)
            pool_key_ptrs = getattr(self.context, "pool_key_ptrs", None)
            pool_value_ptrs = getattr(self.context, "pool_value_ptrs", None)

            reorder_unshared_kv(
                parent,
                self.slot,
                self.context.pools,
                state.decode_step,
                aux_key=aux_key,
                aux_value=aux_value,
                pool_key_ptrs=pool_key_ptrs,
                pool_value_ptrs=pool_value_ptrs,
                layout_prevalidated=layout_prevalidated,
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
    def pools(self) -> list[BeamAttentionPool]:
        return self.context.pools

    @property
    def suffix_k_buf(self) -> torch.Tensor:
        k_buf, _ = self.context.get_suffix_buf(self.slot, self.beam_width)
        return k_buf

    @property
    def suffix_v_buf(self) -> torch.Tensor:
        _, v_buf = self.context.get_suffix_buf(self.slot, self.beam_width)
        return v_buf
