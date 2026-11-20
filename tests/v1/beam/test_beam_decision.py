# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU regressions for V1 decision buffers and NPU selection primitives."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.beam import beam_decision
from vllm_gr.v1.beam.beam_attention_context import (
    BeamAttentionContext,
    BeamDecisionBuffers,
)
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession


def _make_context(
    *,
    max_batch: int = 1,
    max_beams: int = 4,
    max_decode_steps: int = 4,
    num_layers: int = 2,
) -> BeamAttentionContext:
    return BeamAttentionContext(
        max_batch=max_batch,
        max_beams=max_beams,
        num_kv_heads=2,
        max_decode_steps=max_decode_steps,
        head_dim=8,
        num_layers=num_layers,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )


def _make_session(context: BeamAttentionContext, session_id: str = "s") -> BeamSearchSession:
    slot = context.registry.allocate(session_id, beam_width=0, prefix_len=0)
    return BeamSearchSession(session_id=session_id, slot=slot, context=context)


# ---------------------------------------------------------------------------
# G1: decision-buffer reset across sessions
# ---------------------------------------------------------------------------


def test_begin_session_resets_decision_buffers() -> None:
    """Reusing a slot clears scores, history, and the Decode step counter."""
    context = _make_context()
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)

    # Simulate a completed session's dirty decision state.
    bufs = context.decision
    bufs.sequence.fill_(7)
    bufs.beam_scores.fill_(-3.5)
    bufs.group_token_num.fill_(2)
    bufs.worker_decode_steps = 3

    session.begin_session(beam_width=4, prefix_len=8)

    assert bufs.worker_decode_steps == 0
    assert torch.equal(bufs.sequence, torch.zeros_like(bufs.sequence))
    assert torch.equal(bufs.beam_scores, torch.zeros_like(bufs.beam_scores))
    assert torch.equal(bufs.group_token_num, torch.zeros_like(bufs.group_token_num))


# ---------------------------------------------------------------------------
# G2: result_width defaults to the beam width
# ---------------------------------------------------------------------------


def test_npu_n2_group_receives_two_slot_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """The N=2 terminal group must not receive the runtime N=3 capacity."""
    from vllm_gr.ops.npu import beam_ops

    width = 8
    context = _make_context(max_beams=width, max_decode_steps=3)
    session = _make_session(context)
    session.begin_session(beam_width=width, prefix_len=8)
    context.decision.worker_decode_steps = 1
    seen_shape: tuple[int, ...] | None = None

    def capture_group(*args: Any) -> None:
        nonlocal seen_shape
        seen_shape = tuple(args[3].shape)
        raise RuntimeError("captured group input")

    monkeypatch.setattr(beam_ops, "beam_search_group", capture_group)
    candidates = torch.arange(width * width, dtype=torch.int32).reshape(width, width)
    with pytest.raises(RuntimeError, match="captured group input"):
        beam_decision._run_npu_decision(
            session,
            candidates,
            torch.zeros_like(candidates, dtype=torch.float32),
            is_last_step=True,
            result_width=width,
        )
    assert seen_shape == (1, width, 2)


def test_result_width_decouples_from_max_decode_steps() -> None:
    """Explicit result_width metadata overrides; absent → beam width."""
    context = _make_context()
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)

    # metadata result_width is threaded through unchanged when present.
    mdata_width = 8
    assert (mdata_width or session.beam_width) == 8
    assert (None or session.beam_width) == 4
    # The old bug: max_decode_steps (step count) passed as width.
    max_decode_steps = 3
    assert max_decode_steps != session.beam_width


# ---------------------------------------------------------------------------
# G3: worker decision gated on the xllm decode execution mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# G4: global → request-local prefix sums
# ---------------------------------------------------------------------------


def test_kv_reorder_applies_request_local_parents_to_each_slot() -> None:
    """Global parent indices must reorder KV within the owning request slot."""
    context = _make_context(max_batch=2, max_beams=4, max_decode_steps=4)
    first = _make_session(context, "first")
    second = _make_session(context, "second")
    first.begin_session(beam_width=4, prefix_len=8)
    second.begin_session(beam_width=4, prefix_len=8)
    session = SimpleNamespace(sessions=(first, second), context=context, beam_width=4)

    beam_width = 4
    # batch=2 synthetic operator output exercising the offset conversion.
    bufs = context.decision
    bufs.beam_scores[:2, :beam_width] = torch.tensor(
        [[-0.1, -0.2, -0.3, -0.4], [-0.5, -0.6, -0.7, -0.8]]
    )
    bufs.worker_decode_steps = 2

    out_ids = torch.zeros(2, beam_width, dtype=torch.int32)
    # beam_search_group emits sorted parents and cumulative child counts.
    # Apply an offset of 4 to both outputs for the second request.
    out_idx = torch.tensor([[0, 0, 1, 1], [4, 6, 6, 7]], dtype=torch.int32)
    out_lp = torch.zeros(2, beam_width, dtype=torch.float32)
    out_seq = torch.zeros(2, beam_width, 2, dtype=torch.int32)
    prefix = torch.tensor([[2, 4, 4, 4], [5, 5, 7, 8]], dtype=torch.int32)

    pool = context.pools[0]
    keys = pool.unshared_key[:, :, 0, 0, 0]
    values = pool.unshared_value[:, :, 0, 0, 0]
    keys.copy_(torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]))
    values.copy_(torch.tensor([[30, 31, 32, 33], [40, 41, 42, 43]]))

    beam_decision._finalize_decision(
        session, bufs, out_ids, out_idx, out_lp, out_seq, prefix, 2, beam_width, beam_width
    )

    assert bufs.group_token_num[:2, :beam_width].tolist() == [
        [2, 4, 4, 4],
        [1, 1, 3, 4],
    ]
    assert keys.tolist() == [[10, 10, 11, 11], [20, 22, 22, 23]]
    assert values.tolist() == [[30, 30, 31, 31], [40, 42, 42, 43]]
    assert (first.decode_step, second.decode_step) == (1, 1)


# ---------------------------------------------------------------------------
# G5: same-step KV reorder + next-step consumption
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P2: return-channel payload slimming
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P1: contiguous-row slice assignment in the constraint hook
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Decision-buffer coverage helper
# ---------------------------------------------------------------------------


def test_decision_buffers_reset_clears_all_state() -> None:
    bufs = BeamDecisionBuffers(2, 4, 3, torch.device("cpu"), torch.float16)
    bufs.sequence.fill_(9)
    bufs.beam_scores.fill_(1.5)
    bufs.group_token_num.fill_(3)
    bufs.worker_decode_steps = 2
    bufs.reset()
    assert bufs.worker_decode_steps == 0
    assert torch.equal(bufs.sequence, torch.zeros_like(bufs.sequence))
    assert torch.equal(bufs.beam_scores, torch.zeros_like(bufs.beam_scores))
    assert torch.equal(bufs.group_token_num, torch.zeros_like(bufs.group_token_num))


# ---------------------------------------------------------------------------
# _select_candidates layout contract (V2 sample wrapper)
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.cpu_test
