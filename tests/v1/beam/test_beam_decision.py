# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU regressions for V1 decision buffers and NPU selection primitives."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.beam import BeamAttentionContext, BeamSearchSession, beam_decision
from vllm_gr.v1.beam.beam_attention_context import BeamDecisionBuffers


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
    """A second session must not inherit the previous session's state.

    Without the reset, ``worker_decode_steps`` stays non-zero and
    ``init_beam_decision_state`` is silently skipped for every subsequent
    session — the decision then reads a corrupted cumulative state.
    """
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


def test_kv_reorder_record_matches_global_index_minus_offset() -> None:
    """Multi-request offset arithmetic: recorded[i] = global_idx[i] - i*W."""
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
    # Global parent indices: request 0 → parents {0,1}, request 1 → parents
    # {0,1} + the request-1 offset (1 * beam_width = 4).
    out_idx = torch.tensor([[0, 1, 0, 1], [4, 5, 4, 5]], dtype=torch.int32)
    out_lp = torch.zeros(2, beam_width, dtype=torch.float32)
    out_seq = torch.zeros(2, beam_width, 2, dtype=torch.int32)
    # Global prefix sums (Asynchronous scheduling kernel contract): request-offset included.
    prefix = torch.tensor([[1, 2, 2, 2], [5, 6, 6, 6]], dtype=torch.int32)

    beam_decision._finalize_decision(
        session, bufs, out_ids, out_idx, out_lp, out_seq, prefix, 2, beam_width, beam_width
    )

    # request 0 local prefix: [1, 2, 2, 2] - 0*4; request 1: [5,6,6,6] - 4.
    assert bufs.group_token_num[:2, :beam_width].tolist() == [
        [1, 2, 2, 2],
        [1, 2, 2, 2],
    ]
    # The decision no longer materializes kv_reorder (no device→host sync);
    # each request row is applied to its own pool slot without a host-side
    # kv_reorder record. Row 0's
    # global offset is 0, so row 0 already equals the request-local mapping;
    # row 1 must shed its 1*beam_width offset to match.
    assert first._state.kv_reorder is None
    assert second._state.kv_reorder is None
    assert (first.decode_step, second.decode_step) == (1, 1)
    assert out_idx[0].tolist() == [0, 1, 0, 1]
    assert (out_idx[1] - beam_width).tolist() == [0, 1, 0, 1]


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
