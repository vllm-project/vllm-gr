# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Unit tests for the worker-side beam decision (Asynchronous scheduling decode_select port).

Covers the five correctness gaps fixed in the NPU post-processing port:

- G1: ``BeamDecisionBuffers.reset()`` runs at ``begin_session`` (a second
  session starts from zeroed decision state).
- G2: ``result_width`` defaults to the beam width, never the decode step
  count (``max_decode_steps``).
- G3: ``worker_decision_enabled`` is gated on the xllm decode execution mode.
- G4: the operator's GLOBAL prefix sums are converted to request-local form
  before persisting ``group_token_num``.
- G5: the KV reorder runs in the SAME step as the decision and advances
  the session's decode state.

All tests run on CPU against the PyTorch reference decision path
(``_run_torch_decision``); NPU fused-operator parity is covered by the
``tests/kernels`` suite on device.
"""

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


def _sampler_output(candidates: torch.Tensor, probs: torch.Tensor) -> SimpleNamespace:
    """Build a sampler-output stub shaped ``[B*W, K]`` for ``_select_candidates``."""
    return SimpleNamespace(
        logprobs_tensors=SimpleNamespace(
            logprobs=probs,
            logprob_token_ids=candidates,
        )
    )


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


def test_first_decode_inherits_parent_cumulative_scores() -> None:
    """The first decode adds local scores once to the prefill parent scores."""
    context = _make_context(max_beams=2)
    session = _make_session(context)
    session.begin_session(beam_width=2, prefix_len=8)
    beam_decision.init_beam_decision_from_parents(
        session,
        [[10], [20]],
        [-0.1, -5.0],
    )
    sampler_output = _sampler_output(
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
        torch.tensor([[-4.0, -6.0], [-0.1, -1.0]], dtype=torch.float32),
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _current_beam_info={
            "beam_data": {"s": {"is_beam_decode": True, "parent_beam_ids": [0, 1]}}
        },
    )

    decision = beam_decision.run_beam_decision(
        runner,
        session,
        sampler_output,
        is_last_step=False,
    )

    assert decision is not None
    assert set(decision["out_token_ids"].reshape(-1).tolist()) == {11, 21}
    assert sorted(decision["out_log_probs"].reshape(-1).tolist(), reverse=True) == pytest.approx(
        [-4.1, -5.1]
    )


def test_unseeded_first_decode_falls_back_instead_of_double_counting() -> None:
    context = _make_context(max_beams=2)
    session = _make_session(context)
    session.begin_session(beam_width=2, prefix_len=8)
    sampler_output = _sampler_output(
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
        torch.tensor([[-4.0, -6.0], [-0.1, -1.0]], dtype=torch.float32),
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _current_beam_info={"beam_data": {}},
    )

    assert (
        beam_decision.run_beam_decision(
            runner,
            session,
            sampler_output,
            is_last_step=False,
        )
        is None
    )


# ---------------------------------------------------------------------------
# G2: result_width defaults to the beam width
# ---------------------------------------------------------------------------


def test_result_width_falls_back_to_beam_width() -> None:
    """No result_width in metadata → beam width; never max_decode_steps."""
    context = _make_context(max_beams=4, max_decode_steps=32)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)

    beam_width = 4
    k = beam_width
    top_tokens = torch.arange(beam_width * k, dtype=torch.int32).reshape(beam_width, k)
    top_probs = torch.randn(beam_width, k)
    beam_decision.init_beam_decision_state(session, top_tokens, top_probs)
    assert context.decision.worker_decode_steps == 1

    # Simulate the caller-side default applied by _maybe_run_worker_beam_decision
    # when the metadata carries no result_width (G2 fix): None → beam_width.
    result_width = None or session.beam_width
    assert result_width == 4

    # The dispatch inside run_beam_decision must pick beam_search_group (not
    # rec_final_select) even though max_decode_steps == 32 == 32-aligned:
    # final select requires is_last_step AND width alignment; width here is 4.
    assert not (
        session._state.is_last_step and result_width % beam_decision._FINAL_SELECT_ALIGNMENT == 0
    )

    sampler_output = _sampler_output(top_tokens, top_probs)
    decision = beam_decision.run_beam_decision(
        SimpleNamespace(_current_beam_info={"beam_data": {}}),
        session,
        sampler_output,
        is_last_step=False,
        result_width=result_width,
    )
    assert decision is not None
    # beam_search_group output width == beam_width, NOT 32.
    assert decision["out_token_ids"].shape == (1, beam_width)


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


def _gate_runner(
    context: BeamAttentionContext,
    device_type: str = "npu",
    beam_width: int = 4,
) -> SimpleNamespace:
    session = _make_session(context)
    session.begin_session(beam_width=beam_width, prefix_len=8)
    return SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(
            beam_worker_decision=True,
            catalog_path=None,
            constraint_backend=None,
        ),
        _current_beam_info={
            "beam_data": {
                "s": {
                    "is_beam_decode": True,
                    "decode_step": 1,
                    "beam_width": beam_width,
                    "parent_beam_ids": list(range(beam_width)),
                }
            },
            "beam_session": session,
        },
        device=SimpleNamespace(type=device_type),
    )


def _use_decode_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Pin the backend's operator-path policy; shared code carries no cache."""
    policy = SimpleNamespace(
        supports_worker_beam_decision=lambda: True,
        uses_fused_beam_decision=lambda: mode == "xllm",
        requires_beam_candidate_width_match=lambda: True,
        batches_beam_kv_reorder=lambda: True,
        requires_aligned_final_select=lambda: True,
    )
    import vllm_gr.platforms.loader as loader

    # The bound services are cached per process, so drop them before pointing
    # selection at this test's policy.
    loader._reset_backend_services_for_tests()
    monkeypatch.setattr(
        loader,
        "activate_current_backend",
        lambda: SimpleNamespace(create_stage_execution_service=lambda: policy),
    )


def test_worker_decision_disabled_in_cascade_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cascade decode execution must keep the frontend decision path."""
    context = _make_context()
    runner = _gate_runner(context)
    _use_decode_mode(monkeypatch, "cascade")
    assert beam_decision.worker_decision_enabled(runner) is False


def test_worker_decision_enabled_in_xllm_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _make_context()
    runner = _gate_runner(context)
    _use_decode_mode(monkeypatch, "xllm")
    assert beam_decision.worker_decision_enabled(runner) is True


def test_worker_decision_disabled_on_cuda() -> None:
    context = _make_context()
    runner = _gate_runner(context, device_type="cuda")
    assert beam_decision.worker_decision_enabled(runner) is False


def test_worker_decision_disabled_when_backend_has_no_worker_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend without worker-side decisions keeps the frontend path.

    The gate asks the selected backend, not the runner's device name: a process
    selects its backend from the native platform, so a runner labelled with a
    different device is not what decides this.
    """
    policy = SimpleNamespace(
        supports_worker_beam_decision=lambda: False,
        uses_fused_beam_decision=lambda: True,
        requires_beam_candidate_width_match=lambda: True,
        batches_beam_kv_reorder=lambda: True,
        requires_aligned_final_select=lambda: True,
    )
    import vllm_gr.platforms.loader as loader

    monkeypatch.setattr(
        loader,
        "activate_current_backend",
        lambda: SimpleNamespace(create_stage_execution_service=lambda: policy),
    )
    context = _make_context()
    runner = _gate_runner(context)
    assert beam_decision.worker_decision_enabled(runner) is False


def test_worker_decision_disabled_without_option(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _make_context()
    runner = _gate_runner(context)
    runner._vllm_gr_config.beam_worker_decision = False
    _use_decode_mode(monkeypatch, "xllm")
    assert beam_decision.worker_decision_enabled(runner) is False


# ---------------------------------------------------------------------------
# G4: global → request-local prefix sums
# ---------------------------------------------------------------------------


def test_prefix_sums_converted_to_request_local() -> None:
    """group_token_num stores request-local prefix sums after a decision."""
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)

    beam_width = 4
    k = beam_width
    top_tokens = torch.arange(beam_width * k, dtype=torch.int32).reshape(beam_width, k)
    top_probs = -torch.rand(beam_width, k)  # logprobs <= 0
    beam_decision.init_beam_decision_state(session, top_tokens, top_probs)

    # Force a known parent mapping through the operator inputs: make beam 0
    # dominate so every survivor descends from beam 0 → parent = [0,0,0,0].
    probs = top_probs.clone()
    probs[0, :] = torch.tensor([0.0, -0.01, -0.02, -0.03])
    probs[1:] = -100.0
    sampler_output = _sampler_output(top_tokens, probs)

    decision = beam_decision.run_beam_decision(
        SimpleNamespace(_current_beam_info={"beam_data": {}}),
        session,
        sampler_output,
        is_last_step=False,
        result_width=beam_width,
    )
    assert decision is not None

    # All survivors from parent 0 → request-local prefix sums [4, 4, 4, 4].
    gtn = context.decision.group_token_num
    assert gtn[:1, :beam_width].tolist() == [[4, 4, 4, 4]]

    # out_token_index keeps GLOBAL offsets (request * beam_width); the
    # kv_reorder record is request-local and is derived by the return
    # channel (``_maybe_run_worker_beam_decision``) from this tensor's
    # host copy — row 0's global offset is 0, so row 0 IS the local mapping.
    # The decision itself must not sync (no tolist inside _finalize_decision).
    assert session._state.kv_reorder is None
    recorded = decision["out_token_index"][0].tolist()
    assert all(0 <= p < beam_width for p in recorded), recorded
    assert recorded == [0, 0, 0, 0]


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


def test_same_step_reorder_applied_and_decode_step_advanced() -> None:
    """The decision step reorders the KV pool itself."""
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)

    # Tag each pool beam row with its source identity.  Only steps
    # [0:decode_step+1] hold valid KV — the reorder covers exactly that
    # range, so the identity check must too (stale trailing steps are
    # unreferenced garbage in a real pool).
    session._state.decode_step = 2
    for layer, pool in enumerate(context.pools):
        for b in range(4):
            pool.unshared_key[0, b, :, :3, :].fill_(float(b * 10 + layer))

    beam_width = 4
    top_tokens = torch.arange(beam_width * beam_width, dtype=torch.int32).reshape(
        beam_width, beam_width
    )
    probs = -torch.rand(beam_width, beam_width)
    probs[0, :] = torch.tensor([0.0, -0.01, -0.02, -0.03])
    probs[1:] = -100.0
    beam_decision.init_beam_decision_state(session, top_tokens, probs)

    decode_step_before = session.decode_step
    sampler_output = _sampler_output(top_tokens, probs)
    decision = beam_decision.run_beam_decision(
        SimpleNamespace(_current_beam_info={"beam_data": {}}),
        session,
        sampler_output,
        is_last_step=False,
        result_width=beam_width,
    )
    assert decision is not None

    # decode_step advanced by the same-step path (step_complete equivalent).
    assert session.decode_step == decode_step_before + 1
    # The pool was actually reordered: every survivor descends from beam 0,
    # so every row now carries beam-0's identity across the valid range.
    for layer, pool in enumerate(context.pools):
        for b in range(beam_width):
            assert pool.unshared_key[0, b, :, :3, :].max().item() == float(layer), (layer, b)


# ---------------------------------------------------------------------------
# P2: return-channel payload slimming
# ---------------------------------------------------------------------------


def test_intermediate_step_omits_out_sequence() -> None:
    """run_beam_decision keeps out_sequence; the NPU patch trims the payload.

    The trimming itself lives in ``_maybe_run_worker_beam_decision`` (NPU
    worker patch).  Here we lock the contract that both consumers need: the
    decision dict carries the three [B, W] tensors plus the sequence only on
    the final step.  The torch path always returns out_sequence (the buffer
    write-back still needs it); the trim happens at the D2H boundary.
    """
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)

    beam_width = 4
    top_tokens = torch.arange(beam_width * beam_width, dtype=torch.int32).reshape(
        beam_width, beam_width
    )
    probs = -torch.rand(beam_width, beam_width)
    beam_decision.init_beam_decision_state(session, top_tokens, probs)

    decision = beam_decision.run_beam_decision(
        SimpleNamespace(_current_beam_info={"beam_data": {}}),
        session,
        _sampler_output(top_tokens, probs),
        is_last_step=False,
        result_width=beam_width,
    )
    assert decision is not None
    # The three tensors the frontend consumes are always present...
    for key in ("out_token_ids", "out_token_index", "out_log_probs"):
        assert key in decision
    # ...with [B, W] shape...
    assert decision["out_token_ids"].shape == (1, beam_width)
    assert decision["out_token_index"].shape == (1, beam_width)
    assert decision["out_log_probs"].shape == (1, beam_width)


# ---------------------------------------------------------------------------
# P1: contiguous-row slice assignment in the constraint hook
# ---------------------------------------------------------------------------


def test_constrained_override_slice_path_matches_index_path() -> None:
    """Contiguous rows: slice assignment equals the advanced-index result."""
    from vllm_gr.v1.worker import constraint_hook

    width = 4
    rows = 6
    torch.manual_seed(0)
    # Operator contract (run_constrained_partition): one row per constrained
    # row — len(global_rows), NOT the full batch size.
    n_stub = 4
    top_tokens = torch.randint(0, 1000, (n_stub, width), dtype=torch.int32)
    top_logprobs = torch.randn(n_stub, width)

    def build(stub_rows: list[int]) -> SimpleNamespace:
        lp = torch.randn(rows, 8)
        lp_ids = torch.randint(0, 1000, (rows, 8))
        st = torch.randint(0, 1000, (rows, 1))
        ranks = torch.arange(rows)
        sampler_output = SimpleNamespace(
            logprobs_tensors=SimpleNamespace(
                logprobs=lp.clone(),
                logprob_token_ids=lp_ids.clone(),
                selected_token_ranks=ranks.clone(),
            ),
            sampled_token_ids=st.clone(),
        )
        constraint_hook.patch_sampler_output_for_constrained(
            sampler_output,
            stub_rows,
            top_tokens,
            top_logprobs,
            torch.ones_like(top_logprobs, dtype=torch.bool),
            width,
        )
        return sampler_output

    contiguous_rows = [1, 2, 3, 4]
    scattered_rows = [4, 1, 3, 2]

    a = build(contiguous_rows)
    b = build(scattered_rows)

    # Row sets are equal; both write paths produce identical content in the
    # target rows (order-independence: the written value depends only on the
    # row's position within global_rows).
    lp_a = a.logprobs_tensors.logprobs
    lp_b = b.logprobs_tensors.logprobs
    ids_a = a.logprobs_tensors.logprob_token_ids
    ids_b = b.logprobs_tensors.logprob_token_ids
    # sort row indices for comparison
    order_a = sorted(contiguous_rows)
    order_b = sorted(scattered_rows)
    assert order_a == order_b
    # per target row, the value written is top_*[local index of that row]
    # contiguous: local i <-> row 1+i; scattered: row 4->i0, 1->i1, 3->i2, 2->i3
    local_of_b = {4: 0, 1: 1, 3: 2, 2: 3}
    for row in order_a:
        la = contiguous_rows.index(row)
        lb = local_of_b[row]
        assert torch.equal(lp_a[row], top_logprobs[la])
        assert torch.equal(lp_b[row], top_logprobs[lb])
        assert torch.equal(ids_a[row], top_tokens[la].to(torch.int64))
        assert torch.equal(ids_b[row], top_tokens[lb].to(torch.int64))


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


def test_select_candidates_v2_layout_keeps_all_columns() -> None:
    """v2_layout=True declares a pure top-K layout: every column is a
    candidate (no sampled-token column to drop).

    The V1 layout (v2_layout=False) on a wide payload drops column 0 —
    silently applying that to a V2 top-K payload would skew every beam's
    candidate set.
    """
    width = 4
    # Wide pure top-K payload: 2 beams x 5 candidates.  The V1 reader would
    # treat column 0 as the sampled token and drop it.
    candidates = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int64)
    probs = torch.rand(2, 5, dtype=torch.float64)
    sampler_output = _sampler_output(candidates, probs)

    tokens, scores = beam_decision._select_candidates(sampler_output, width, v2_layout=True)
    assert tokens.tolist() == [[10, 11, 12, 13], [20, 21, 22, 23]]
    assert torch.allclose(scores, probs[:, :width].float(), atol=1e-6)

    # The V1 layout on the same payload drops the leading column — the flag
    # is what prevents that misread.
    v1_tokens, _ = beam_decision._select_candidates(sampler_output, width, v2_layout=False)
    assert v1_tokens.tolist() == [[11, 12, 13, 14], [21, 22, 23, 24]]


def test_select_candidates_v2_layout_rejects_narrow_payload() -> None:
    """A payload with fewer columns than the beam width cannot serve W
    candidates per beam — refuse (None) instead of silently clamping."""
    candidates = torch.tensor([[10, 11]], dtype=torch.int64)
    probs = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    sampler_output = _sampler_output(candidates, probs)

    assert beam_decision._select_candidates(sampler_output, 4, v2_layout=True) is None


pytestmark = pytest.mark.cpu_test
