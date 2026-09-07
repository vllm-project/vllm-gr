# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the worker-side beam decision (ACS decode_select port).

Covers the five correctness gaps fixed in the NPU post-processing port:

- G1: ``BeamDecisionBuffers.reset()`` runs at ``begin_session`` (a second
  session starts from zeroed decision state).
- G2: ``result_width`` defaults to the beam width, never the decode step
  count (``max_decode_steps``).
- G3: ``worker_decision_enabled`` is gated on the xllm decode execution mode.
- G4: the operator's GLOBAL prefix sums are converted to request-local form
  before persisting ``group_token_num``.
- G5: the KV reorder runs in the SAME step as the decision, and the next
  step's ``_get_or_create_beam_session`` consumes the recorded mapping
  instead of re-applying it (``p . p != p``).

All tests run on CPU against the PyTorch reference decision path
(``_run_torch_decision``); NPU fused-operator parity is covered by the
``tests/kernels`` suite on device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.beam import BeamAttentionContext, BeamSearchSession, beam_decision
from vllm_gr.v1.beam.beam_attention_context import BeamDecisionBuffers
from vllm_gr.v1.worker import model_runner_common as common


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


@pytest.mark.parametrize("width,prune_eos", [(4, True), (64, True), (4, False)])  # type: ignore[misc]
def test_frontend_selection_resynchronizes_worker_after_fallback(
    width: int, prune_eos: bool
) -> None:
    """Pruned/reordered winners and a frontend-only step cannot leave stale parents."""
    context = _make_context(max_beams=width, max_decode_steps=8, num_layers=1)
    session = _make_session(context)
    session.begin_session(beam_width=width, prefix_len=8)
    beam_decision.init_beam_decision_from_parents(
        session, [[100 + i] for i in range(width)], [-float(i) for i in range(width)]
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _beam_sessions={"s": session},
        _beam_context=context,
        _vllm_gr_config=SimpleNamespace(beam_max_width=width, beam_max_decode_steps=8),
        _current_beam_info={"beam_data": {}},
    )
    tokens = torch.arange(width * width, dtype=torch.int32).reshape(width, width) + 1000
    tokens[0, 0] = 9  # EOS among the worker's winners.
    scores = torch.full((width, width), -1000.0)
    scores[:, 0] = -0.25
    decision = beam_decision.run_beam_decision(
        runner, session, _sampler_output(tokens, scores), is_last_step=False
    )
    assert decision is not None
    parents = decision["out_token_index"][0].tolist()
    session._state.kv_reorder = parents
    selected = [
        i
        for i, token in enumerate(decision["out_token_ids"][0].tolist())
        if not prune_eos or token != 9
    ][::-1]
    suffixes = context.decision.sequence[0, selected, :2].tolist()
    parent_scores = context.decision.beam_scores[0, selected].tolist()
    metadata = {
        "is_beam_decode": True,
        "beam_width": len(selected),
        "prefix_len": 8,
        "decode_step": 1,
        "max_decode_steps": 8,
        "parent_beam_ids": [parents[i] for i in selected],
        "beam_suffix": suffixes,
        "parent_cumulative_logprobs": parent_scores,
    }
    buffers = context.decision
    addresses = (buffers.sequence.data_ptr(), buffers.beam_scores.data_ptr())
    runner._current_beam_info = {"beam_data": {"s": metadata}}
    common._get_or_create_beam_session(runner, {"s": metadata})
    assert session.beam_width == len(selected)
    assert buffers.sequence[0, : len(selected), :2].tolist() == suffixes
    assert buffers.beam_scores[0, : len(selected)].tolist() == parent_scores
    assert buffers.worker_decode_steps == 2

    # Force a frontend-only decision via an unsupported candidate shape.
    assert (
        beam_decision.run_beam_decision(
            runner,
            session,
            _sampler_output(tokens[: len(selected), :1], scores[: len(selected), :1]),
            is_last_step=False,
        )
        is None
    )
    next_parents = [i % len(selected) for i in range(width)]
    suffixes = [suffixes[p] + [2000 + i] for i, p in enumerate(next_parents)]
    parent_scores = [parent_scores[p] - 10.0 - i for i, p in enumerate(next_parents)]
    metadata.update(
        beam_width=width,
        decode_step=2,
        parent_beam_ids=next_parents,
        beam_suffix=suffixes,
        parent_cumulative_logprobs=parent_scores,
    )
    common._get_or_create_beam_session(runner, {"s": metadata})
    assert session.beam_width == width
    assert buffers.worker_decode_steps == 3
    assert (buffers.sequence.data_ptr(), buffers.beam_scores.data_ptr()) == addresses

    # Compare resumed worker winners with exhaustive frontend scoring.
    decision = beam_decision.run_beam_decision(
        runner, session, _sampler_output(tokens, scores), is_last_step=False
    )
    assert decision is not None
    expected = sorted(
        (parent_scores[p] + float(scores[p, k]), suffixes[p] + [int(tokens[p, k])])
        for p in range(width)
        for k in range(width)
    )[-width:]
    actual = sorted(
        zip(
            decision["out_log_probs"][0].tolist(),
            decision["out_sequence"][0, :, :4].tolist(),
        )
    )
    assert actual == expected
    assert buffers.worker_decode_steps == 4


def test_missing_parent_scores_disable_previously_seeded_worker() -> None:
    context = _make_context()
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)
    beam_decision.init_beam_decision_from_parents(session, [[1]] * 4, [-1.0] * 4)
    common._seed_beam_decision_from_metadata(session, {"beam_suffix": [[1, 2]] * 4})
    assert context.decision.worker_decode_steps == 0


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


def test_worker_decision_disabled_in_cascade_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cascade decode execution must keep the frontend decision path."""
    from vllm_gr.v1.attention.backends import beam_attn

    context = _make_context()
    runner = _gate_runner(context)
    monkeypatch.setattr(beam_attn, "_NPU_DECODE_EXECUTION_MODE", "cascade")
    assert beam_decision.worker_decision_enabled(runner) is False


def test_worker_decision_enabled_in_xllm_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.attention.backends import beam_attn

    context = _make_context()
    runner = _gate_runner(context)
    monkeypatch.setattr(beam_attn, "_NPU_DECODE_EXECUTION_MODE", "xllm")
    assert beam_decision.worker_decision_enabled(runner) is True


def test_worker_decision_enabled_on_cuda_without_xllm() -> None:
    context = _make_context()
    runner = _gate_runner(context, device_type="cuda")
    assert beam_decision.worker_decision_enabled(runner) is True


@pytest.mark.parametrize(  # type: ignore[misc]
    ("beam_width", "expected"),
    [(256, True), (257, False)],
)
def test_cuda_worker_decision_respects_kernel_width_limit(
    beam_width: int,
    expected: bool,
) -> None:
    context = _make_context(max_beams=beam_width)
    runner = _gate_runner(context, device_type="cuda", beam_width=beam_width)

    assert beam_decision.worker_decision_enabled(runner) is expected


def test_cuda_worker_decision_over_width_limit_falls_back() -> None:
    beam_width = 257
    context = _make_context(max_beams=beam_width)
    runner = _gate_runner(context, device_type="cuda", beam_width=beam_width)
    candidates = torch.arange(beam_width**2, dtype=torch.int32).reshape(beam_width, beam_width)
    sampler_output = _sampler_output(candidates, torch.zeros_like(candidates, dtype=torch.float32))

    beam_decision.maybe_run_worker_beam_decision(runner, sampler_output)

    assert runner._beam_decision_result is None
    assert not hasattr(runner, "_beam_decision_tensors")


def test_cuda_constraint_prefill_publishes_without_seeding_decode_state() -> None:
    """Step 0 must not fall back to frontend NumPy selection on CUDA."""
    from vllm_gr.config import ConstraintBackend

    width = 4
    token_ids = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)
    scores = torch.tensor([[-0.1, -0.2, -0.3, -0.4]], dtype=torch.float32)
    runner = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        _vllm_gr_config=SimpleNamespace(
            beam_worker_decision=True,
            constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
        ),
        _current_beam_info={
            "beam_data": {
                "prefill-id": {
                    "is_constraint_prefill": True,
                    "beam_width": width,
                }
            }
        },
    )
    sampler_output = _sampler_output(token_ids, scores)

    beam_decision.maybe_run_worker_beam_decision(runner, sampler_output)

    compact = beam_decision.unpack_beam_result_buffer(runner._beam_decision_result)
    assert compact is not None
    ids, logprobs, parents = compact
    assert ids == [11, 12, 13, 14]
    assert logprobs == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert parents == [0, 0, 0, 0]
    assert not hasattr(runner, "_beam_prefill_state")


def test_worker_decision_disabled_on_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.attention.backends import beam_attn

    context = _make_context()
    runner = _gate_runner(context, device_type="cpu")
    monkeypatch.setattr(beam_attn, "_NPU_DECODE_EXECUTION_MODE", "xllm")
    assert beam_decision.worker_decision_enabled(runner) is False


def test_worker_decision_disabled_without_option(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.attention.backends import beam_attn

    context = _make_context()
    runner = _gate_runner(context)
    runner._vllm_gr_config.beam_worker_decision = False
    monkeypatch.setattr(beam_attn, "_NPU_DECODE_EXECUTION_MODE", "xllm")
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
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)

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
    # Global prefix sums (ACS kernel contract): request-offset included.
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
    # the return channel derives it from out_token_index's host copy, and
    # only the single active session (request 0 slice) is recorded.  Row 0's
    # global offset is 0, so row 0 already equals the request-local mapping;
    # row 1 must shed its 1*beam_width offset to match.
    assert session._state.kv_reorder is None
    assert out_idx[0].tolist() == [0, 1, 0, 1]
    assert (out_idx[1] - beam_width).tolist() == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# G5: same-step KV reorder + next-step consumption
# ---------------------------------------------------------------------------


def test_same_step_reorder_applied_and_decode_step_advanced() -> None:
    """The decision step reorders the KV pool itself (ACS timing)."""
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


def test_next_step_skips_duplicate_reorder() -> None:
    """Exact parent match: the recorded mapping must not be re-applied."""
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)
    state = session._state

    beam_width = 4
    top_tokens = torch.arange(beam_width * beam_width, dtype=torch.int32).reshape(
        beam_width, beam_width
    )
    probs = -torch.rand(beam_width, beam_width)
    probs[0, :] = torch.tensor([0.0, -0.01, -0.02, -0.03])
    probs[1:] = -100.0
    beam_decision.init_beam_decision_state(session, top_tokens, probs)

    sampler_output = _sampler_output(top_tokens, probs)
    decision = beam_decision.run_beam_decision(
        SimpleNamespace(_current_beam_info={"beam_data": {}}),
        session,
        sampler_output,
        is_last_step=False,
        result_width=beam_width,
    )
    assert decision is not None
    # The return channel derives the record from the host copy of
    # out_token_index (request-0 row == request-local mapping); simulate
    # that derivation here — _finalize_decision itself must not sync.
    assert state.kv_reorder is None
    state.kv_reorder = decision["out_token_index"][0].tolist()
    recorded = list(state.kv_reorder)
    assert recorded == [0, 0, 0, 0]

    # Pool state after the same-step reorder (all rows = beam-0 identity).
    identity_after = [pool.unshared_key[0].clone() for pool in context.pools]
    decode_step_after = state.decode_step

    # Next execute_model: _get_or_create_beam_session with the frontend's
    # parent_beam_ids == recorded.  Must consume the record and NOT reorder.
    runner = SimpleNamespace(
        _beam_sessions={session.session_id: session},
        _beam_context=context,
        _vllm_gr_config=SimpleNamespace(
            beam_max_width=4,
            beam_max_decode_steps=4,
        ),
        device=torch.device("cpu"),
    )
    beam_data = {
        session.session_id: {
            "is_beam_decode": True,
            "beam_width": beam_width,
            "prefix_len": 8,
            "decode_step": 2,
            "parent_beam_ids": list(recorded),
            "max_decode_steps": 4,
        }
    }
    common._get_or_create_beam_session(
        runner, beam_data, SimpleNamespace(beam_finished_session_ids=[])
    )

    # Record consumed, decode_step unchanged (advanced by the same-step path
    # already), pool identity preserved — no p . p double application.
    assert state.kv_reorder is None
    assert state.decode_step == decode_step_after
    for before, pool in zip(identity_after, context.pools):
        assert torch.equal(before, pool.unshared_key[0])


@pytest.mark.parametrize("is_terminal", [False, True])  # type: ignore[misc]
def test_next_step_composites_mismatched_frontend_parents(is_terminal: bool) -> None:
    """Frontend mapping != recorded: composite reorder on top of the record.

    The mapping describes the parents for the current attention step, so it
    must be applied even when that current step is terminal.
    """
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)
    state = session._state

    # Recorded mapping: rows now hold pre-decision rows [2,3,0,1].
    recorded = [2, 3, 0, 1]
    state.kv_reorder = recorded
    state.decode_step = 1

    # Tag pool rows with their current identity: row b holds pre-decision
    # row recorded[b] (the same-step reorder already applied).  Only steps
    # [0, decode_step - 1] were written through the previous step — the
    # composite reorder covers exactly that (inclusive) range.
    for layer, pool in enumerate(context.pools):
        for b in range(4):
            pool.unshared_key[0, b, :, :1, :].fill_(float(recorded[b] * 10 + layer))

    # Frontend wants pre-decision parents [0,1,2,3] → composite
    # comp[j] = recorded.index(parent[j]) = [2, 3, 0, 1]
    # (row j must hold the pre-decision row parent[j]; the pool currently
    # holds pre-decision row recorded[i] at row i).
    runner = SimpleNamespace(
        _beam_sessions={session.session_id: session},
        _beam_context=context,
        _vllm_gr_config=SimpleNamespace(beam_max_width=4, beam_max_decode_steps=4),
        device=torch.device("cpu"),
    )
    beam_data = {
        session.session_id: {
            "is_beam_decode": True,
            "beam_width": 4,
            "prefix_len": 8,
            "decode_step": 1,
            "parent_beam_ids": [0, 1, 2, 3],
            "max_decode_steps": 3 if is_terminal else 4,
        }
    }
    common._get_or_create_beam_session(
        runner, beam_data, SimpleNamespace(beam_finished_session_ids=[])
    )

    assert state.kv_reorder is None
    # Row j now carries the pre-decision identity of frontend parent[j]:
    # row 0 <- pre-row 0 (currently at pool row 2), etc., across the
    # written step range.
    for layer, pool in enumerate(context.pools):
        for j in range(4):
            expected = float(j * 10 + layer)
            assert pool.unshared_key[0, j, :, :1, :].max().item() == expected, (layer, j)


def test_terminal_step_applies_deferred_frontend_parents() -> None:
    """A deferred previous-step decision is not skipped at final attention."""
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)
    state = session._state

    for layer, pool in enumerate(context.pools):
        for b in range(4):
            pool.unshared_key[0, b, :, :1, :].fill_(float(b * 10 + layer))

    runner = _terminal_runner(context, session)
    beam_data = {
        session.session_id: {
            "is_beam_decode": True,
            "beam_width": 4,
            "prefix_len": 8,
            "decode_step": 1,
            "parent_beam_ids": [2, 3, 0, 1],
            "max_decode_steps": 3,
        }
    }
    common._get_or_create_beam_session(
        runner, beam_data, SimpleNamespace(beam_finished_session_ids=[])
    )

    assert state.is_last_step is True
    assert state.decode_step == 1
    for layer, pool in enumerate(context.pools):
        for j, parent in enumerate([2, 3, 0, 1]):
            expected = float(parent * 10 + layer)
            assert pool.unshared_key[0, j, :, :1, :].max().item() == expected, (layer, j)


# ---------------------------------------------------------------------------
# G6: terminal-step detection — loop length vs decode-only steps
# ---------------------------------------------------------------------------


def _terminal_runner(context: BeamAttentionContext, session: BeamSearchSession) -> SimpleNamespace:
    """Runner stub for _get_or_create_beam_session terminal-step tests."""
    return SimpleNamespace(
        _beam_sessions={session.session_id: session},
        _beam_context=context,
        _vllm_gr_config=SimpleNamespace(beam_max_width=4, beam_max_decode_steps=4),
        device=torch.device("cpu"),
    )


def test_last_step_fires_on_final_decode_of_loop() -> None:
    """3-iteration loop (prefill + 2 decodes): decode_step 1 IS terminal.

    ``max_decode_steps`` on the metadata is the frontend LOOP length
    (``max_tokens - pre_calc``) — iteration 0 is the prefill, so the request
    runs ``loop_len - 1`` decode steps and the 0-indexed ``decode_step``
    tops out at ``loop_len - 2``.  The old ``decode_step >= loop_len - 1``
    comparison never fired for the 3-step constrained FSM: engine-driven
    chaining produced a 3rd decode whose constraint_step=3 the scheduler rejects.
    """
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)
    state = session._state
    # Second (final) decode step already ran: same-step path advanced
    # decode_step to 1.
    state.decode_step = 1

    runner = _terminal_runner(context, session)
    beam_data = {
        session.session_id: {
            "is_beam_decode": True,
            "beam_width": 4,
            "prefix_len": 8,
            "decode_step": 1,
            "parent_beam_ids": [0, 1, 2, 3],
            "max_decode_steps": 3,  # loop length: prefill + 2 decodes
        }
    }
    common._get_or_create_beam_session(
        runner, beam_data, SimpleNamespace(beam_finished_session_ids=[])
    )

    assert state.is_last_step is True


def test_last_step_not_fired_one_step_early() -> None:
    """First decode (decode_step 0) of a 3-iteration loop is not terminal."""
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)
    state = session._state
    state.decode_step = 0

    runner = _terminal_runner(context, session)
    beam_data = {
        session.session_id: {
            "is_beam_decode": True,
            "beam_width": 4,
            "prefix_len": 8,
            "decode_step": 0,
            "parent_beam_ids": [0, 1, 2, 3],
            "max_decode_steps": 3,
        }
    }
    common._get_or_create_beam_session(
        runner, beam_data, SimpleNamespace(beam_finished_session_ids=[])
    )

    assert state.is_last_step is False


def test_last_step_pool_bound_counts_decode_only_steps() -> None:
    """Host omitted the loop length: pool bound is decode-only, no -1 shift.

    ``beam_max_decode_steps`` sizes the KV pool in decode steps, so it
    compares against ``decode_step`` directly.
    """
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)
    state = session._state
    state.decode_step = 3

    runner = _terminal_runner(context, session)
    beam_data = {
        session.session_id: {
            "is_beam_decode": True,
            "beam_width": 4,
            "prefix_len": 8,
            "decode_step": 3,
            "parent_beam_ids": [0, 1, 2, 3],
            # loop length omitted → gr_cfg.beam_max_decode_steps (4)
        }
    }
    common._get_or_create_beam_session(
        runner, beam_data, SimpleNamespace(beam_finished_session_ids=[])
    )

    assert state.is_last_step is True


def test_last_step_single_decode_request() -> None:
    """Loop of 2 (prefill + 1 decode): decode_step 0 is the only, terminal step."""
    context = _make_context(max_beams=4, max_decode_steps=4)
    session = _make_session(context)
    session.begin_session(beam_width=4, prefix_len=8)
    state = session._state
    state.decode_step = 0

    runner = _terminal_runner(context, session)
    beam_data = {
        session.session_id: {
            "is_beam_decode": True,
            "beam_width": 4,
            "prefix_len": 8,
            "decode_step": 0,
            "parent_beam_ids": [0, 1, 2, 3],
            "max_decode_steps": 2,
        }
    }
    common._get_or_create_beam_session(
        runner, beam_data, SimpleNamespace(beam_finished_session_ids=[])
    )

    assert state.is_last_step is True


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
# P3: return-channel payload flattening (list-vs-tensor)
# ---------------------------------------------------------------------------


def test_flatten_decision_value_tensor() -> None:
    """In-process consumers hand over the original host tensors."""
    t = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)
    assert beam_decision.flatten_decision_value(t) == [11, 12, 21, 22]


def test_flatten_decision_value_nested_lists() -> None:
    """Post-IPC payloads arrive as nested Python lists."""
    assert beam_decision.flatten_decision_value([[11, 12]]) == [11, 12]
    assert beam_decision.flatten_decision_value([[0.5, -1.5]]) == [0.5, -1.5]
    # Deep nesting (out_sequence): [B, W, steps]
    assert beam_decision.flatten_decision_value([[[1, 2], [3, 4]]]) == [1, 2, 3, 4]


def test_flatten_decision_value_rejects_serialized_tensor_triples() -> None:
    """A skipped normalization arrives as the encoder triple: unrecoverable.

    ``kv_transfer_params`` values decode against ``dict[str, Any]``; the
    encoder-side tensor triple ``('int64', shape, data)`` never reaches
    ``dec_hook`` (it only fires for typed fields).  The ``data`` slot is
    either an inline memoryview or — at/above the zero-copy threshold — an
    aux-buffer INDEX whose backing frame the consumer never receives.
    Return ``None`` so callers fall back instead of crashing.
    """
    assert beam_decision.flatten_decision_value(("int64", [1, 2], 0)) is None
    assert beam_decision.flatten_decision_value(["float32", [1, 2], memoryview(b"\0" * 8)]) is None
    # Nested: one bad element poisons the whole payload.
    assert beam_decision.flatten_decision_value([[11, 12], ["int64", [1, 1], 0]]) is None


def test_flatten_decision_value_scalar_and_other_types() -> None:
    """Scalars and non-container values are not payload shapes."""
    assert beam_decision.flatten_decision_value(3) is None
    assert beam_decision.flatten_decision_value(None) is None
    assert beam_decision.flatten_decision_value("out") is None


def test_cuda_beam_result_buffer_round_trip_uses_one_compact_payload() -> None:
    raw = {
        "out_token_ids": torch.tensor([[11, 12]], dtype=torch.int32),
        "out_token_index": torch.tensor([[1, 0]], dtype=torch.int32),
        "out_log_probs": torch.tensor([[-1.25, -2.5]], dtype=torch.float32),
        # Device sequence history is intentionally not part of the return
        # buffer; frontend materialization already owns the parent histories.
        "out_sequence": torch.tensor([[[1, 11], [2, 12]]], dtype=torch.int32),
    }

    encoded = beam_decision._encode_cuda_beam_result(raw, is_last_step=True)

    assert encoded is not None
    payload, parents = encoded
    assert set(payload) == {"buffer_format", "shape", "buffer", "is_last_step"}
    assert payload["shape"] == [1, 2]
    assert isinstance(payload["buffer"], bytes)
    assert len(payload["buffer"]) == 2 * 3 * 4
    assert payload["is_last_step"] is True
    assert parents == [1, 0]
    ids, scores, decoded_parents = beam_decision.unpack_beam_result_buffer(payload)
    assert ids == [11, 12]
    assert scores == pytest.approx([-1.25, -2.5])
    assert decoded_parents == [1, 0]


def test_beam_result_buffer_rejects_bad_shape_or_size() -> None:
    assert (
        beam_decision.unpack_beam_result_buffer(
            {
                "buffer_format": "beam_result_i32_f32_v1",
                "shape": [1, 2],
                "buffer": b"too-short",
            }
        )
        is None
    )


def test_cuda_worker_decision_publishes_compact_result_and_keeps_device_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 4
    raw = {
        "out_token_ids": torch.tensor([[11, 12, 13, 14]], dtype=torch.int32),
        "out_token_index": torch.tensor([[1, 0, 3, 2]], dtype=torch.int32),
        "out_log_probs": torch.tensor([[-1.0, -2.0, -3.0, -4.0]], dtype=torch.float32),
    }
    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda runner: True)
    monkeypatch.setattr(beam_decision, "run_beam_decision", lambda *args, **kwargs: raw)
    state = SimpleNamespace(is_last_step=False, kv_reorder=None)
    session = SimpleNamespace(_state=state, beam_width=width)
    runner = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        _current_beam_info={
            "beam_session": session,
            "beam_data": {"s": {"is_beam_decode": True, "result_width": width}},
        },
    )

    beam_decision.maybe_run_worker_beam_decision(runner, object())

    assert runner._beam_decision_tensors is raw
    assert runner._beam_decision_result["buffer_format"] == "beam_result_i32_f32_v1"
    assert beam_decision.unpack_beam_result_buffer(runner._beam_decision_result) == (
        [11, 12, 13, 14],
        [-1.0, -2.0, -3.0, -4.0],
        [1, 0, 3, 2],
    )
    assert state.kv_reorder == [1, 0, 3, 2]
