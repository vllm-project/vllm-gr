# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_gr.v1.beam.cpu_reference import (
    BeamItemCandidates,
    BeamReferenceBeam,
    BeamReferenceItemState,
    BeamReferenceState,
    normalize_constraint_candidates,
    normalize_sampler_candidates,
    select_final_sequences,
    select_initial_beams,
    select_next_beams,
)


def _constraint_candidates(
    item_index: int,
    parents: list[int],
    tokens: list[list[int]],
    scores: list[list[float]],
    mask: list[list[bool]] | None = None,
) -> BeamItemCandidates:
    return normalize_constraint_candidates(
        item_index=item_index,
        parent_beam_indices=np.asarray(parents, dtype=np.int64),
        token_ids=np.asarray(tokens, dtype=np.int64),
        logprobs=np.asarray(scores, dtype=np.float32),
        valid_mask=np.asarray(
            mask if mask is not None else np.ones_like(tokens, dtype=np.bool_),
            dtype=np.bool_,
        ),
    )


def test_sampler_normalization_freezes_sampled_slot_rule() -> None:
    candidates = normalize_sampler_candidates(
        item_index=0,
        parent_beam_indices=np.asarray([0, 1]),
        token_ids=np.asarray(
            [
                [7, 5, 7, 6],
                [9, 5, 7, 6],
            ]
        ),
        logprobs=np.asarray(
            [[-0.3, -0.1, -0.3, -0.4], [-0.7, -0.1, -0.2, -0.3]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(candidates.token_ids, [[5, 7, 6], [9, 5, 7]])
    np.testing.assert_allclose(
        candidates.logprobs,
        [[-0.1, -0.3, -0.4], [-0.7, -0.1, -0.2]],
    )


def test_constraint_normalization_keeps_duplicates_and_explicit_mask() -> None:
    candidates = _constraint_candidates(
        0,
        [0],
        [[8, 8, 123]],
        [[-0.1, -0.2, -1e20]],
        [[True, True, False]],
    )
    np.testing.assert_array_equal(candidates.token_ids, [[8, 8, 123]])
    assert np.isfinite(candidates.logprobs[0, 2])
    np.testing.assert_array_equal(candidates.valid_mask, [[True, True, False]])


def test_initial_selection_is_one_parent_to_w_and_separates_stops() -> None:
    candidates = _constraint_candidates(
        0,
        [0],
        [[10, 99, 11, 12]],
        [[-0.10, -0.05, -0.20, -0.15]],
    )
    state = select_initial_beams((candidates,), beam_width=3, stop_token_ids=(99,))
    item = state.items[0]
    assert item.num_active_beams == 3
    assert [beam.token_ids for beam in item.beams] == [(10,), (12,), (11,)]
    assert [beam.parent_beam_indices for beam in item.beams] == [(0,), (0,), (0,)]
    assert len(item.completed_beams) == 1
    assert item.completed_beams[0].token_ids == (99,)


def test_next_selection_scores_then_groups_stably_by_parent() -> None:
    initial = _constraint_candidates(0, [0], [[10, 11]], [[0.0, -0.1]])
    state = select_initial_beams((initial,), beam_width=2)
    decode = _constraint_candidates(
        0,
        [0, 1],
        [[20, 21], [22, 23]],
        [[-1.0, -0.2], [0.0, -0.3]],
    )
    state = select_next_beams(state, (decode,))
    beams = state.items[0].beams
    assert [beam.token_ids for beam in beams] == [(10, 21), (11, 22)]
    assert [beam.parent_beam_indices for beam in beams] == [(0, 0), (0, 1)]
    np.testing.assert_allclose([beam.cumulative_logprob for beam in beams], [-0.2, -0.1])


def test_exact_tie_prefers_lower_parent_major_logical_index() -> None:
    initial = _constraint_candidates(0, [0], [[10, 11]], [[0.0, -0.1]])
    state = select_initial_beams((initial,), beam_width=2)
    decode = _constraint_candidates(
        0,
        [0, 1],
        [[20, 21], [22, 23]],
        [[-0.05, -0.2], [-0.1, -1.0]],
    )
    state = select_next_beams(state, (decode,))
    assert [beam.token_ids[-1] for beam in state.items[0].beams] == [20, 21]


def test_stop_candidates_complete_without_entering_next_state() -> None:
    initial = _constraint_candidates(0, [0], [[10, 11]], [[0.0, -0.1]])
    state = select_initial_beams((initial,), beam_width=2)
    decode = _constraint_candidates(
        0,
        [0, 1],
        [[99, 20], [99, 21]],
        [[0.0, -0.4], [0.0, -0.5]],
    )
    stopped = select_next_beams(state, (decode,), stop_token_ids=(99,))
    assert [beam.token_ids[-1] for beam in stopped.items[0].beams] == [20, 21]
    assert [beam.stop_token_id for beam in stopped.items[0].completed_beams] == [99, 99]

    ignored = select_next_beams(
        state,
        (decode,),
        stop_token_ids=(99,),
        ignore_stop_tokens=True,
    )
    assert not ignored.items[0].completed_beams
    assert [beam.token_ids[-1] for beam in ignored.items[0].beams] == [99, 99]


def test_one_token_request_finalizes_without_intermediate_state() -> None:
    candidates = _constraint_candidates(0, [0], [[2, 10]], [[-0.1, -0.2]])
    output = select_final_sequences(
        None,
        (candidates,),
        beam_width=2,
        num_return_sequences=2,
        stop_token_ids=(2,),
        include_stop_token=False,
        include_cumulative_logprob=False,
    )
    sequences = output.items[0].sequences
    assert sequences[0].finish_reason == "stop"
    assert sequences[0].token_ids == ()
    assert sequences[0].parent_beam_indices == (0,)
    assert sequences[0].cumulative_logprob is None
    assert sequences[1].finish_reason == "length"
    assert sequences[1].token_ids == (10,)


def test_final_tie_prefers_earlier_completed_generation() -> None:
    initial = _constraint_candidates(
        0,
        [0],
        [[99, 10, 11]],
        [[-0.25, 0.0, -0.5]],
    )
    state = select_initial_beams((initial,), beam_width=2, stop_token_ids=(99,))
    final = _constraint_candidates(0, [0, 1], [[20], [21]], [[-0.25], [-1.0]])
    output = select_final_sequences(
        state,
        (final,),
        beam_width=2,
        num_return_sequences=1,
        stop_token_ids=(99,),
    )
    assert output.items[0].sequences[0].token_ids == (99,)
    assert output.items[0].sequences[0].finish_reason == "stop"


def test_final_shortfall_is_compact_normal_output() -> None:
    candidates = _constraint_candidates(
        0,
        [0],
        [[10, 11, 12]],
        [[-0.1, -1e20, -1e20]],
        [[True, False, False]],
    )
    output = select_final_sequences(
        None,
        (candidates,),
        beam_width=3,
        num_return_sequences=3,
    )
    assert [sequence.token_ids for sequence in output.items[0].sequences] == [(10,)]


def test_items_with_zero_active_beams_continue_independently() -> None:
    initial = (
        _constraint_candidates(0, [0], [[10]], [[0.0]]),
        _constraint_candidates(1, [0], [[20]], [[0.0]]),
    )
    state = select_initial_beams(initial, beam_width=1)
    decode = (
        _constraint_candidates(0, [0], [[99]], [[0.0]]),
        _constraint_candidates(1, [0], [[21]], [[-0.1]]),
    )
    state = select_next_beams(state, decode, stop_token_ids=(99,))
    assert state.items[0].num_active_beams == 0
    assert state.items[1].num_active_beams == 1

    empty = normalize_constraint_candidates(
        item_index=0,
        parent_beam_indices=np.empty((0,), dtype=np.int64),
        token_ids=np.empty((0, 1), dtype=np.int64),
        logprobs=np.empty((0, 1), dtype=np.float32),
        valid_mask=np.empty((0, 1), dtype=np.bool_),
    )
    final = (empty, _constraint_candidates(1, [0], [[22]], [[-0.1]]))
    output = select_final_sequences(
        state,
        final,
        beam_width=1,
        num_return_sequences=1,
        stop_token_ids=(99,),
    )
    assert output.items[0].sequences[0].token_ids == (10, 99)
    assert output.items[1].sequences[0].token_ids == (20, 21, 22)


def test_fp32_accumulation_is_intentional() -> None:
    state = BeamReferenceState(
        beam_width=2,
        num_generated_tokens=1,
        items=(
            BeamReferenceItemState(
                item_index=0,
                beams=(
                    BeamReferenceBeam(True, (10,), np.float32(-1e8), (0,)),
                    BeamReferenceBeam(True, (11,), np.float32(-1e8), (0,)),
                ),
                num_active_beams=2,
                completed_beams=(),
            ),
        ),
    )
    candidates = _constraint_candidates(0, [0, 1], [[20], [21]], [[-2.0], [-1.0]])
    output = select_final_sequences(
        state,
        (candidates,),
        beam_width=2,
        num_return_sequences=1,
    )
    assert output.items[0].sequences[0].token_ids[-1] == 20
    assert float(np.float32(-1e8)) - 1.0 > float(np.float32(-1e8)) - 2.0


def test_legacy_argpartition_matches_away_from_ties_and_fp32_boundaries() -> None:
    rng = np.random.default_rng(20260821)
    beam_width = candidate_width = 4
    parent_scores = np.asarray([-0.1, -0.4, -0.7, -1.0], dtype=np.float32)
    state = BeamReferenceState(
        beam_width=beam_width,
        num_generated_tokens=1,
        items=(
            BeamReferenceItemState(
                item_index=0,
                beams=tuple(
                    BeamReferenceBeam(True, (100 + beam,), score, (0,))
                    for beam, score in enumerate(parent_scores)
                ),
                num_active_beams=beam_width,
                completed_beams=(),
            ),
        ),
    )
    local = rng.uniform(-2.0, -0.01, size=(beam_width, candidate_width)).astype(np.float32)
    tokens = np.arange(beam_width * candidate_width).reshape(beam_width, candidate_width)
    candidates = _constraint_candidates(
        0,
        list(range(beam_width)),
        tokens.tolist(),
        local.tolist(),
    )
    reference = select_next_beams(state, (candidates,))

    flat_scores = (parent_scores[:, None].astype(np.float64) + local.astype(np.float64)).reshape(-1)
    legacy_indices = np.argpartition(-flat_scores, beam_width)[:beam_width]
    legacy = {
        (int(index) // candidate_width, int(tokens.reshape(-1)[index])) for index in legacy_indices
    }
    actual = {
        (beam.parent_beam_indices[-1], beam.token_ids[-1])
        for beam in reference.items[0].beams
        if beam.is_active
    }
    assert actual == legacy


def test_beam_package_and_cpu_reference_import_without_torch() -> None:
    code = """
import sys
import vllm_gr.v1.beam
assert 'torch' not in sys.modules
import vllm_gr.v1.beam.cpu_reference
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr


def test_reference_trusts_explicit_mask_for_finite_padding_sentinel() -> None:
    candidates = _constraint_candidates(0, [0], [[10]], [[-1e20]], [[True]])
    state = select_initial_beams((candidates,), beam_width=1)
    assert state.items[0].beams[0].is_active
    assert np.isclose(
        state.items[0].beams[0].cumulative_logprob,
        -1e20,
        rtol=1e-6,
        atol=0.0,
    )


def test_initial_selection_rejects_fabricated_w_parent_rows() -> None:
    candidates = _constraint_candidates(0, [0, 1], [[10], [11]], [[-0.1], [-0.2]])
    with pytest.raises(ValueError, match="exactly one"):
        select_initial_beams((candidates,), beam_width=2)
