# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_gr.beam import (
    BeamCandidate,
    BeamFinishedReason,
    BeamState,
    length_penalized_score,
    prune_candidates,
    rank_candidates,
    rank_states,
)


def _candidate(parent_index: int, token_id: int, score: float) -> BeamCandidate:
    return BeamCandidate(
        parent_index=parent_index,
        token_id=token_id,
        token_logprob=score,
        cumulative_logprob=score,
    )


def test_rank_candidates_uses_score_then_deterministic_tie_breakers() -> None:
    candidates = (
        _candidate(1, 5, -0.1),
        _candidate(0, 7, -0.1),
        _candidate(0, 6, -0.1),
        _candidate(0, 8, -0.2),
    )

    ranked = rank_candidates(candidates)

    assert [(item.parent_index, item.token_id) for item in ranked] == [
        (0, 6),
        (0, 7),
        (1, 5),
        (0, 8),
    ]


def test_prune_candidates_removes_masked_candidates_and_limits_width() -> None:
    candidates = (
        _candidate(0, 1, -0.5),
        _candidate(0, 2, -float("inf")),
        _candidate(1, 3, -0.1),
        _candidate(1, 4, -0.2),
    )

    selected = prune_candidates(candidates, beam_width=2)

    assert [item.token_id for item in selected] == [3, 4]


def test_prune_candidates_removes_non_finite_candidates() -> None:
    candidates = (
        _candidate(0, 1, float("nan")),
        _candidate(0, 2, float("inf")),
        _candidate(0, 3, -0.1),
    )

    assert prune_candidates(candidates, beam_width=3) == (candidates[2],)


def test_prune_candidates_matches_full_ranking_at_top_k_boundary() -> None:
    candidates = (
        _candidate(2, 3, -0.1),
        _candidate(0, 7, -0.1),
        _candidate(0, 6, -0.1),
        _candidate(1, 4, -0.1),
        _candidate(0, 5, -0.2),
    )

    assert prune_candidates(candidates, beam_width=3) == rank_candidates(candidates)[:3]


def test_length_penalty_excludes_eos_and_optional_prompt() -> None:
    assert length_penalized_score((10, 11, 12, 2), -6.0, 2) == pytest.approx(-2.0)
    assert length_penalized_score((10, 11, 12, 2), -6.0, 2, prompt_length=2) == pytest.approx(-6.0)


def test_length_penalty_rejects_sequence_without_scored_tokens() -> None:
    with pytest.raises(ValueError, match="scored token"):
        length_penalized_score((2,), -1.0, 2)


def test_rank_states_applies_length_penalty() -> None:
    shorter = BeamState(
        token_ids=(10, 20, 2),
        cumulative_logprob=-2.0,
        finish_reason=BeamFinishedReason.STOP,
    )
    longer = BeamState(
        token_ids=(10, 20, 21, 22, 2),
        cumulative_logprob=-3.0,
        finish_reason=BeamFinishedReason.STOP,
    )

    assert rank_states((shorter, longer), eos_token_id=2) == (longer, shorter)
