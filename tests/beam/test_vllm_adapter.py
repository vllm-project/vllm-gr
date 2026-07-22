# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.beam import BeamCandidateSegment, BeamRequestContext
from vllm_gr.entrypoints.beam_adapter import (
    apply_vllm_beam_step,
    candidate_segment_from_flat,
    candidate_segments_from_combined_flat,
    select_best_beams,
)


def _sequence(tokens: list[int], score: float = 0.0) -> BeamSearchSequence:
    sequence = BeamSearchSequence(
        orig_prompt=tokens_input(tokens),
        tokens=tokens,
        logprobs=FlatLogprobs(),
        cum_logprob=score,
    )
    sequence._lp_parent = None
    sequence._lp_step_data = None
    return sequence


def test_candidate_segment_from_flat_uses_shared_deduplication() -> None:
    flat = FlatLogprobs(
        start_indices=[0],
        end_indices=[4],
        token_ids=[7, 3, 7, 5],
        logprobs=[-0.01, -0.1, -0.2, -0.3],
        ranks=[1, 2, 3, 4],
        decoded_tokens=["seven", "three", "seven", "five"],
    )

    segment = candidate_segment_from_flat(flat, candidate_count=3)

    assert segment.token_ids == (3, 7, 5)
    assert segment.logprobs == (-0.1, -0.2, -0.3)
    assert segment.ranks == (2, 3, 4)


def test_combined_flat_segments_use_active_parent_count_for_stride() -> None:
    flat = FlatLogprobs(
        token_ids=[7, 3, 7, 8, 4, 8],
        logprobs=[-0.01, -0.1, -0.2, -0.02, -0.3, -0.4],
        ranks=[1, 2, 3, 1, 2, 3],
        decoded_tokens=[None] * 6,
    )

    segments = candidate_segments_from_combined_flat(
        flat,
        parent_count=2,
        candidate_count=2,
    )

    assert [segment.token_ids for segment in segments] == [(3, 7), (4, 8)]


def test_combined_flat_segments_reject_uneven_parent_blocks() -> None:
    flat = FlatLogprobs(token_ids=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])

    with pytest.raises(ValueError, match="divide evenly"):
        candidate_segments_from_combined_flat(flat, parent_count=2, candidate_count=1)


def test_apply_vllm_beam_step_materializes_domain_result_and_logprob_trace() -> None:
    parents = [_sequence([10], -0.5), _sequence([20], -0.2)]
    segments = (
        BeamCandidateSegment(
            token_ids=(2, 11),
            logprobs=(-0.1, -0.3),
            ranks=(1, 2),
            decoded_tokens=("</s>", "a"),
        ),
        BeamCandidateSegment(
            token_ids=(21, 22),
            logprobs=(-0.2, -0.4),
        ),
    )
    context = BeamRequestContext(
        beam_width=2,
        eos_token_id=2,
        include_stop_token=False,
    )

    result = apply_vllm_beam_step(
        parents,
        segments,
        context,
        FlatLogprobs(),
        allowed_token_ids=(frozenset({2, 11}), frozenset({21})),
    )

    assert [beam.tokens for beam in result.active] == [[20, 21], [10, 11]]
    assert [beam.cum_logprob for beam in result.active] == [-0.4, -0.8]
    assert result.fork_info == [(1, 21), (0, 11)]
    assert len(result.completed) == 1
    assert result.completed[0].tokens == [10]
    assert result.completed[0].finish_reason == "stop"
    assert result.completed[0].stop_reason == 2
    assert result.completed[0]._lp_parent is parents[0]
    assert result.active[0]._lp_step_data[1] == [-0.2, -float("inf")]
    assert result.active[0]._lp_step_data[2] == [None, None]


def test_select_best_beams_uses_length_penalty() -> None:
    shorter = _sequence([10, 20, 2], -2.0)
    longer = _sequence([10, 20, 21, 22, 2], -3.0)
    context = BeamRequestContext(beam_width=1, eos_token_id=2, length_penalty=1.0)

    assert select_best_beams([shorter, longer], context) == [longer]
