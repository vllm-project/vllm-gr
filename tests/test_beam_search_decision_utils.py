# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    extract_and_dedup_segment,
    mask_candidate_logprobs,
    materialize_selected_beams,
    select_top_indices,
)

CandidateCache = tuple[
    list[int],
    list[float],
    list[int | None],
    list[str | None],
]


def _beam(
    tokens: list[int],
    score: float = 0.0,
    lora_request: LoRARequest | None = None,
) -> BeamSearchSequence:
    beam = BeamSearchSequence(
        orig_prompt=tokens_input(tokens),
        tokens=tokens,
        logprobs=FlatLogprobs(),
        cum_logprob=score,
        lora_request=lora_request,
    )
    beam._lp_parent = None
    beam._lp_step_data = None
    return beam


def _candidate_cache(
    token_ids: list[int],
    logprobs: list[float],
) -> CandidateCache:
    size = len(token_ids)
    return token_ids, logprobs, [None] * size, [None] * size


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([7, 3, 7, 5], [3, 7, 5]),
    ],
)  # type: ignore[misc]
def test_extract_and_dedup_segment_preserves_candidate_order(
    token_ids: list[int],
    expected: list[int],
) -> None:
    result = extract_and_dedup_segment(
        token_ids,
        [-0.01, -0.1, -0.2, -0.3],
        [1, 2, 3, 4],
        ["a", "b", "c", "d"],
        0,
        4,
        3,
    )

    assert result[0] == expected
    assert all(len(values) == len(expected) for values in result)


def test_mask_candidate_logprobs_keeps_candidate_geometry() -> None:
    result = mask_candidate_logprobs(
        [7, 3, 5, 9],
        [-0.1, -0.2, -0.3, -0.4],
        {3, 9},
    )

    assert result == [-float("inf"), -0.2, -float("inf"), -0.4]


def test_complete_stop_candidates_uses_explicit_nonuniform_parent_mapping() -> None:
    first_lora = LoRARequest("first", 1, "/tmp/first")
    third_lora = LoRARequest("third", 3, "/tmp/third")
    parents = [
        _beam([10], lora_request=first_lora),
        _beam([20]),
        _beam([30], lora_request=third_lora),
    ]
    cache = [
        _candidate_cache([2, 11], [-0.1, -0.2]),
        _candidate_cache([], []),
        _candidate_cache([31, 3], [-0.3, -0.4]),
    ]
    scores = np.asarray([-0.1, -0.2, -0.3, -0.4])

    completed, masked_scores = complete_eos_candidates(
        parents=parents,
        token_ids=np.asarray([2, 11, 31, 3]),
        scores=scores,
        beam_idx=np.asarray([0, 0, 2, 2], dtype=np.intp),
        beam_flat_cache=cache,
        stop_token_ids={2, 3},
        initial_logprobs=FlatLogprobs(),
        include_eos_token=True,
        inherit_lora_request=True,
    )

    assert [beam.tokens for beam in completed] == [[10, 2], [30, 3]]
    assert [beam.stop_reason for beam in completed] == [2, 3]
    assert [beam._lp_parent for beam in completed] == [parents[0], parents[2]]
    assert [beam._lp_step_data for beam in completed] == [cache[0], cache[2]]
    assert [beam.lora_request for beam in completed] == [first_lora, third_lora]
    assert masked_scores.tolist() == [-float("inf"), -0.2, -0.3, -float("inf")]
    assert scores.tolist() == [-0.1, -0.2, -0.3, -0.4]


def test_complete_stop_candidates_preserves_openai_output_policy() -> None:
    lora_request = LoRARequest("adapter", 1, "/tmp/adapter")
    parent = _beam([10], lora_request=lora_request)

    completed, _ = complete_eos_candidates(
        parents=[parent],
        token_ids=np.asarray([3]),
        scores=np.asarray([-0.1]),
        beam_idx=np.asarray([0], dtype=np.intp),
        beam_flat_cache=[_candidate_cache([3], [-0.1])],
        stop_token_ids={2, 3},
        initial_logprobs=FlatLogprobs(),
        include_eos_token=False,
        inherit_lora_request=False,
    )

    assert completed[0].tokens is parent.tokens
    assert completed[0].stop_reason == 3
    assert completed[0].lora_request is None


@pytest.mark.parametrize(
    ("scores", "beam_width"),
    [
        ([-0.5, -float("inf"), -0.1, -0.2], 2),
        ([-0.1, -0.2], 2),
    ],
)  # type: ignore[misc]
def test_fill_top_indices_preserves_legacy_argpartition(
    scores: list[float],
    beam_width: int,
) -> None:
    score_array = np.asarray(scores)
    expected = (
        np.argpartition(np.negative(score_array), beam_width)[:beam_width]
        if score_array.size > beam_width
        else np.arange(score_array.size)
    )

    actual = select_top_indices(score_array, beam_width, "fill")

    assert np.array_equal(actual, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize(
    ("scores", "beam_width", "expected"),
    [
        ([-0.1, -float("inf"), -0.2], 4, [0, 2]),
        ([-float("inf"), -float("inf")], 2, []),
    ],
)  # type: ignore[misc]
def test_null_top_indices_only_returns_finite_candidates(
    scores: list[float],
    beam_width: int,
    expected: list[int],
) -> None:
    actual = select_top_indices(np.asarray(scores), beam_width, "null")

    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.intp
    assert actual.tolist() == expected


def test_materialize_selected_beams_uses_explicit_nonuniform_parent_mapping() -> None:
    parents = [_beam([10]), _beam([20]), _beam([30])]
    cache = [
        _candidate_cache([11, 12], [-0.4, -0.5]),
        _candidate_cache([], []),
        _candidate_cache([31, 32], [-0.1, -0.2]),
    ]

    new_beams, fork_info = materialize_selected_beams(
        parents=parents,
        selected_indices=np.asarray([3, 0], dtype=np.intp),
        token_ids=np.asarray([11, 12, 31, 32]),
        scores=np.asarray([-0.4, -0.5, -0.1, -0.2]),
        beam_idx=np.asarray([0, 0, 2, 2], dtype=np.intp),
        beam_flat_cache=cache,
        initial_logprobs=FlatLogprobs(),
    )

    assert [beam.tokens for beam in new_beams] == [[30, 32], [10, 11]]
    assert [beam._lp_parent for beam in new_beams] == [parents[2], parents[0]]
    assert [beam._lp_step_data for beam in new_beams] == [cache[2], cache[0]]
    assert fork_info == [(2, 32), (0, 11)]
