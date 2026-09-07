# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-def"

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    extract_and_dedup_flat_logprobs,
    extract_and_dedup_segment,
    extract_constrained_flat_logprobs,
    extract_constrained_segment,
    mask_candidate_logprobs,
    materialize_selected_beams,
    select_top_indices,
)

# Padding sentinel shared with the constrained top-k operator (execution.py).
INVALID_LOGIT = -1.0e20

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
)
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


def test_extract_constrained_segment_filters_transport_padding() -> None:
    # The Worker converts structurally invalid slots to -inf for transport.
    token_ids, logprobs, ranks, decoded = extract_constrained_segment(
        [11, 12, 13, 14],
        [-0.1, -0.2, -float("inf"), -float("inf")],
        [0, 0, 0, 0],
        ["a", "b", None, None],
        0,
        4,
        2,
    )

    assert token_ids == [11, 12]
    assert logprobs == [-0.1, -0.2]
    assert ranks == [0, 0]
    assert decoded == ["a", "b"]


def test_extract_constrained_segment_keeps_legal_sentinel_magnitude() -> None:
    token_ids, logprobs, *_ = extract_constrained_segment(
        [11, 12, 0],
        [0.0, INVALID_LOGIT, -float("inf")],
        None,
        None,
        0,
        3,
        2,
    )

    assert token_ids == [11, 12]
    assert logprobs == [0.0, INVALID_LOGIT]


def test_extract_constrained_segment_never_dedups() -> None:
    # Unlike the sampled-token path, constrained rows are fixed-width operator
    # output with no duplicate slot to strip; identical tokens survive.
    token_ids, logprobs, *_ = extract_constrained_segment(
        [5, 5, 9, 9],
        [-0.1, -0.2, -float("inf"), -float("inf")],
        None,
        None,
        0,
        4,
        2,
    )

    assert token_ids == [5, 5]


def test_extract_constrained_segment_drops_inf_remux_padding() -> None:
    # The beam remux pads short rows with -inf; the non-finite filter drops it.
    token_ids, logprobs, *_ = extract_constrained_segment(
        [11, 0, 0, 0],
        [-0.1, -float("inf"), -float("inf"), -float("inf")],
        None,
        None,
        0,
        4,
        2,
    )

    assert token_ids == [11]
    assert logprobs == [-0.1]


def test_extract_constrained_segment_empty_valid_set() -> None:
    token_ids, logprobs, *_ = extract_constrained_segment(
        [0, 0, 0, 0],
        [-float("inf")] * 4,
        None,
        None,
        0,
        4,
        2,
    )

    assert token_ids == []
    assert logprobs == []


def test_extract_constrained_segment_respects_slice_bounds() -> None:
    # Slicing into a wider flat (e.g. one parent block of a grouped decode);
    # the block is exactly ``logprobs_num`` wide so every entry survives.
    token_ids, logprobs, *_ = extract_constrained_segment(
        [99, 21, 22, 99, 23, 24],
        [-9.0, -0.1, -0.2, -9.0, -float("inf"), -float("inf")],
        None,
        None,
        1,
        4,
        2,
    )

    assert token_ids == [21, 22]
    assert logprobs == [-0.1, -0.2]


def test_extract_constrained_segment_truncates_to_logprobs_num() -> None:
    # Prefill-style wide segment: more valid entries than logprobs_num must be
    # capped (in practice there are never more than W valid per parent).
    token_ids, logprobs, *_ = extract_constrained_segment(
        [21, 22, 23, 24],
        [-0.1, -0.2, -0.3, -float("inf")],
        None,
        None,
        0,
        4,
        2,
    )

    assert token_ids == [21, 22]
    assert logprobs == [-0.1, -0.2]


def test_extract_constrained_segment_ignores_row_wide_rank_metadata() -> None:
    token_ids, logprobs, ranks, decoded = extract_constrained_segment(
        [10, 11, 12, 13],
        [-0.1, -0.2, -0.3, -0.4],
        [0],
        None,
        0,
        4,
        4,
    )
    assert token_ids == [10, 11, 12, 13]
    assert logprobs == [-0.1, -0.2, -0.3, -0.4]
    assert ranks == [None, None, None, None]
    assert decoded is None


def test_extract_constrained_flat_logprobs_prefill_wide_segment() -> None:
    # Constrained prefill emits one wide flat segment: W valid candidates then
    # structural transport padding.
    flat = SimpleNamespace(
        token_ids=[31, 32, 33, 0, 0, 0],
        logprobs=[-0.1, -0.2, -0.3, -float("inf"), -float("inf"), -float("inf")],
        ranks=[0, 0, 0, 0, 0, 0],
        decoded_tokens=[None, None, None, None, None, None],
        start_indices=[0],
        end_indices=[6],
    )

    token_ids, logprobs, ranks, decoded = extract_constrained_flat_logprobs(flat, 3)

    assert token_ids == [31, 32, 33]
    assert logprobs == [-0.1, -0.2, -0.3]
    assert ranks == [0, 0, 0]
    assert decoded == [None, None, None]


@pytest.mark.parametrize(
    "extract_fn",
    [extract_and_dedup_flat_logprobs, extract_constrained_flat_logprobs],
)
def test_extract_flat_logprobs_empty_output_returns_no_candidates(extract_fn) -> None:
    # An aborted / mid-shutdown output can carry ``logprobs is not None`` with
    # no transported candidate block; extraction must degrade to "no
    # candidates", never raise IndexError.
    flat = SimpleNamespace(
        token_ids=[],
        logprobs=[],
        ranks=None,
        decoded_tokens=None,
        start_indices=[],
        end_indices=[],
    )

    token_ids, logprobs, ranks, decoded = extract_fn(flat, 5)

    assert token_ids == []
    assert logprobs == []
    assert ranks is None
    assert decoded is None


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
)
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
)
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


def test_top_indices_exclude_completed_and_invalid_candidates() -> None:
    scores = np.array([-np.inf, -0.2, -np.inf, -0.4])

    selected = select_top_indices(
        scores,
        beam_width=4,
        catalog_strategy="fill",
        exclude_nonfinite=True,
    )

    assert selected.tolist() == [1, 3]
