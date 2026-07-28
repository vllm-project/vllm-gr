# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm_gr.beam_search_decision_utils import (
    extract_and_dedup_segment,
    find_eos_indices,
    mask_candidate_logprobs,
    select_top_indices,
)
from vllm_gr.logprobs import extract_and_dedup_flat_logprobs


class _FlatLogprobsStub:
    start_indices = [0]
    end_indices = [4]
    token_ids = [7, 3, 7, 5]
    logprobs = [-0.01, -0.1, -0.2, -0.3]
    ranks = [1, 2, 3, 4]
    decoded_tokens = ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([7, 3, 7, 5], [3, 7, 5]),
        ([7, 3, 4, 5], [7, 3, 4]),
    ],
)  # type: ignore[misc]
def test_extract_and_dedup_segment_matches_legacy_rule(
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
    assert len(result[1]) == len(expected)
    assert len(result[2]) == len(expected)
    assert len(result[3]) == len(expected)


def test_extract_and_dedup_segment_preserves_optional_metadata() -> None:
    result = extract_and_dedup_segment(
        [90, 7, 3],
        [-9.0, -0.1, -0.2],
        None,
        None,
        1,
        3,
        2,
    )

    assert result == ([7, 3], [-0.1, -0.2], None, None)


def test_extract_and_dedup_segment_preserves_optional_elements() -> None:
    result = extract_and_dedup_segment(
        [7, 3],
        [-0.1, -0.2],
        [None, 2],
        [None, "b"],
        0,
        2,
        2,
    )

    assert result == ([7, 3], [-0.1, -0.2], [None, 2], [None, "b"])


def test_logprobs_wrapper_uses_shared_legacy_extraction() -> None:
    result = extract_and_dedup_flat_logprobs(_FlatLogprobsStub(), 3)

    assert result == (
        [3, 7, 5],
        [-0.1, -0.2, -0.3],
        [2, 3, 4],
        ["b", "c", "d"],
    )


def test_mask_candidate_logprobs_without_catalog_returns_list_copy() -> None:
    logprobs = [-0.1, -0.2]

    result = mask_candidate_logprobs([7, 3], logprobs, None)

    assert result == logprobs
    assert result is not logprobs


def test_mask_candidate_logprobs_preserves_candidates_and_order() -> None:
    result = mask_candidate_logprobs(
        [7, 3, 5, 9],
        [-0.1, -float("inf"), -0.3, -0.4],
        {3, 9},
    )

    assert result == [-float("inf"), -float("inf"), -float("inf"), -0.4]
    assert len(result) == 4


def test_mask_candidate_logprobs_preserves_legacy_zip_length() -> None:
    result = mask_candidate_logprobs(
        [7, 3, 5],
        [-0.1, -0.2],
        set(),
    )

    assert result == [-float("inf"), -float("inf")]


def test_find_eos_indices_matches_parent_expression() -> None:
    token_ids = np.asarray([2, 7, 2, 9], dtype=np.int64)

    actual = find_eos_indices(token_ids, 2)
    expected = np.where(token_ids == 2)[0]

    assert np.array_equal(actual, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize(
    ("scores", "beam_width"),
    [
        ([-0.5, -float("inf"), -0.1, -0.2], 2),
        ([-float("inf")] * 4, 2),
        ([-0.1, -0.2], 2),
    ],
)  # type: ignore[misc]
def test_select_top_indices_matches_parent_expression(
    scores: list[float],
    beam_width: int,
) -> None:
    score_array = np.asarray(scores)
    if score_array.size > beam_width:
        expected = np.argpartition(np.negative(score_array), beam_width)[:beam_width]
    else:
        expected = np.arange(score_array.size)

    actual = select_top_indices(score_array, beam_width)

    assert np.array_equal(actual, expected)
    assert actual.dtype == expected.dtype
