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
    find_eos_indices,
    mask_candidate_logprobs,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.logprobs import extract_and_dedup_flat_logprobs

CandidateCache = tuple[
    list[int],
    list[float],
    list[int | None],
    list[str | None],
]


class _FlatLogprobsStub:
    start_indices = [0]
    end_indices = [4]
    token_ids = [7, 3, 7, 5]
    logprobs = [-0.01, -0.1, -0.2, -0.3]
    ranks = [1, 2, 3, 4]
    decoded_tokens = ["a", "b", "c", "d"]


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


def test_complete_eos_candidates_preserves_offline_behavior() -> None:
    first_lora = LoRARequest("first", 1, "/tmp/first")
    second_lora = LoRARequest("second", 2, "/tmp/second")
    parents = [
        _beam([10], score=-0.1, lora_request=first_lora),
        _beam([20], score=-0.2, lora_request=second_lora),
    ]
    cache = [
        _candidate_cache([2, 11], [-0.3, -0.4]),
        _candidate_cache([21, 2], [-0.5, -0.6]),
    ]
    scores = np.asarray([-0.4, -0.5, -0.7, -0.8], dtype=np.float32)
    initial_logprobs = FlatLogprobs()

    eos_beams, masked_scores = complete_eos_candidates(
        parents=parents,
        token_ids=np.asarray([2, 11, 21, 2], dtype=np.int64),
        scores=scores,
        beam_idx=np.array([0, 0, 1, 1]),
        beam_flat_cache=cache,
        candidates_per_parent=2,
        eos_token_id=2,
        initial_logprobs=initial_logprobs,
        include_eos_token=True,
        inherit_lora_request=True,
    )

    assert masked_scores is not scores
    assert scores.tolist() == pytest.approx([-0.4, -0.5, -0.7, -0.8])
    assert masked_scores.tolist() == pytest.approx([-float("inf"), -0.5, -0.7, -float("inf")])
    assert [beam.tokens for beam in eos_beams] == [[10, 2], [20, 2]]
    assert [beam.finish_reason for beam in eos_beams] == ["stop", "stop"]
    assert [beam.stop_reason for beam in eos_beams] == [2, 2]
    assert [beam.lora_request for beam in eos_beams] == [first_lora, second_lora]
    assert eos_beams[0].orig_prompt is parents[0].orig_prompt
    assert eos_beams[1].orig_prompt is parents[1].orig_prompt
    assert [beam._lp_parent for beam in eos_beams] == parents
    assert [beam._lp_step_data for beam in eos_beams] == cache
    assert all(beam.logprobs is initial_logprobs for beam in eos_beams)
    assert all(type(beam.cum_logprob) is float for beam in eos_beams)


def test_complete_eos_candidates_preserves_openai_behavior() -> None:
    lora_request = LoRARequest("adapter", 1, "/tmp/adapter")
    parent = _beam([10], lora_request=lora_request)
    cache = [_candidate_cache([2, 11], [-0.1, -0.2])]

    eos_beams, masked_scores = complete_eos_candidates(
        parents=[parent],
        token_ids=np.asarray([2, 11]),
        scores=np.asarray([-0.1, -0.2]),
        beam_idx=np.array([0, 0]),
        beam_flat_cache=cache,
        candidates_per_parent=2,
        eos_token_id=2,
        initial_logprobs=FlatLogprobs(),
        include_eos_token=False,
        inherit_lora_request=False,
    )

    assert len(eos_beams) == 1
    assert eos_beams[0].tokens is parent.tokens
    assert eos_beams[0].lora_request is None
    assert masked_scores.tolist() == [-float("inf"), -0.2]


def test_complete_eos_candidates_rejects_missing_parent_cache() -> None:
    with pytest.raises(AssertionError):
        complete_eos_candidates(
            parents=[_beam([10])],
            token_ids=np.asarray([2]),
            scores=np.asarray([-0.1]),
            beam_idx=np.array([0]),
            beam_flat_cache=[None],
            candidates_per_parent=1,
            eos_token_id=2,
            initial_logprobs=FlatLogprobs(),
            include_eos_token=True,
            inherit_lora_request=True,
        )


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


def test_materialize_selected_beams_propagates_catalog_node_on_valid_token() -> None:
    parent = _beam([10])
    parent._catalog_node = {11: {12: {}}, 99: {}}
    cache = [_candidate_cache([11, 99], [-0.1, -0.2])]
    new_beams, fork_info = materialize_selected_beams(
        parents=[parent],
        selected_indices=np.asarray([0]),
        token_ids=np.asarray([11, 99], dtype=np.int64),
        scores=np.asarray([-0.1, -0.2], dtype=np.float32),
        beam_flat_cache=cache,
        candidates_per_parent=2,
        initial_logprobs=FlatLogprobs(),
    )

    assert len(new_beams) == 1
    assert new_beams[0].tokens == [10, 11]
    # Must be *the same dict object* as parent._catalog_node[11],
    # proving we're caching the trie node pointer, not copying it.
    assert new_beams[0]._catalog_node is parent._catalog_node[11]


def test_materialize_selected_beams_propagates_catalog_node_on_invalid_token() -> None:
    parent = _beam([10])
    parent._catalog_node = {11: {12: {}}}
    cache = [_candidate_cache([99], [-0.1])]
    new_beams, fork_info = materialize_selected_beams(
        parents=[parent],
        selected_indices=np.asarray([0]),
        token_ids=np.asarray([99], dtype=np.int64),
        scores=np.asarray([-0.1], dtype=np.float32),
        beam_flat_cache=cache,
        candidates_per_parent=1,
        initial_logprobs=FlatLogprobs(),
    )

    assert len(new_beams) == 1
    assert new_beams[0].tokens == [10, 99]
    # token 99 is not a child of parent._catalog_node, so new node is empty dict
    assert new_beams[0]._catalog_node == {}


def test_materialize_selected_beams_sets_catalog_node_none_when_parent_missing() -> None:
    parent = _beam([10])  # no _catalog_node set
    cache = [_candidate_cache([11], [-0.1])]
    new_beams, fork_info = materialize_selected_beams(
        parents=[parent],
        selected_indices=np.asarray([0]),
        token_ids=np.asarray([11], dtype=np.int64),
        scores=np.asarray([-0.1], dtype=np.float32),
        beam_flat_cache=cache,
        candidates_per_parent=1,
        initial_logprobs=FlatLogprobs(),
    )

    assert new_beams[0]._catalog_node is None


def test_complete_eos_candidates_propagates_catalog_node() -> None:
    parent = _beam([10])
    parent._catalog_node = {2: {99: {}}}
    cache = [_candidate_cache([2, 11], [-0.1, -0.2])]

    eos_beams, masked_scores = complete_eos_candidates(
        parents=[parent],
        token_ids=np.asarray([2, 11], dtype=np.int64),
        scores=np.asarray([-0.1, -0.2], dtype=np.float32),
        beam_flat_cache=cache,
        beam_idx=np.asarray([0, 0], dtype=np.int64),
        candidates_per_parent=2,
        eos_token_id=2,
        initial_logprobs=FlatLogprobs(),
        include_eos_token=True,
        inherit_lora_request=True,
    )

    assert len(eos_beams) == 1
    assert eos_beams[0].tokens == [10, 2]
    # Must be *the same dict object* as parent._catalog_node[2],
    # proving pointer propagation through the EOS path.
    assert eos_beams[0]._catalog_node is parent._catalog_node[2]


def test_materialize_selected_beams_preserves_selection_order_and_lineage() -> None:
    first_lora = LoRARequest("first", 1, "/tmp/first")
    second_lora = LoRARequest("second", 2, "/tmp/second")
    parents = [
        _beam([10], lora_request=first_lora),
        _beam([20], lora_request=second_lora),
    ]
    cache = [
        _candidate_cache([11, 12], [-0.1, -0.2]),
        _candidate_cache([21, 22], [-0.3, -0.4]),
    ]
    initial_logprobs = FlatLogprobs()

    new_beams, fork_info = materialize_selected_beams(
        parents=parents,
        selected_indices=np.asarray([3, 0], dtype=np.int64),
        token_ids=np.asarray([11, 12, 21, 22], dtype=np.int64),
        scores=np.asarray([-0.1, -0.2, -0.3, -0.4], dtype=np.float32),
        beam_flat_cache=cache,
        candidates_per_parent=2,
        initial_logprobs=initial_logprobs,
    )

    assert [beam.tokens for beam in new_beams] == [[20, 22], [10, 11]]
    assert fork_info == [(1, 22), (0, 11)]
    assert [beam.lora_request for beam in new_beams] == [second_lora, first_lora]
    assert new_beams[0].orig_prompt is parents[1].orig_prompt
    assert new_beams[1].orig_prompt is parents[0].orig_prompt
    assert [beam._lp_parent for beam in new_beams] == [parents[1], parents[0]]
    assert [beam._lp_step_data for beam in new_beams] == [cache[1], cache[0]]
    assert all(beam.logprobs is initial_logprobs for beam in new_beams)
    assert all(type(beam.tokens[-1]) is int for beam in new_beams)
    assert all(type(beam.cum_logprob) is float for beam in new_beams)


def test_materialize_selected_beams_rejects_missing_parent_cache() -> None:
    with pytest.raises(AssertionError):
        materialize_selected_beams(
            parents=[_beam([10])],
            selected_indices=[0],
            token_ids=np.asarray([11]),
            scores=np.asarray([-0.1]),
            beam_flat_cache=[None],
            candidates_per_parent=1,
            initial_logprobs=FlatLogprobs(),
        )
