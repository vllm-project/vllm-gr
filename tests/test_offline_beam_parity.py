# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.entrypoints.gr import _parse_step_logprobs

CandidateCache = tuple[
    list[int],
    list[float],
    list[int | None],
    list[str | None],
]


def _beam(tokens: list[int], score: float = 0.0) -> BeamSearchSequence:
    beam = BeamSearchSequence(
        orig_prompt=tokens_input(tokens),
        tokens=tokens,
        logprobs=FlatLogprobs(),
        cum_logprob=score,
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


def _result_with_flat(flat: object) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(finish_reason=None, logprobs=flat)],
    )


def _run_legacy_decision_step(
    *,
    parents: list[BeamSearchSequence],
    token_ids: list[int],
    scores: list[float],
    beam_flat_cache: list[CandidateCache],
    candidates_per_parent: int,
    beam_width: int,
    eos_token_id: int = 2,
    ignore_eos: bool = False,
) -> tuple[
    list[BeamSearchSequence],
    list[BeamSearchSequence],
    list[tuple[int, int]],
    np.ndarray,
]:
    instance = SimpleNamespace(beams=parents, completed=[])
    token_array = np.asarray(token_ids)
    score_array = np.asarray(scores)
    if ignore_eos:
        selection_scores = score_array
    else:
        eos_beams, selection_scores = complete_eos_candidates(
            parents=instance.beams,
            token_ids=token_array,
            scores=score_array,
            beam_flat_cache=beam_flat_cache,
            candidates_per_parent=candidates_per_parent,
            eos_token_id=eos_token_id,
            initial_logprobs=FlatLogprobs(),
            include_eos_token=True,
            inherit_lora_request=True,
        )
        instance.completed.extend(eos_beams)
    selected_indices = select_top_indices(selection_scores, beam_width)
    instance.beams, fork_info = materialize_selected_beams(
        parents=instance.beams,
        selected_indices=selected_indices,
        token_ids=token_array,
        scores=selection_scores,
        beam_flat_cache=beam_flat_cache,
        candidates_per_parent=candidates_per_parent,
        initial_logprobs=FlatLogprobs(),
    )
    return instance.completed, instance.beams, fork_info, selection_scores


@pytest.mark.parametrize(
    ("raw_token_ids", "expected_token_ids", "expected_logprobs"),
    [
        ([7, 3, 7, 5], [3, 7, 5], [-0.1, -0.2, -0.3]),
        ([7, 3, 4, 5], [7, 3, 4], [-0.01, -0.1, -0.2]),
    ],
)  # type: ignore[misc]
def test_initial_candidate_extraction_matches_legacy_sampled_token_rule(
    raw_token_ids: list[int],
    expected_token_ids: list[int],
    expected_logprobs: list[float],
) -> None:
    parent = _beam([10], -0.5)
    instance = SimpleNamespace(beams=[parent], completed=[], prompt_len=1)
    flat = SimpleNamespace(
        start_indices=[0],
        end_indices=[4],
        token_ids=raw_token_ids,
        logprobs=[-0.01, -0.1, -0.2, -0.3],
        ranks=[None, 2, 3, 4],
        decoded_tokens=[None, "b", "c", "d"],
    )

    is_error, token_ids, scores, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=3,
        fork_info_j=None,
    )

    assert not is_error
    assert token_ids == expected_token_ids
    assert scores == pytest.approx([-0.5 + value for value in expected_logprobs])
    assert cache[0][0] == expected_token_ids
    assert cache[0][1] == expected_logprobs
    assert len(cache[0][2]) == len(expected_token_ids)
    assert len(cache[0][3]) == len(expected_token_ids)


def test_grouped_flat_logprobs_uses_floor_stride_and_ignores_remainder() -> None:
    parents = [_beam([10], -0.5), _beam([20], -0.2)]
    instance = SimpleNamespace(beams=parents, completed=[], prompt_len=1)
    flat = SimpleNamespace(
        token_ids=[7, 3, 7, 8, 4, 8, 999],
        logprobs=[-0.01, -0.1, -0.2, -0.02, -0.3, -0.4, -9.0],
    )
    is_error, token_ids, scores, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=1,
        logprobs_num=2,
        fork_info_j=[(0, 7), (1, 8)],
    )

    assert not is_error
    assert token_ids == [3, 7, 4, 8]
    assert scores == pytest.approx([-0.6, -0.7, -0.5, -0.6])
    assert [entry[0] for entry in cache] == [[3, 7], [4, 8]]
    assert all(entry[2] is None and entry[3] is None for entry in cache)
    assert 999 not in token_ids


def test_catalog_mask_keeps_candidate_count_order_and_query_timing() -> None:
    parent = _beam([10, 99], -0.5)
    instance = SimpleNamespace(beams=[parent], completed=[], prompt_len=1)
    flat = SimpleNamespace(
        start_indices=[0],
        end_indices=[2],
        token_ids=[7, 3],
        logprobs=[-0.1, -0.2],
        ranks=[None, 2],
        decoded_tokens=[None, "three"],
    )
    catalog_calls: list[list[int]] = []

    class _Catalog:
        def valid(self, generated_tokens: list[int]) -> set[int]:
            catalog_calls.append(generated_tokens)
            return {3}

    is_error, token_ids, scores, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=2,
        fork_info_j=None,
        catalog=_Catalog(),
    )

    assert not is_error
    assert catalog_calls == [[99]]
    assert token_ids == [7, 3]
    assert scores == [-float("inf"), -0.7]
    assert cache[0][0] == [7, 3]
    assert cache[0][1] == [-float("inf"), -0.2]


def test_catalog_mask_uses_catalog_node_when_available() -> None:
    """When beam has _catalog_node, the optimized path must NOT call catalog.valid()."""
    parent = _beam([10, 99], -0.5)
    parent._catalog_node = {3: {}}  # pre-computed node: only token 3 is valid
    instance = SimpleNamespace(beams=[parent], completed=[], prompt_len=1)
    flat = SimpleNamespace(
        start_indices=[0],
        end_indices=[2],
        token_ids=[7, 3],
        logprobs=[-0.1, -0.2],
        ranks=[None, 2],
        decoded_tokens=[None, "three"],
    )
    calls: list[list[int]] = []

    class _Catalog:
        def valid(self, generated_tokens: list[int]) -> set[int]:
            calls.append(generated_tokens)
            return {7, 3}  # if called, would NOT mask token 7

    is_error, token_ids, scores, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=2,
        fork_info_j=None,
        catalog=_Catalog(),
    )

    assert not is_error
    assert calls == [], "catalog.valid() must NOT be called when _catalog_node is set"
    assert scores == [-float("inf"), -0.7]  # only token 3 survives (dict {3: {}} keys)
    assert cache[0][1] == [-float("inf"), -0.2]


@pytest.mark.parametrize(
    ("scores", "beam_width", "candidates_per_parent"),
    [
        ([-0.5, -0.1, -0.3], 2, 3),
        ([-0.1, -0.2], 2, 2),
        ([-float("inf")] * 4, 2, 2),
        ([-0.1, -float("inf"), -0.2, -float("inf")], 2, 2),
    ],
)  # type: ignore[misc]
def test_selected_beams_keep_legacy_argpartition_order_and_geometry(
    scores: list[float],
    beam_width: int,
    candidates_per_parent: int,
) -> None:
    token_ids = list(range(100, 100 + len(scores)))
    parent_count = len(scores) // candidates_per_parent
    parents = [_beam([10 + index]) for index in range(parent_count)]
    cache = [
        _candidate_cache(
            token_ids[
                parent_index * candidates_per_parent : (parent_index + 1) * candidates_per_parent
            ],
            scores[
                parent_index * candidates_per_parent : (parent_index + 1) * candidates_per_parent
            ],
        )
        for parent_index in range(parent_count)
    ]
    score_array = np.asarray(scores)
    if len(scores) > beam_width:
        expected_indices = np.argpartition(np.negative(score_array), beam_width)[:beam_width]
    else:
        expected_indices = np.arange(len(scores))
    initial_logprobs = FlatLogprobs()

    new_beams, fork_info = materialize_selected_beams(
        parents=parents,
        selected_indices=expected_indices,
        token_ids=np.asarray(token_ids),
        scores=score_array,
        beam_flat_cache=cache,
        candidates_per_parent=candidates_per_parent,
        initial_logprobs=initial_logprobs,
    )

    assert len(new_beams) == min(len(scores), beam_width)
    assert len(fork_info) == len(new_beams)
    assert fork_info == [
        (
            int(index) // candidates_per_parent,
            token_ids[int(index)],
        )
        for index in expected_indices
    ]
    assert [beam.tokens[-1] for beam in new_beams] == [
        token_ids[int(index)] for index in expected_indices
    ]
    assert [beam.cum_logprob for beam in new_beams] == [
        scores[int(index)] for index in expected_indices
    ]
    assert [beam._lp_parent for beam in new_beams] == [
        parents[int(index) // candidates_per_parent] for index in expected_indices
    ]
    assert [beam._lp_step_data for beam in new_beams] == [
        cache[int(index) // candidates_per_parent] for index in expected_indices
    ]
    assert all(beam.logprobs is initial_logprobs for beam in new_beams)


def test_selected_beam_rejects_missing_parent_candidate_cache() -> None:
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


def test_single_eos_is_completed_and_masked_before_active_selection() -> None:
    parent = _beam([10], -0.5)
    cache = [_candidate_cache([2, 11], [-0.1, -0.3])]

    completed, active, fork_info, selection_scores = _run_legacy_decision_step(
        parents=[parent],
        token_ids=[2, 11],
        scores=[-0.6, -0.8],
        beam_flat_cache=cache,
        candidates_per_parent=2,
        beam_width=1,
    )

    assert selection_scores.tolist() == [-float("inf"), -0.8]
    assert len(completed) == 1
    assert completed[0].tokens == [10, 2]
    assert completed[0].finish_reason == "stop"
    assert completed[0].stop_reason == 2
    assert completed[0]._lp_parent is parent
    assert completed[0]._lp_step_data is cache[0]
    assert [beam.tokens for beam in active] == [[10, 11]]
    assert fork_info == [(0, 11)]


def test_eos_completion_returns_masked_score_copy() -> None:
    parent = _beam([10], -0.5)
    cache = [_candidate_cache([2, 11], [-0.1, -0.3])]
    scores = np.asarray([-0.6, -0.8])
    initial_logprobs = FlatLogprobs()

    completed, masked_scores = complete_eos_candidates(
        parents=[parent],
        token_ids=np.asarray([2, 11]),
        scores=scores,
        beam_flat_cache=cache,
        candidates_per_parent=2,
        eos_token_id=2,
        initial_logprobs=initial_logprobs,
        include_eos_token=True,
        inherit_lora_request=True,
    )

    assert masked_scores is not scores
    assert scores.tolist() == [-0.6, -0.8]
    assert masked_scores.tolist() == [-float("inf"), -0.8]
    assert parent.tokens == [10]
    assert completed[0].cum_logprob == -0.6
    assert completed[0].logprobs is initial_logprobs


def test_multiple_parents_complete_eos_with_parent_candidate_cache() -> None:
    parents = [_beam([10]), _beam([20])]
    cache = [
        _candidate_cache([2, 11], [-0.1, -0.5]),
        _candidate_cache([2, 21], [-0.2, -0.4]),
    ]

    completed, active, fork_info, selection_scores = _run_legacy_decision_step(
        parents=parents,
        token_ids=[2, 11, 2, 21],
        scores=[-0.1, -0.5, -0.2, -0.4],
        beam_flat_cache=cache,
        candidates_per_parent=2,
        beam_width=2,
    )

    assert selection_scores.tolist() == [
        -float("inf"),
        -0.5,
        -float("inf"),
        -0.4,
    ]
    assert [beam.tokens for beam in completed] == [[10, 2], [20, 2]]
    assert completed[0]._lp_step_data is cache[0]
    assert completed[1]._lp_step_data is cache[1]
    expected_indices = np.argpartition(np.asarray([np.inf, 0.5, np.inf, 0.4]), 2)[:2]
    assert [beam.tokens[-1] for beam in active] == [
        [2, 11, 2, 21][int(index)] for index in expected_indices
    ]
    assert fork_info == [
        (int(index) // 2, [2, 11, 2, 21][int(index)]) for index in expected_indices
    ]


def test_ignore_eos_keeps_eos_as_an_ordinary_active_token() -> None:
    parent = _beam([10])
    cache = [_candidate_cache([2, 11], [-0.1, -0.3])]

    completed, active, fork_info, selection_scores = _run_legacy_decision_step(
        parents=[parent],
        token_ids=[2, 11],
        scores=[-0.1, -0.3],
        beam_flat_cache=cache,
        candidates_per_parent=2,
        beam_width=1,
        ignore_eos=True,
    )

    assert completed == []
    assert selection_scores.tolist() == [-0.1, -0.3]
    assert [beam.tokens for beam in active] == [[10, 2]]
    assert fork_info == [(0, 2)]


def test_masked_eos_can_remain_the_all_negative_infinity_fallback() -> None:
    parent = _beam([10])
    cache = [_candidate_cache([2, 11], [-0.1, -float("inf")])]

    completed, active, fork_info, selection_scores = _run_legacy_decision_step(
        parents=[parent],
        token_ids=[2, 11],
        scores=[-0.1, -float("inf")],
        beam_flat_cache=cache,
        candidates_per_parent=2,
        beam_width=1,
    )

    expected_indices = np.argpartition(np.negative(selection_scores), 1)[:1]
    assert expected_indices.tolist() == [0]
    assert len(completed) == 1
    assert [beam.tokens for beam in active] == [[10, 2]]
    assert active[0].cum_logprob == -float("inf")
    assert fork_info == [(0, 2)]


def test_all_catalog_masked_candidates_keep_grouped_beam_geometry() -> None:
    parents = [_beam([10]), _beam([20])]
    token_ids = [11, 12, 21, 22]
    scores = [-float("inf")] * 4
    cache = [
        _candidate_cache([11, 12], scores[:2]),
        _candidate_cache([21, 22], scores[2:]),
    ]

    completed, active, fork_info, selection_scores = _run_legacy_decision_step(
        parents=parents,
        token_ids=token_ids,
        scores=scores,
        beam_flat_cache=cache,
        candidates_per_parent=2,
        beam_width=2,
        ignore_eos=True,
    )

    expected_indices = np.argpartition(np.negative(selection_scores), 2)[:2]
    assert completed == []
    assert len(active) == 2
    assert len(fork_info) == len(active)
    assert fork_info == [(int(index) // 2, token_ids[int(index)]) for index in expected_indices]
    assert [beam.tokens[-1] for beam in active] == [
        token_ids[int(index)] for index in expected_indices
    ]
    assert all(beam.cum_logprob == -float("inf") for beam in active)
