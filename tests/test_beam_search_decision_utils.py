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
    resolve_catalog_node,
    select_top_indices,
)
from vllm_gr.utils.catalog import Catalog

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


# ---------------------------------------------------------------------------
# O(1) catalog prefix-node propagation (_catalog_node)
# ---------------------------------------------------------------------------


def _catalog_trie() -> dict[int, dict[int | str, object]]:
    """Build a small trie mirroring tests/test_catalog.py's structure.

    [1,2,3,4], [1,2,3,5], [8,9].  Each node maps token_id -> child dict, and
    terminal nodes carry the Catalog._TERMINAL_MARKER sentinel.
    """
    return {
        1: {
            2: {
                3: {
                    4: {Catalog._TERMINAL_MARKER: True},
                    5: {Catalog._TERMINAL_MARKER: True},
                }
            }
        },
        8: {9: {Catalog._TERMINAL_MARKER: True}},
    }


def test_catalog_node_propagates_to_selected_beams() -> None:
    """fork: new beam's _catalog_node follows the token via O(1) get()."""
    root = _catalog_trie()
    parent = _beam([1])
    # a beam whose tokens end at [1] carries the node AFTER consuming '1',
    # i.e. root[1] — the set of valid next-token children.
    parent._catalog_node = root[1]
    cache = [_candidate_cache([2, 3], [-0.1, -0.2])]

    new_beams, _ = materialize_selected_beams(
        parents=[parent],
        selected_indices=np.asarray([0, 1], dtype=np.intp),
        token_ids=np.asarray([2, 3]),
        scores=np.asarray([-0.1, -0.2]),
        beam_idx=np.asarray([0, 0], dtype=np.intp),
        beam_flat_cache=cache,
        initial_logprobs=FlatLogprobs(),
    )

    # child [1,2] -> node {3: {...}}; child [1,3] -> node {} (not a valid prefix)
    assert new_beams[0]._catalog_node == {
        3: {4: {Catalog._TERMINAL_MARKER: True}, 5: {Catalog._TERMINAL_MARKER: True}}
    }
    assert new_beams[1]._catalog_node == {}
    assert parent._catalog_node is root[1]  # parent untouched


def test_catalog_node_propagates_to_eos_beams() -> None:
    """EOS beam: _catalog_node follows the token just like a normal fork."""
    root = _catalog_trie()
    parent = _beam([1])
    parent._catalog_node = root[1]
    cache = [_candidate_cache([2], [-0.1])]

    completed, _ = complete_eos_candidates(
        parents=[parent],
        token_ids=np.asarray([2]),
        scores=np.asarray([-0.1]),
        beam_idx=np.asarray([0], dtype=np.intp),
        beam_flat_cache=cache,
        stop_token_ids={2},
        initial_logprobs=FlatLogprobs(),
        include_eos_token=True,
        inherit_lora_request=False,
    )

    assert completed[0]._catalog_node == {
        3: {4: {Catalog._TERMINAL_MARKER: True}, 5: {Catalog._TERMINAL_MARKER: True}}
    }
    assert parent._catalog_node is root[1]  # parent untouched


def test_catalog_node_missing_parent_uses_empty_node() -> None:
    """A parent without _catalog_node yields None (caller falls back to valid())."""
    parent = _beam([1])
    cache = [_candidate_cache([2], [-0.1])]

    new_beams, _ = materialize_selected_beams(
        parents=[parent],
        selected_indices=np.asarray([0], dtype=np.intp),
        token_ids=np.asarray([2]),
        scores=np.asarray([-0.1]),
        beam_idx=np.asarray([0], dtype=np.intp),
        beam_flat_cache=cache,
        initial_logprobs=FlatLogprobs(),
    )

    assert new_beams[0]._catalog_node is None
    assert getattr(new_beams[0], "_catalog_node", None) is None


def test_mask_candidate_logprobs_accepts_trie_node_dict() -> None:
    """dict node passed directly (no set() allocation) behaves like the set."""
    node: dict[int | str, object] = {2: {}, 5: {}}
    result = mask_candidate_logprobs([1, 2, 3, 5], [-0.1, -0.2, -0.3, -0.4], node)

    assert result == [-float("inf"), -0.2, -float("inf"), -0.4]


def test_catalog_node_invalid_token_lands_on_empty_node() -> None:
    """Forking on a token absent from the node yields {} (beam dead-ends)."""
    root = _catalog_trie()
    parent = _beam([8])
    parent._catalog_node = root[8]
    cache = [_candidate_cache([99], [-0.1])]

    new_beams, _ = materialize_selected_beams(
        parents=[parent],
        selected_indices=np.asarray([0], dtype=np.intp),
        token_ids=np.asarray([99]),
        scores=np.asarray([-0.1]),
        beam_idx=np.asarray([0], dtype=np.intp),
        beam_flat_cache=cache,
        initial_logprobs=FlatLogprobs(),
    )

    assert new_beams[0]._catalog_node == {}


def test_resolve_catalog_node_uses_cached_node_without_fallback() -> None:
    """Optimized path: a beam with a cached node never calls catalog.valid()."""
    from unittest import mock

    from vllm_gr.utils.catalog import Catalog

    catalog = mock.MagicMock(spec=Catalog)
    catalog.valid.return_value = {99}

    parent = _beam([1, 2])
    parent._catalog_node = {3: {Catalog._TERMINAL_MARKER: True}, 5: {}}

    node = resolve_catalog_node(parent, catalog, prompt_len=2)

    assert node is parent._catalog_node
    catalog.valid.assert_not_called()


def test_resolve_catalog_node_falls_back_to_valid_when_node_missing() -> None:
    """Fallback: a beam without _catalog_node triggers the full re-traversal."""
    from unittest import mock

    from vllm_gr.utils.catalog import Catalog

    catalog = mock.MagicMock(spec=Catalog)
    catalog.valid.return_value = {99}

    parent = _beam([1, 2])
    # NOTE: no _catalog_node attached on purpose

    node = resolve_catalog_node(parent, catalog, prompt_len=2)

    assert node == {99}
    catalog.valid.assert_called_once_with(parent.tokens[2:])
