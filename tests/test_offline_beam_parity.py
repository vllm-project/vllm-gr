# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.entrypoints.gr import _parse_step_logprobs
from vllm_gr.utils.catalog import Catalog


def _beam(
    tokens: list[int],
    score: float = 0.0,
    catalog_node: dict[int | str, object] | None = None,
) -> BeamSearchSequence:
    beam = BeamSearchSequence(
        orig_prompt=tokens_input(tokens),
        tokens=tokens,
        logprobs=FlatLogprobs(),
        cum_logprob=score,
    )
    beam._lp_parent = None
    beam._lp_step_data = None
    if catalog_node is not None:
        beam._catalog_node = catalog_node
    return beam


def _catalog() -> Catalog:
    """Real Catalog over paths [10,3], [10,5], [8,9]; valid([10]) == {3, 5}."""
    catalog = Catalog.__new__(Catalog)
    catalog._TERMINAL_MARKER = Catalog._TERMINAL_MARKER
    catalog.trie = {
        10: {
            3: {Catalog._TERMINAL_MARKER: True},
            5: {Catalog._TERMINAL_MARKER: True},
        },
        8: {9: {Catalog._TERMINAL_MARKER: True}},
    }
    return catalog


# Candidate window [2, 3, 4, 5] with logprobs [-0.1, -0.2, -0.3, -0.4].
# Catalog-valid tokens are {3, 5}.  With fork_info_j=None the single-beam
# (else) branch of _parse_step_logprobs is exercised.
_CANDIDATE_TIDS = [2, 3, 4, 5]
_CANDIDATE_LPS = [-0.1, -0.2, -0.3, -0.4]
# Beam cum_logprob = -0.5, so all_beams_logprob = -0.5 + filtered logprob.
_BEAM_CUM = -0.5


def _flat(tids: list[int], lps: list[float]) -> SimpleNamespace:
    # Single-beam FlatLogprobs shape: one contiguous block covering the whole
    # candidate window, so start_indices=[0] and end_indices=[len(tids)].
    return SimpleNamespace(
        token_ids=tids,
        logprobs=lps,
        ranks=None,
        decoded_tokens=None,
        start_indices=[0],
        end_indices=[len(tids)],
    )


def _parse_with_catalog(
    instance: SimpleNamespace,
    catalog: Catalog,
    strategy: str,
) -> tuple[bool, list[int], list[float], list[int], list[Any]]:
    # _parse_step_logprobs is an untyped vllm helper; the return shape is
    # (is_error, token_ids, scores, beam_idx, beam_flat_cache).
    return _parse_step_logprobs(  # type: ignore[no-any-return]
        result=_result_with_flat(_flat(_CANDIDATE_TIDS, _CANDIDATE_LPS)),
        instance=instance,
        token=0,
        logprobs_num=len(_CANDIDATE_TIDS),
        fork_info_j=None,
        catalog=catalog,
        catalog_strategy=strategy,
    )


def _result_with_flat(flat: object) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(finish_reason=None, logprobs=flat)],
    )


def test_offline_grouped_candidates_keep_explicit_parent_indices() -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], -0.5), _beam([20], -0.2)],
        prompt_len=1,
    )
    flat = SimpleNamespace(
        token_ids=[7, 3, 7, 8, 4, 8, 999],
        logprobs=[-0.01, -0.1, -0.2, -0.02, -0.3, -0.4, -9.0],
        ranks=None,
        decoded_tokens=None,
    )

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=1,
        logprobs_num=2,
        fork_info_j=[(0, 7), (1, 8)],
        catalog=None,
        catalog_strategy="fill",
    )

    assert not is_error
    assert token_ids == [3, 7, 4, 8]
    assert scores == pytest.approx([-0.6, -0.7, -0.5, -0.6])
    assert beam_idx == [0, 0, 1, 1]
    assert [entry[0] for entry in cache] == [[3, 7], [4, 8]]
    assert 999 not in token_ids


# ---------------------------------------------------------------------------
# Offline catalog wiring: _parse_step_logprobs must route cached-node and
# fallback beams through Catalog.apply_catalog_filter for BOTH strategies.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "catalog_node",
    [
        # cached node: {3, 5} are the valid next tokens
        {3: {Catalog._TERMINAL_MARKER: True}, 5: {Catalog._TERMINAL_MARKER: True}},
        # fallback: no _catalog_node -> resolve_catalog_node calls catalog.valid()
        None,
    ],
)  # type: ignore[misc]
def test_parse_step_logprobs_fill_uses_catalog(
    catalog_node: dict[int | str, object] | None,
) -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], _BEAM_CUM, catalog_node=catalog_node)],
        prompt_len=0,
    )
    is_error, token_ids, scores, beam_idx, cache = _parse_with_catalog(instance, _catalog(), "fill")

    assert not is_error
    # invalid {2, 4} are filled with valid tokens {3, 5}
    assert token_ids == [3, 3, 5, 5]
    # cum_logprob is added to every filtered logprob, including injected ones
    # (injected lp is -10000.0 / -10000.1; in-window lp is -0.2 / -0.4).
    assert scores == pytest.approx(
        [_BEAM_CUM + -10000.0, _BEAM_CUM + -0.2, _BEAM_CUM + -10000.1, _BEAM_CUM + -0.4]
    )
    assert beam_idx == [0, 0, 0, 0]
    assert [entry[0] for entry in cache] == [[3, 3, 5, 5]]


@pytest.mark.parametrize(
    "catalog_node",
    [
        {3: {Catalog._TERMINAL_MARKER: True}, 5: {Catalog._TERMINAL_MARKER: True}},
        None,
    ],
)  # type: ignore[misc]
def test_parse_step_logprobs_null_uses_catalog(
    catalog_node: dict[int | str, object] | None,
) -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], _BEAM_CUM, catalog_node=catalog_node)],
        prompt_len=0,
    )
    is_error, token_ids, scores, beam_idx, cache = _parse_with_catalog(instance, _catalog(), "null")

    assert not is_error
    # invalid {2, 4} masked to -inf; geometry preserved
    assert token_ids == [2, 3, 4, 5]
    assert scores == pytest.approx(
        [-float("inf"), _BEAM_CUM + -0.2, -float("inf"), _BEAM_CUM + -0.4]
    )
    assert beam_idx == [0, 0, 0, 0]
    assert [entry[0] for entry in cache] == [[2, 3, 4, 5]]


# ---------------------------------------------------------------------------
# Grouped (multi-beam) branch: stride slicing, per-beam catalog nodes, and
# per-beam cum_logprob must all be wired correctly.  beam0 uses the cached
# node; beam1 falls back to catalog.valid().  Both share prompt_len=0.
# ---------------------------------------------------------------------------


def _grouped_instance() -> SimpleNamespace:
    return SimpleNamespace(
        beams=[
            # cached node: valid next tokens are {3, 5}
            _beam(
                [10],
                -0.5,
                catalog_node={
                    3: {Catalog._TERMINAL_MARKER: True},
                    5: {Catalog._TERMINAL_MARKER: True},
                },
            ),
            # fallback: no _catalog_node -> catalog.valid([8]) == {9}
            _beam([8], -0.3),
        ],
        prompt_len=0,
    )


# flat: beam0 window [2,3,4,5] (stride 0), beam1 window [9,10,11,12] (stride 1)
_GROUPED_FLAT = SimpleNamespace(
    token_ids=[2, 3, 4, 5, 9, 10, 11, 12],
    logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
    ranks=None,
    decoded_tokens=None,
)


@pytest.mark.parametrize(
    "strategy, expected_tids, expected_scores, expected_cache",
    [
        (
            "fill",
            [3, 3, 5, 5, 9, 9, 9, 9],
            # beam0: cum -0.5 over [3,3,5,5]; beam1: cum -0.3 over [9,9,9,9]
            [
                -0.5 + -10000.0,
                -0.5 + -0.2,
                -0.5 + -10000.1,
                -0.5 + -0.4,
                -0.3 + -0.5,
                -0.3 + -10000.0,
                -0.3 + -10000.0,
                -0.3 + -10000.0,
            ],
            [[3, 3, 5, 5], [9, 9, 9, 9]],
        ),
        (
            "null",
            [2, 3, 4, 5, 9, 10, 11, 12],
            [
                -float("inf"),
                -0.5 + -0.2,
                -float("inf"),
                -0.5 + -0.4,
                -0.3 + -0.5,
                -float("inf"),
                -float("inf"),
                -float("inf"),
            ],
            [[2, 3, 4, 5], [9, 10, 11, 12]],
        ),
    ],
)  # type: ignore[misc]
def test_parse_step_logprobs_grouped_mixes_cached_and_fallback(
    strategy: str,
    expected_tids: list[int],
    expected_scores: list[float],
    expected_cache: list[list[int]],
) -> None:
    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(_GROUPED_FLAT),
        instance=_grouped_instance(),
        token=1,
        logprobs_num=4,
        fork_info_j=[(0, 3), (1, 9)],
        catalog=_catalog(),
        catalog_strategy=strategy,
    )

    assert not is_error
    assert token_ids == expected_tids
    assert scores == pytest.approx(expected_scores)
    # beam0's 4 candidates map to b_idx 0; beam1's to b_idx 1.
    assert beam_idx == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [entry[0] for entry in cache] == expected_cache
