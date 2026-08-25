# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Frontend parsing of Constraint Table candidates (issue #246).

The Worker overwrites the sampler output with the constrained top-k operator's
fixed-width rows, encoding only structural padding as ``-inf``;
``_parse_step_logprobs`` must slice them per parent, retain every finite legal
score regardless of magnitude, and never dedup.
"""

from types import SimpleNamespace

import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.constraints import BeamConstraintCoordinator
from vllm_gr.constraints.strategies import ConstraintTableStrategy
from vllm_gr.entrypoints.gr import _parse_step_logprobs

INVALID_LOGIT = -1.0e20


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


def _coordinator() -> BeamConstraintCoordinator:
    return BeamConstraintCoordinator(ConstraintTableStrategy("catalog.json"))


def _result_with_flat(flat: object) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(finish_reason=None, logprobs=flat)],
    )


def test_constrained_decode_parses_fixed_width_rows_per_parent() -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], -0.5), _beam([20], -0.2)],
        prompt_len=1,
    )
    # W=2 parents: parent0 block [11, padded], parent1 block [12, 13].
    flat = SimpleNamespace(
        token_ids=[11, 0, 12, 13],
        logprobs=[-0.1, -float("inf"), -0.2, -0.3],
        ranks=None,
        decoded_tokens=None,
    )

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=1,
        logprobs_num=2,
        fork_info_j=[(0, 11), (1, 12)],
        catalog=None,
        catalog_strategy="fill",
        constraint_coordinator=_coordinator(),
    )

    assert not is_error
    # Padding dropped; no dedup ran; parent mapping explicit.
    assert token_ids == [11, 12, 13]
    assert scores == pytest.approx([-0.6, -0.4, -0.5])
    assert beam_idx == [0, 1, 1]
    assert [entry[0] for entry in cache] == [[11], [12, 13]]


def test_constrained_decode_empty_valid_parent_contributes_no_candidates() -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], -0.5), _beam([20], -0.2)],
        prompt_len=1,
    )
    # parent0 has no valid continuation; parent1 yields [12, 13].
    flat = SimpleNamespace(
        token_ids=[0, 0, 12, 13],
        logprobs=[-float("inf"), -float("inf"), -0.2, -0.3],
        ranks=None,
        decoded_tokens=None,
    )

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=1,
        logprobs_num=2,
        fork_info_j=[(0, 0), (1, 12)],
        catalog=None,
        catalog_strategy="fill",
        constraint_coordinator=_coordinator(),
    )

    assert not is_error
    assert token_ids == [12, 13]
    assert beam_idx == [1, 1]
    assert [entry[0] for entry in cache] == [[], [12, 13]]


def test_constrained_prefill_parses_wide_segment_no_dedup() -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], -0.5)],
        prompt_len=1,
    )
    # W=3 prefill flat: 3 valid candidates then padding.
    flat = SimpleNamespace(
        token_ids=[31, 32, 33, 0, 0, 0],
        logprobs=[-0.1, -0.2, -0.3, -float("inf"), -float("inf"), -float("inf")],
        ranks=None,
        decoded_tokens=None,
        start_indices=[0],
        end_indices=[6],
    )

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=3,
        fork_info_j=None,
        catalog=None,
        catalog_strategy="fill",
        constraint_coordinator=_coordinator(),
    )

    assert not is_error
    assert token_ids == [31, 32, 33]
    assert scores == pytest.approx([-0.6, -0.7, -0.8])
    assert beam_idx == [0, 0, 0]
    assert list(cache[0][0]) == [31, 32, 33]


def test_constrained_decode_keeps_legal_score_at_legacy_sentinel() -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], 0.0)],
        prompt_len=1,
    )
    flat = SimpleNamespace(
        token_ids=[11, 12],
        logprobs=[0.0, INVALID_LOGIT],
        ranks=None,
        decoded_tokens=None,
    )

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=1,
        logprobs_num=2,
        fork_info_j=[(0, 11)],
        catalog=None,
        catalog_strategy="fill",
        constraint_coordinator=_coordinator(),
    )

    assert not is_error
    assert token_ids == [11, 12]
    assert scores == [0.0, INVALID_LOGIT]
    assert beam_idx == [0, 0]
    assert [entry[0] for entry in cache] == [[11, 12]]
