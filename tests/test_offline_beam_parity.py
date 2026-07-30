# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.entrypoints.gr import _parse_step_logprobs


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


def _result_with_flat(flat: object) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(finish_reason=None, logprobs=flat)],
    )


@pytest.mark.parametrize(
    ("raw_token_ids", "expected_token_ids", "expected_logprobs"),
    [
        ([7, 3, 7, 5], [3, 7, 5], [-0.1, -0.2, -0.3]),
        ([7, 3, 4, 5], [7, 3, 4], [-0.01, -0.1, -0.2]),
    ],
)  # type: ignore[misc]
def test_offline_initial_candidates_preserve_sampled_token_rule(
    raw_token_ids: list[int],
    expected_token_ids: list[int],
    expected_logprobs: list[float],
) -> None:
    instance = SimpleNamespace(beams=[_beam([99], -0.5)], prompt_len=0)
    flat = SimpleNamespace(
        start_indices=[0],
        end_indices=[4],
        token_ids=raw_token_ids,
        logprobs=[-0.01, -0.1, -0.2, -0.3],
        ranks=[1, 2, 3, 4],
        decoded_tokens=[None, "b", "c", "d"],
    )

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=3,
        fork_info_j=None,
        catalog=None,
        catalog_strategy="fill",
    )

    assert not is_error
    assert token_ids == expected_token_ids
    assert scores == pytest.approx([-0.5 + value for value in expected_logprobs])
    assert beam_idx == [0] * len(expected_token_ids)
    assert cache[0][0] == expected_token_ids


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


def test_offline_catalog_mask_does_not_replace_or_remove_candidates() -> None:
    instance = SimpleNamespace(beams=[_beam([99], -0.5)], prompt_len=0)
    flat = SimpleNamespace(
        start_indices=[0],
        end_indices=[3],
        token_ids=[7, 7, 3],
        logprobs=[-0.01, -0.1, -0.2],
        ranks=[1, 2, 3],
        decoded_tokens=["a", "b", "c"],
    )
    catalog_calls: list[list[int]] = []

    class _Catalog:
        def valid(self, generated_tokens: list[int]) -> set[int]:
            catalog_calls.append(generated_tokens)
            return {3}

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=2,
        fork_info_j=None,
        catalog=_Catalog(),
        catalog_strategy="null",
    )

    assert not is_error
    assert catalog_calls == [[99]]
    assert token_ids == [7, 3]
    assert scores == [-float("inf"), -0.7]
    assert beam_idx == [0, 0]
    assert cache[0][0] == [7, 3]
