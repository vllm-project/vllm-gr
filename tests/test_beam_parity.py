# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest

from vllm_gr.beam_search_decision_utils import (
    BeamCandidateCache,
    complete_eos_candidates,
    extract_and_dedup_segment,
    mask_candidate_logprobs,
    materialize_selected_beams,
    select_top_indices,
)

NormalizedCache: TypeAlias = tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[int | None, ...] | None,
    tuple[str | None, ...] | None,
]


@dataclass(frozen=True)
class ParentCase:
    tokens: tuple[int, ...]
    score: float = 0.0
    lora_name: str | None = None


@dataclass(frozen=True)
class BeamDecisionCase:
    parents: tuple[ParentCase, ...]
    raw_token_ids: tuple[int, ...]
    raw_logprobs: tuple[float, ...]
    logprobs_num: int
    beam_width: int
    raw_ranks: tuple[int | None, ...] | None = None
    raw_decoded_tokens: tuple[str | None, ...] | None = None
    allowed_token_ids: tuple[frozenset[int] | None, ...] | None = None
    eos_token_id: int = 2
    ignore_eos: bool = False
    include_stop_str_in_output: bool = True
    missing_cache_index: int | None = None


@dataclass(frozen=True)
class NormalizedBeam:
    tokens: tuple[int, ...]
    score: float
    parent_index: int
    finish_reason: str | None
    stop_reason: int | str | None
    lora_name: str | None


@dataclass(frozen=True)
class NormalizedBeamDecisionResult:
    candidate_token_ids: tuple[int, ...]
    masked_logprobs: tuple[float, ...]
    eos_masked_scores: tuple[float, ...]
    selected_indices: tuple[int, ...]
    completed: tuple[NormalizedBeam, ...]
    active: tuple[NormalizedBeam, ...]
    candidate_cache: tuple[NormalizedCache | None, ...]
    fork_info: tuple[tuple[int, int], ...]
    ranking_input: tuple[tuple[float, int, bool], ...]


def _make_parents(case: BeamDecisionCase) -> list[BeamSearchSequence]:
    parents = []
    for index, parent_case in enumerate(case.parents):
        lora_request = (
            LoRARequest(parent_case.lora_name, index + 1, f"/tmp/{parent_case.lora_name}")
            if parent_case.lora_name is not None
            else None
        )
        parent = BeamSearchSequence(
            orig_prompt=tokens_input(list(parent_case.tokens)),
            tokens=list(parent_case.tokens),
            logprobs=FlatLogprobs(),
            cum_logprob=parent_case.score,
            lora_request=lora_request,
        )
        parent._lp_parent = None
        parent._lp_step_data = None
        parents.append(parent)
    return parents


def _parent_index(
    beam: BeamSearchSequence,
    parents: list[BeamSearchSequence],
) -> int:
    return next(index for index, parent in enumerate(parents) if beam._lp_parent is parent)


def _normalize_beam(
    beam: BeamSearchSequence,
    parents: list[BeamSearchSequence],
) -> NormalizedBeam:
    return NormalizedBeam(
        tokens=tuple(beam.tokens),
        score=beam.cum_logprob,
        parent_index=_parent_index(beam, parents),
        finish_reason=beam.finish_reason,
        stop_reason=beam.stop_reason,
        lora_name=beam.lora_request.lora_name if beam.lora_request is not None else None,
    )


def _normalize_cache(cache: BeamCandidateCache | None) -> NormalizedCache | None:
    if cache is None:
        return None
    token_ids, logprobs, ranks, decoded_tokens = cache
    return (
        tuple(token_ids),
        tuple(logprobs),
        tuple(ranks) if ranks is not None else None,
        tuple(decoded_tokens) if decoded_tokens is not None else None,
    )


def _run_decision_adapter(
    case: BeamDecisionCase,
    *,
    include_eos_token: bool,
    inherit_eos_lora_request: bool,
) -> NormalizedBeamDecisionResult:
    parents = _make_parents(case)
    parent_count = len(parents)
    stride_k = len(case.raw_token_ids) // parent_count
    all_token_ids: list[int] = []
    all_logprobs: list[float] = []
    all_scores: list[float] = []
    beam_flat_cache: list[BeamCandidateCache | None] = []

    for parent_index, parent in enumerate(parents):
        start_offset = parent_index * stride_k
        end_offset = (parent_index + 1) * stride_k
        token_ids, logprobs, ranks, decoded_tokens = extract_and_dedup_segment(
            case.raw_token_ids,
            case.raw_logprobs,
            case.raw_ranks,
            case.raw_decoded_tokens,
            start_offset,
            end_offset,
            case.logprobs_num,
        )
        allowed = (
            case.allowed_token_ids[parent_index] if case.allowed_token_ids is not None else None
        )
        masked_logprobs = mask_candidate_logprobs(token_ids, logprobs, allowed)
        cache: BeamCandidateCache = (
            token_ids,
            masked_logprobs,
            ranks,
            decoded_tokens,
        )
        beam_flat_cache.append(cache)
        all_token_ids.extend(token_ids)
        all_logprobs.extend(masked_logprobs)
        all_scores.extend(parent.cum_logprob + value for value in masked_logprobs)

    if case.missing_cache_index is not None:
        beam_flat_cache[case.missing_cache_index] = None

    token_array = np.asarray(all_token_ids)
    score_array = np.asarray(all_scores)
    completed: list[BeamSearchSequence] = []
    if case.ignore_eos:
        selection_scores = score_array
    else:
        beam_idx = np.arange(len(token_array)) // case.logprobs_num
        completed, selection_scores = complete_eos_candidates(
            parents=parents,
            token_ids=token_array,
            scores=score_array,
            beam_idx=beam_idx,
            beam_flat_cache=beam_flat_cache,
            candidates_per_parent=case.logprobs_num,
            eos_token_id=case.eos_token_id,
            initial_logprobs=FlatLogprobs(),
            include_eos_token=include_eos_token,
            inherit_lora_request=inherit_eos_lora_request,
        )

    selected_indices = select_top_indices(selection_scores, case.beam_width)
    active, fork_info = materialize_selected_beams(
        parents=parents,
        selected_indices=selected_indices,
        token_ids=token_array,
        scores=selection_scores,
        beam_flat_cache=beam_flat_cache,
        candidates_per_parent=case.logprobs_num,
        initial_logprobs=FlatLogprobs(),
    )
    ranking_beams = completed + active

    return NormalizedBeamDecisionResult(
        candidate_token_ids=tuple(all_token_ids),
        masked_logprobs=tuple(all_logprobs),
        eos_masked_scores=tuple(selection_scores.tolist()),
        selected_indices=tuple(int(index) for index in selected_indices),
        completed=tuple(_normalize_beam(beam, parents) for beam in completed),
        active=tuple(_normalize_beam(beam, parents) for beam in active),
        candidate_cache=tuple(_normalize_cache(cache) for cache in beam_flat_cache),
        fork_info=tuple(fork_info),
        ranking_input=tuple(
            (
                beam.cum_logprob,
                _parent_index(beam, parents),
                beam.finish_reason is not None,
            )
            for beam in ranking_beams
        ),
    )


def run_offline_decision_adapter(
    case: BeamDecisionCase,
) -> NormalizedBeamDecisionResult:
    return _run_decision_adapter(
        case,
        include_eos_token=True,
        inherit_eos_lora_request=True,
    )


def run_openai_decision_adapter(
    case: BeamDecisionCase,
) -> NormalizedBeamDecisionResult:
    return _run_decision_adapter(
        case,
        include_eos_token=case.include_stop_str_in_output,
        inherit_eos_lora_request=False,
    )


def _assert_shared_decision_parity(
    offline: NormalizedBeamDecisionResult,
    openai: NormalizedBeamDecisionResult,
) -> None:
    assert offline.candidate_token_ids == openai.candidate_token_ids
    assert offline.masked_logprobs == openai.masked_logprobs
    assert offline.eos_masked_scores == openai.eos_masked_scores
    assert offline.selected_indices == openai.selected_indices
    assert offline.active == openai.active
    assert offline.candidate_cache == openai.candidate_cache
    assert offline.fork_info == openai.fork_info
    assert offline.ranking_input == openai.ranking_input
    assert [
        (
            beam.score,
            beam.parent_index,
            beam.finish_reason,
            beam.stop_reason,
        )
        for beam in offline.completed
    ] == [
        (
            beam.score,
            beam.parent_index,
            beam.finish_reason,
            beam.stop_reason,
        )
        for beam in openai.completed
    ]


_DEFAULT_PARENT = ParentCase(tokens=(10,), score=-0.5)


@pytest.mark.parametrize(
    ("case", "expected_token_ids"),
    [
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(7, 3, 7),
                raw_logprobs=(-0.01, -0.1, -0.2),
                raw_ranks=(1, 2, 3),
                raw_decoded_tokens=("seven", "three", "seven"),
                logprobs_num=2,
                beam_width=2,
            ),
            (3, 7),
            id="sampled-token-present-in-top-k",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(7, 3, 4),
                raw_logprobs=(-0.01, -0.1, -0.2),
                raw_ranks=None,
                raw_decoded_tokens=None,
                logprobs_num=2,
                beam_width=2,
            ),
            (7, 3),
            id="sampled-token-absent-with-optional-metadata-none",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(7, 3),
                raw_logprobs=(-0.1, -0.2),
                allowed_token_ids=(frozenset({3}),),
                logprobs_num=2,
                beam_width=2,
            ),
            (7, 3),
            id="catalog-partial-mask",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(7, 3),
                raw_logprobs=(-0.1, -0.2),
                allowed_token_ids=(frozenset(),),
                logprobs_num=2,
                beam_width=2,
                ignore_eos=True,
            ),
            (7, 3),
            id="catalog-all-mask",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(2, 11),
                raw_logprobs=(-0.1, -0.3),
                logprobs_num=2,
                beam_width=1,
            ),
            (2, 11),
            id="single-eos",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(
                    ParentCase(tokens=(10,), score=-0.5),
                    ParentCase(tokens=(20,), score=-0.2),
                ),
                raw_token_ids=(2, 11, 2, 21),
                raw_logprobs=(-0.1, -0.3, -0.2, -0.4),
                logprobs_num=2,
                beam_width=2,
            ),
            (2, 11, 2, 21),
            id="multiple-parent-eos",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(2, 11),
                raw_logprobs=(-0.1, -0.3),
                logprobs_num=2,
                beam_width=1,
                ignore_eos=True,
            ),
            (2, 11),
            id="ignore-eos",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(11, 12),
                raw_logprobs=(-0.1, -0.2),
                logprobs_num=3,
                beam_width=3,
            ),
            (11, 12),
            id="candidate-count-below-beam-width",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(11, 12, 13, 14),
                raw_logprobs=(-np.inf, -np.inf, -np.inf, -np.inf),
                logprobs_num=4,
                beam_width=2,
            ),
            (11, 12, 13, 14),
            id="all-negative-infinity",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(11, 12, 13, 14),
                raw_logprobs=(-0.1, -np.inf, -0.3, -np.inf),
                logprobs_num=4,
                beam_width=2,
            ),
            (11, 12, 13, 14),
            id="finite-and-negative-infinity",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(_DEFAULT_PARENT,),
                raw_token_ids=(11, 12, 13, 14),
                raw_logprobs=(-0.1, -0.1, -0.1, -0.1),
                logprobs_num=4,
                beam_width=2,
            ),
            (11, 12, 13, 14),
            id="tied-scores",
        ),
        pytest.param(
            BeamDecisionCase(
                parents=(
                    ParentCase(tokens=(10,), score=-0.5),
                    ParentCase(tokens=(20,), score=-0.2),
                ),
                raw_token_ids=(7, 3, 7, 8, 4, 8, 999),
                raw_logprobs=(-0.01, -0.1, -0.2, -0.02, -0.3, -0.4, -9.0),
                raw_ranks=None,
                raw_decoded_tokens=None,
                logprobs_num=2,
                beam_width=2,
            ),
            (3, 7, 4, 8),
            id="grouped-floor-stride-ignores-remainder",
        ),
    ],
)  # type: ignore[misc]
def test_offline_and_openai_use_identical_production_decisions(
    case: BeamDecisionCase,
    expected_token_ids: tuple[int, ...],
) -> None:
    offline = run_offline_decision_adapter(case)
    openai = run_openai_decision_adapter(case)

    _assert_shared_decision_parity(offline, openai)
    assert offline.candidate_token_ids == expected_token_ids

    expected_scores = np.asarray(offline.eos_masked_scores)
    expected_indices = select_top_indices(expected_scores, case.beam_width)
    assert offline.selected_indices == tuple(int(index) for index in expected_indices)


@pytest.mark.parametrize("include_stop_str_in_output", [True, False])  # type: ignore[misc]
def test_eos_output_policy_differs_without_changing_shared_decision(
    include_stop_str_in_output: bool,
) -> None:
    case = BeamDecisionCase(
        parents=(ParentCase(tokens=(10,), score=-0.5, lora_name="adapter"),),
        raw_token_ids=(2, 11),
        raw_logprobs=(-0.1, -0.2),
        logprobs_num=2,
        beam_width=1,
        include_stop_str_in_output=include_stop_str_in_output,
    )

    offline = run_offline_decision_adapter(case)
    openai = run_openai_decision_adapter(case)

    _assert_shared_decision_parity(offline, openai)
    assert offline.completed[0].tokens == (10, 2)
    assert openai.completed[0].tokens == ((10, 2) if include_stop_str_in_output else (10,))
    assert offline.completed[0].lora_name == "adapter"
    assert openai.completed[0].lora_name is None
    assert offline.active[0].lora_name == "adapter"
    assert openai.active[0].lora_name == "adapter"


@pytest.mark.parametrize(
    "raw_token_ids",
    [
        pytest.param((2,), id="eos-completion"),
        pytest.param((11,), id="child-materialization"),
    ],
)  # type: ignore[misc]
def test_both_adapters_preserve_missing_candidate_cache_failure(
    raw_token_ids: tuple[int, ...],
) -> None:
    case = BeamDecisionCase(
        parents=(_DEFAULT_PARENT,),
        raw_token_ids=raw_token_ids,
        raw_logprobs=(-0.1,),
        logprobs_num=1,
        beam_width=1,
        missing_cache_index=0,
    )

    with pytest.raises(AssertionError):
        run_offline_decision_adapter(case)
    with pytest.raises(AssertionError):
        run_openai_decision_adapter(case)
