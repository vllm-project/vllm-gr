# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared model-independent transition for one beam-search generation step."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.logprobs import FlatLogprobs

from vllm_gr.beam_search_decision_utils import (
    BeamCandidateCache,
    complete_eos_candidates,
    extract_and_dedup_flat_logprobs,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.entrypoints.beam_search_config import BeamSearchStep


@dataclass
class BeamSearchStepResult:
    active: list[BeamSearchSequence]
    completed: list[BeamSearchSequence]
    fork_info: list[tuple[int, int]]


def apply_beam_search_step(
    parents: Sequence[BeamSearchSequence],
    outputs: Sequence[Any],
    step: BeamSearchStep,
    initial_logprobs: FlatLogprobs,
    *,
    allowed_token_ids_by_parent: Sequence[Collection[int]] | None = None,
    stop_token_ids: Collection[int] | None = None,
    include_stop_token: bool = True,
    catalog_strategy: str = "fill",
) -> BeamSearchStepResult:
    """Expand, score and prune one step from engine-produced logprobs."""

    if len(outputs) != len(parents):
        raise ValueError(
            "Beam search expected one engine output for every parent beam"
        )
    if (
        allowed_token_ids_by_parent is not None
        and len(allowed_token_ids_by_parent) != len(parents)
    ):
        raise ValueError(
            "Beam search expected one allowed-token set for every parent beam"
        )

    all_token_ids: list[int] = []
    all_scores: list[float] = []
    all_parent_indices: list[int] = []
    beam_flat_cache: list[BeamCandidateCache | None] = []

    for parent_index, (parent, output) in enumerate(zip(parents, outputs)):
        flat = output.outputs[0].logprobs
        if flat is None:
            raise ValueError("Beam search expected non-empty logprobs for every beam")
        token_ids, logprobs, ranks, decoded = extract_and_dedup_flat_logprobs(
            flat,
            step.expand_width,
        )
        if allowed_token_ids_by_parent is not None:
            allowed = allowed_token_ids_by_parent[parent_index]
            logprobs = [
                value if token_id in allowed else -float("inf")
                for token_id, value in zip(token_ids, logprobs)
            ]

        beam_flat_cache.append((token_ids, logprobs, ranks, decoded))
        all_token_ids.extend(token_ids)
        all_scores.extend(parent.cum_logprob + value for value in logprobs)
        all_parent_indices.extend([parent_index] * len(token_ids))

    token_ids_array = np.asarray(all_token_ids)
    scores_array = np.asarray(all_scores)
    parent_indices_array = np.asarray(all_parent_indices, dtype=np.intp)
    completed: list[BeamSearchSequence] = []

    if stop_token_ids:
        completed, scores_array = complete_eos_candidates(
            parents=parents,
            token_ids=token_ids_array,
            scores=scores_array,
            beam_idx=parent_indices_array,
            beam_flat_cache=beam_flat_cache,
            stop_token_ids=stop_token_ids,
            initial_logprobs=initial_logprobs,
            include_eos_token=include_stop_token,
            inherit_lora_request=True,
        )

    selected = select_top_indices(scores_array, step.keep_width, catalog_strategy)
    active, fork_info = materialize_selected_beams(
        parents=parents,
        selected_indices=selected,
        token_ids=token_ids_array,
        scores=scores_array,
        beam_idx=parent_indices_array,
        beam_flat_cache=beam_flat_cache,
        initial_logprobs=initial_logprobs,
    )
    return BeamSearchStepResult(active, completed, fork_info)


__all__ = ["BeamSearchStepResult", "apply_beam_search_step"]
