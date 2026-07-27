# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, allocation-light helpers for beam-search candidate decisions.

These helpers deliberately preserve the legacy NumPy data flow.  They operate
on the same lists and arrays as the original implementation and do not create
an intermediate beam domain model.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import numpy as np
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence

if TYPE_CHECKING:
    from vllm.logprobs import FlatLogprobs

BeamCandidateCache = tuple[
    Sequence[int],
    Sequence[float],
    Sequence[int | None] | None,
    Sequence[str | None] | None,
]
ForkInfo = list[tuple[int, int]]


def mask_candidate_logprobs(
    token_ids: Sequence[int],
    logprobs: Sequence[float],
    allowed_token_ids: Collection[int] | None,
) -> list[float]:
    """Mask disallowed candidates without filtering or reordering them."""

    if allowed_token_ids is None:
        return list(logprobs)
    return [
        logprob if token_id in allowed_token_ids else -float("inf")
        for token_id, logprob in zip(token_ids, logprobs)
    ]


def extract_and_dedup_segment(
    raw_token_ids: Sequence[int],
    raw_logprobs: Sequence[float],
    raw_ranks: Sequence[int | None] | None,
    raw_decoded: Sequence[str | None] | None,
    start_offset: int,
    end_offset: int,
    logprobs_num: int,
) -> tuple[
    Sequence[int],
    Sequence[float],
    Sequence[int | None] | None,
    Sequence[str | None] | None,
]:
    """Extract one candidate block using the legacy sampled-token rule."""

    token_ids_pos = raw_token_ids[start_offset:end_offset]
    logprobs_pos = raw_logprobs[start_offset:end_offset]
    ranks_pos = raw_ranks[start_offset:end_offset] if raw_ranks is not None else None
    decoded_pos = raw_decoded[start_offset:end_offset] if raw_decoded is not None else None

    if len(token_ids_pos) > logprobs_num:
        if any(token_ids_pos[0] == token_ids_pos[j] for j in range(1, logprobs_num + 1)):
            selected = slice(1, logprobs_num + 1)
        else:
            selected = slice(logprobs_num)
        token_ids_pos = token_ids_pos[selected]
        logprobs_pos = logprobs_pos[selected]
        ranks_pos = ranks_pos[selected] if ranks_pos is not None else None
        decoded_pos = decoded_pos[selected] if decoded_pos is not None else None

    return token_ids_pos, logprobs_pos, ranks_pos, decoded_pos


def find_eos_indices(token_ids: np.ndarray, eos_token_id: int) -> np.ndarray:
    """Return EOS positions using the legacy NumPy comparison."""

    return np.where(token_ids == eos_token_id)[0]


def complete_eos_candidates(
    *,
    parents: Sequence[BeamSearchSequence],
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    candidates_per_parent: int,
    eos_token_id: int,
    initial_logprobs: FlatLogprobs,
    include_eos_token: bool,
    inherit_lora_request: bool,
) -> tuple[list[BeamSearchSequence], np.ndarray]:
    """Materialize EOS candidates and mask their scores on a copy."""

    eos_beams: list[BeamSearchSequence] = []
    eos_indices = find_eos_indices(token_ids, eos_token_id)
    for candidate_index in eos_indices:
        parent_index = candidate_index // candidates_per_parent
        parent = parents[parent_index]
        cached = beam_flat_cache[parent_index]
        assert cached is not None
        eos_beam = BeamSearchSequence(
            orig_prompt=parent.orig_prompt,
            tokens=(
                parent.tokens + [int(token_ids[candidate_index])]
                if include_eos_token
                else parent.tokens
            ),
            logprobs=initial_logprobs,
            cum_logprob=float(scores[candidate_index]),
            finish_reason="stop",
            stop_reason=eos_token_id,
            lora_request=parent.lora_request if inherit_lora_request else None,
        )
        eos_beam._lp_parent = parent
        eos_beam._lp_step_data = cached
        eos_beams.append(eos_beam)

    masked_scores = scores.copy()
    masked_scores[eos_indices] = -np.inf
    return eos_beams, masked_scores


def select_top_indices(scores: np.ndarray, beam_width: int) -> np.ndarray:
    """Return top candidate indices with the legacy ``argpartition`` order."""

    if scores.size > beam_width:
        return np.argpartition(np.negative(scores), beam_width)[:beam_width]
    return np.arange(scores.size)


def materialize_selected_beams(
    *,
    parents: Sequence[BeamSearchSequence],
    selected_indices: Sequence[int],
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    candidates_per_parent: int,
    initial_logprobs: FlatLogprobs,
) -> tuple[list[BeamSearchSequence], ForkInfo]:
    """Build selected child beams and their grouped-request fork mapping."""

    new_beams: list[BeamSearchSequence] = []
    fork_info: ForkInfo = []
    for candidate_index in selected_indices:
        parent_index = candidate_index // candidates_per_parent
        parent = parents[parent_index]
        token_id = int(token_ids[candidate_index])
        cached = beam_flat_cache[parent_index]
        assert cached is not None
        new_beam = BeamSearchSequence(
            orig_prompt=parent.orig_prompt,
            tokens=parent.tokens + [token_id],
            logprobs=initial_logprobs,
            lora_request=parent.lora_request,
            cum_logprob=float(scores[candidate_index]),
        )
        new_beam._lp_parent = parent
        new_beam._lp_step_data = cached
        new_beams.append(new_beam)
        fork_info.append((parent_index, token_id))

    return new_beams, fork_info
