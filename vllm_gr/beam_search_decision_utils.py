# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, allocation-light helpers for beam-search candidate decisions.

These helpers deliberately preserve the legacy NumPy data flow.  They operate
on the same lists and arrays as the original implementation and do not create
an intermediate beam domain model.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence

if TYPE_CHECKING:
    from vllm.logprobs import FlatLogprobs

from vllm_gr.logprobs import reconstruct_beam_logprobs
from vllm_gr.ops.sliced_logits import get_vocab_slice_config, is_vocab_slice_enabled

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

    if is_vocab_slice_enabled():
        offset, slice_size, _ = get_vocab_slice_config()
        if offset is not None and slice_size is not None and slice_size > 0:
            token_ids_pos = [
                tid + offset if 0 <= tid < slice_size else tid for tid in token_ids_pos
            ]

    return token_ids_pos, logprobs_pos, ranks_pos, decoded_pos


def extract_and_dedup_flat_logprobs(flat: Any, logprobs_num: int):
    """Extract position-0 logprobs from FlatLogprobs, dedup sampled token.

    Flat arrays store [sampled, top1, ..., topK] — the only possible
    duplicate is position 0 (sampled token) when it also appears in the
    top-K (positions 1..K).  Returns exactly ``logprobs_num`` entries.
    """
    return extract_and_dedup_segment(
        flat.token_ids,
        flat.logprobs,
        flat.ranks,
        flat.decoded_tokens,
        flat.start_indices[0],
        flat.end_indices[0],
        logprobs_num,
    )


def find_eos_indices(token_ids: np.ndarray, eos_token_id: int) -> np.ndarray:
    """Return EOS positions using the legacy NumPy comparison."""

    return np.where(token_ids == eos_token_id)[0]


def complete_eos_candidates(
    *,
    parents: Sequence[BeamSearchSequence],
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_idx: np.ndarray,
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    stop_token_ids: Collection[int],
    initial_logprobs: FlatLogprobs,
    include_eos_token: bool,
    inherit_lora_request: bool,
) -> tuple[list[BeamSearchSequence], np.ndarray]:
    """Materialize stop-token candidates and mask their scores on a copy."""

    eos_beams: list[BeamSearchSequence] = []
    eos_indices = np.where(np.isin(token_ids, tuple(stop_token_ids)))[0]
    for candidate_index in eos_indices:
        parent_index = int(beam_idx[candidate_index])
        parent = parents[parent_index]
        cached = beam_flat_cache[parent_index]
        assert cached is not None
        stop_token_id = int(token_ids[candidate_index])
        eos_beam = BeamSearchSequence(
            orig_prompt=parent.orig_prompt,
            tokens=(parent.tokens + [stop_token_id] if include_eos_token else parent.tokens),
            logprobs=initial_logprobs,
            cum_logprob=float(scores[candidate_index]),
            finish_reason="stop",
            stop_reason=stop_token_id,
            lora_request=parent.lora_request if inherit_lora_request else None,
        )
        eos_beam._lp_parent = parent
        eos_beam._lp_step_data = cached
        eos_beams.append(eos_beam)

    masked_scores = scores.copy()
    masked_scores[eos_indices] = -np.inf
    return eos_beams, masked_scores


def select_top_indices(
    scores: np.ndarray, beam_width: int, catalog_strategy: str = "fill"
) -> np.ndarray:
    """Return top candidate indices with the legacy ``argpartition`` order."""
    # For "null" strategy: only pick from finite (valid catalog) entries so that
    # -inf slots from catalog-filtered tokens are never selected.
    if catalog_strategy == "null":
        finite_mask = np.isfinite(scores)
        finite_idx = np.where(finite_mask)[0]
        if finite_idx.size == 0:
            # No valid candidates left — stop generating for this instance
            return np.empty(0, dtype=np.intp)
        if finite_idx.size > beam_width:
            # argpartition requires kth < arr.size, so only call it when
            # there are strictly more candidates than we need to keep.
            topn_idx = finite_idx[np.argpartition(-scores[finite_idx], beam_width)[:beam_width]]
        else:
            topn_idx = finite_idx  # at most beam_width finite entries; take all
    elif scores.size > beam_width:
        topn_idx = np.argpartition(np.negative(scores), beam_width)[:beam_width]
    else:
        topn_idx = np.arange(scores.size)
    return topn_idx


def materialize_selected_beams(
    *,
    parents: Sequence[BeamSearchSequence],
    selected_indices: Sequence[int],
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_idx: np.ndarray,
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    initial_logprobs: FlatLogprobs,
) -> tuple[list[BeamSearchSequence], ForkInfo]:
    """Build selected child beams and their grouped-request fork mapping."""

    new_beams: list[BeamSearchSequence] = []
    fork_info: ForkInfo = []
    for candidate_index in selected_indices:
        parent_index = int(beam_idx[candidate_index])
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


class WaveDecisionResult(NamedTuple):
    new_beams: list[BeamSearchSequence]
    fork_info: ForkInfo
    new_completed_eos: list[BeamSearchSequence]


def execute_wave_decision(
    *,
    all_beams: Sequence[BeamSearchSequence],
    per_beam_candidates: Sequence[BeamCandidateCache | None],
    catalog: Any | None,
    catalog_strategy: str,
    valid_tokens_sets: Sequence[Collection[int]] | None,
    beam_width: int,
    eos_token_id: int | None,
    sid_end_token_id: int | None,
    ignore_eos: bool,
    include_stop_str_in_output: bool,
    initial_logprobs: FlatLogprobs,
) -> WaveDecisionResult:
    """Consolidated wave candidate filtering, score summation, EOS, and Top-K selection."""
    all_beams_token_id: list[int] = []
    all_beams_logprob: list[float] = []
    all_beams_beam_idx: list[int] = []
    beam_flat_cache: list[BeamCandidateCache | None] = []

    for b_idx, (current_beam, cached) in enumerate(zip(all_beams, per_beam_candidates)):
        if cached is None:
            beam_flat_cache.append(None)
            continue
        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = cached
        # Catalog filtering only applies when both are present: the caller
        # computes valid_tokens_sets exclusively from the catalog it passes
        # (the frontend passes its serving catalog, the engine session its
        # engine-side trie), so the pair is always consistent.
        if valid_tokens_sets is not None and catalog is not None:
            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = catalog.apply_catalog_filter(
                token_ids_pos,
                logprobs_pos,
                ranks_pos,
                decoded_pos,
                valid_tokens_sets[b_idx],
                catalog_strategy,
            )
        beam_flat_cache.append((token_ids_pos, logprobs_pos, ranks_pos, decoded_pos))
        all_beams_token_id.extend(token_ids_pos)
        all_beams_logprob.extend(current_beam.cum_logprob + lp for lp in logprobs_pos)
        all_beams_beam_idx.extend([b_idx] * len(token_ids_pos))

    all_beams_token_id_arr = np.array(all_beams_token_id)
    all_beams_logprob_arr = np.array(all_beams_logprob)
    all_beams_beam_idx_arr = np.array(all_beams_beam_idx, dtype=np.intp)

    eos_beams: list[BeamSearchSequence] = []
    if not ignore_eos:
        stop_token_ids = set()
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)
        if sid_end_token_id is not None:
            stop_token_ids.add(sid_end_token_id)
        eos_beams, all_beams_logprob_arr = complete_eos_candidates(
            parents=all_beams,
            token_ids=all_beams_token_id_arr,
            scores=all_beams_logprob_arr,
            beam_idx=all_beams_beam_idx_arr,
            beam_flat_cache=beam_flat_cache,
            stop_token_ids=stop_token_ids,
            initial_logprobs=initial_logprobs,
            include_eos_token=include_stop_str_in_output,
            inherit_lora_request=False,
        )

    topn_idx = select_top_indices(all_beams_logprob_arr, beam_width, catalog_strategy)
    new_beams, selected_fork_info = materialize_selected_beams(
        parents=all_beams,
        selected_indices=topn_idx,
        token_ids=all_beams_token_id_arr,
        scores=all_beams_logprob_arr,
        beam_idx=all_beams_beam_idx_arr,
        beam_flat_cache=beam_flat_cache,
        initial_logprobs=initial_logprobs,
    )
    return WaveDecisionResult(
        new_beams=new_beams,
        fork_info=selected_fork_info,
        new_completed_eos=eos_beams,
    )


def finalize_best_beams(
    *,
    all_beams: Sequence[BeamSearchSequence],
    completed: Sequence[BeamSearchSequence],
    beam_width: int,
    initial_logprobs: FlatLogprobs,
    sid_end_token_id: int | None,
) -> tuple[list[BeamSearchSequence], float]:
    """Appends SID tokens, sorts completed beams, reconstructs tree logprobs, and measures sort time."""
    if sid_end_token_id is not None:
        for beam in all_beams:
            beam.tokens.append(sid_end_token_id)
    all_completed = list(completed) + list(all_beams)

    sort_start = time.perf_counter()
    sorted_completed = sorted(all_completed, key=lambda x: x.cum_logprob, reverse=True)
    sort_time = time.perf_counter() - sort_start

    best_beams = sorted_completed[:beam_width]
    for beam in best_beams:
        reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)
    return best_beams, sort_time
