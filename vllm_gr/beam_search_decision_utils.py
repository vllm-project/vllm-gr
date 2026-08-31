# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, allocation-light helpers for beam-search candidate decisions.

These helpers deliberately preserve the legacy NumPy data flow.  They operate
on the same lists and arrays as the original implementation and do not create
an intermediate beam domain model.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence

if TYPE_CHECKING:
    from vllm.logprobs import FlatLogprobs

from vllm_gr.ops.sliced_logits import get_vocab_slice_config, is_vocab_slice_enabled


class GlobalRunningStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.max_val = -np.inf  # Initialize to negative infinity

    def push(self, array):
        x = np.asanyarray(array).ravel()
        batch_n = x.size
        if batch_n == 0:
            return

        with self._lock:
            # 1. Update Maximum
            batch_max = np.max(x)
            if batch_max > self.max_val:
                self.max_val = batch_max

            # 2. Calculate Batch Stats for Variance
            batch_mean = np.mean(x)
            batch_M2 = np.sum((x - batch_mean) ** 2)

            # 3. Merge with Global Stats
            if self.n == 0:
                self.n = batch_n
                self.mean = batch_mean
                self.M2 = batch_M2
            else:
                delta = batch_mean - self.mean
                total_n = self.n + batch_n

                # Update mean
                self.mean += delta * batch_n / total_n

                # Update M2 (Chan et al. Formula)
                self.M2 += batch_M2 + (delta**2) * (self.n * batch_n / total_n)
                self.n = total_n

    @property
    def variance(self):
        with self._lock:
            return self.M2 / self.n if self.n > 0 else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def max(self):
        with self._lock:
            return self.max_val if self.n > 0 else None


_CREATE_STAT_LOCK = threading.Lock()


def create_stat(max_tokens: int) -> list[GlobalRunningStats]:
    """Return a thread-safe singleton list of GlobalRunningStats using function attribute."""
    with _CREATE_STAT_LOCK:
        if not hasattr(create_stat, "_stats"):
            create_stat._stats = []
        while len(create_stat._stats) < max_tokens:
            create_stat._stats.append(GlobalRunningStats())
        return create_stat._stats[:max_tokens]


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
    """Mask disallowed candidates without filtering or reordering them.

    Delegates to the shared catalog masking implementation so the Offline
    path cannot diverge from ``Catalog.apply_catalog_filter``.
    """
    if allowed_token_ids is None:
        return list(logprobs)
    from vllm_gr.utils.catalog import mask_logprobs_by_token_set

    return mask_logprobs_by_token_set(token_ids, logprobs, allowed_token_ids)


_EXCEPT_MASKIN = [0]


def mask_beam_star_candidates(
    scores: np.ndarray,
    indices: np.ndarray,
    stats: GlobalRunningStats,
    token_idx: int,
    token_idx_offset: int,
    coef: float = 3.3,
) -> np.ndarray:
    """Mask candidate indices whose scores fall strictly below the Beam* cutoff.

    The cutoff is ``S_max - coef * (token_idx + token_idx_offset) * std(S)``,
    where the offset is 2 when ``begin_token`` is defined (it occupies a
    sequence position ahead of the first generated token) and 1 otherwise.
    Callers compute ``token_idx_offset`` once, since it never changes during
    iteration.
    """
    if token_idx in _EXCEPT_MASKIN:
        return indices
    max_scores = scores[indices]
    finite_mask = max_scores > -10000.0
    if not np.any(finite_mask):
        return indices
    finite_scores = max_scores[finite_mask]
    stats.push(finite_scores)
    sigma = stats.std
    s_max = stats.max
    cutoff = s_max - coef * (token_idx + token_idx_offset) * sigma
    keep_mask = max_scores >= cutoff
    if not np.any(keep_mask):
        # return the best candidate
        best_idx = np.argmax(max_scores)
        return np.array([indices[best_idx]])
    result_idx = indices[keep_mask]
    return result_idx


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
        if not np.isfinite(scores[candidate_index]):
            continue
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


def select_top_indices(scores: np.ndarray, beam_width: int) -> np.ndarray:
    """Return top finite-score candidate indices, always at least one.

    ``-inf`` entries (catalog-invalid candidates) are always discarded.  If
    every entry is ``-inf``, fall back to the single best index so that the
    beam keeps one candidate alive instead of being eliminated.
    """
    if scores.size == 0:
        return np.empty(0, dtype=np.intp)
    finite_mask = np.isfinite(scores)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        # No finite candidates — keep the beam alive with one index anyway.
        return np.array([int(np.argmax(scores))], dtype=np.intp)
    if finite_idx.size > beam_width:
        # argpartition requires kth < arr.size, so only call it when
        # there are strictly more candidates than we need to keep.
        topn_idx = finite_idx[np.argpartition(-scores[finite_idx], beam_width)[:beam_width]]
    else:
        topn_idx = finite_idx  # at most beam_width finite entries; take all
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
