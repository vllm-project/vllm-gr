# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, allocation-light helpers for beam-search candidate decisions.

These helpers deliberately preserve the legacy NumPy data flow.  They operate
on the same lists and arrays as the original implementation and do not create
an intermediate beam domain model.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Iterator, Sequence
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
            self.max_val = max(self.max_val, batch_max)

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


class ChainedTokens(list[int]):
    """Zero-copy list view that prefixes a shared prompt without allocating full-length copies."""

    __slots__ = ("_len_p", "_len_s", "_prefix", "_suffix")

    def __init__(self, prefix: list[int], suffix: list[int]):
        super().__init__()
        self._prefix = prefix
        self._suffix = suffix
        self._len_p = len(prefix)
        self._len_s = len(suffix)

    def __len__(self) -> int:
        return self._len_p + self._len_s

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, slice):
            start = idx.start or 0
            if start < 0:
                start += self._len_p + self._len_s
            if start >= self._len_p:
                stop = idx.stop
                if stop is not None and stop < 0:
                    stop += self._len_p + self._len_s
                if stop is None:
                    return self._suffix[start - self._len_p :]
                return self._suffix[start - self._len_p : stop - self._len_p]
            return (self._prefix + self._suffix)[idx]
        if idx < 0:
            idx += self._len_p + self._len_s
        if idx < self._len_p:
            return self._prefix[idx]
        return self._suffix[idx - self._len_p]

    def append(self, x: int) -> None:
        self._suffix.append(x)
        self._len_s += 1

    def __iter__(self) -> Iterator[int]:
        yield from self._prefix
        yield from self._suffix

    def __repr__(self) -> str:
        return f"ChainedTokens(prefix_len={self._len_p}, suffix={self._suffix})"

    # Subclassing list[int] with an empty underlying C-level list (elements
    # live in _prefix/_suffix instead) means the inherited list.__eq__ /
    # list.__contains__ / list.__add__ would silently compare against that
    # empty list rather than the real content -- no current caller in this
    # module hits that (everything here uses manual indexing and the
    # explicit ChainedTokens(...) constructor instead), but exposing a
    # ChainedTokens beyond this module without these would violate the
    # standard sequence contract, so implement them for real.
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (list, tuple)):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    def __contains__(self, item: Any) -> bool:
        return item in self._prefix or item in self._suffix

    def __add__(self, other: Any) -> list[int]:
        return list(self) + list(other)


def complete_eos_candidates(
    *,
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_idx: np.ndarray,
    parents: Sequence[BeamSearchSequence],
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    initial_logprobs: FlatLogprobs,
    stop_token_ids: Collection[int],
    include_eos_token: bool = True,
    inherit_lora_request: bool = True,
) -> tuple[list[BeamSearchSequence], np.ndarray]:
    """Materialize stop-token candidates and mask their scores on a copy."""

    eos_beams: list[BeamSearchSequence] = []
    eos_indices = np.where(np.isin(token_ids, tuple(stop_token_ids)))[0]
    if eos_indices.size == 0:
        return eos_beams, scores
    for candidate_index in eos_indices:
        if not np.isfinite(scores[candidate_index]):
            continue
        parent_index = int(beam_idx[candidate_index])
        parent = parents[parent_index]
        cached = beam_flat_cache[parent_index]
        assert cached is not None
        stop_token_id = int(token_ids[candidate_index])
        p_tok = parent.tokens
        if include_eos_token:
            if isinstance(p_tok, ChainedTokens):
                eos_tok = ChainedTokens(p_tok._prefix, p_tok._suffix + [stop_token_id])
            elif len(p_tok) > 64:
                eos_tok = ChainedTokens(p_tok, [stop_token_id])
            else:
                eos_tok = p_tok + [stop_token_id]
        else:
            eos_tok = p_tok
        eos_beam = BeamSearchSequence(
            orig_prompt=parent.orig_prompt,
            tokens=eos_tok,
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

    If all candidates have non-finite scores, returns the single argmax index
    to match legacy behavior.
    """
    if scores.size == 0:
        return np.empty(0, dtype=np.intp)
    finite_idx = np.where(np.isfinite(scores))[0]
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
    if len(selected_indices) == 0 or len(parents) == 0:
        return new_beams, fork_info

    p_indices = beam_idx[selected_indices].tolist()
    t_ids = token_ids[selected_indices].tolist()
    scs = scores[selected_indices].tolist()
    orig_prompt = parents[0].orig_prompt
    lora_req = parents[0].lora_request

    for p_idx, token_id, sc in zip(p_indices, t_ids, scs):
        parent = parents[p_idx]
        cached = beam_flat_cache[p_idx]
        assert cached is not None
        p_tok = parent.tokens
        if isinstance(p_tok, ChainedTokens):
            new_tokens = ChainedTokens(p_tok._prefix, p_tok._suffix + [token_id])
        elif len(p_tok) > 64:
            new_tokens = ChainedTokens(p_tok, [token_id])
        else:
            new_tokens = p_tok + [token_id]

        new_beam = BeamSearchSequence(
            orig_prompt=orig_prompt,
            tokens=new_tokens,
            logprobs=initial_logprobs,
            lora_request=lora_req,
            cum_logprob=float(sc),
        )
        new_beam._lp_parent = parent
        new_beam._lp_step_data = cached
        new_beams.append(new_beam)
        fork_info.append((p_idx, token_id))

    return new_beams, fork_info
