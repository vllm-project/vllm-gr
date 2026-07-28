# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, allocation-light helpers for beam-search candidate decisions.

These helpers deliberately preserve the legacy NumPy data flow.  They operate
on the same lists and arrays as the original implementation and do not create
an intermediate beam domain model.
"""

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np

from vllm_gr.ops.sliced_logits import get_vocab_slice_config, is_vocab_slice_enabled


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
        offset, _, _ = get_vocab_slice_config()
        if offset is not None:
            token_ids_pos = [tid + offset for tid in token_ids_pos]

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


def select_top_indices(scores: np.ndarray, beam_width: int) -> np.ndarray:
    """Return top candidate indices with the legacy ``argpartition`` order."""

    if scores.size > beam_width:
        return np.argpartition(np.negative(scores), beam_width)[:beam_width]
    return np.arange(scores.size)
