# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tokenizer-free candidate filtering shared by Serving and EngineCore."""

from __future__ import annotations

from collections.abc import Sequence


def filter_candidates_by_token_ids(
    token_ids: Sequence[int],
    logprobs: Sequence[float],
    ranks: Sequence[int | None] | None,
    decoded_tokens: Sequence[str | None] | None,
    valid_token_ids: set[int],
    candidate_strategy: str,
) -> tuple[
    list[int],
    list[float],
    list[int | None] | None,
    list[str | None] | None,
]:
    """Apply the legacy catalog ``fill``/``null`` policy.

    The implementation intentionally preserves ``Catalog.apply_catalog_filter``
    ordering, scores, and metadata behavior.  It only removes the dependency on
    the tokenizer-backed ``Catalog`` object so EngineCore can use it.
    """

    token_ids_list = list(token_ids)
    logprobs_list = list(logprobs)
    ranks_list = list(ranks) if ranks is not None else None
    decoded_list = list(decoded_tokens) if decoded_tokens is not None else None

    if not valid_token_ids:
        return (
            [],
            [],
            ([] if ranks_list is not None else None),
            ([] if decoded_list is not None else None),
        )

    if candidate_strategy == "null":
        filtered_lps = logprobs_list[:]
        all_inf = True
        for index, token_id in enumerate(token_ids_list):
            if token_id not in valid_token_ids:
                filtered_lps[index] = -float("inf")
            else:
                all_inf = False
        if all_inf:
            return (
                [],
                [],
                ([] if ranks_list is not None else None),
                ([] if decoded_list is not None else None),
            )
        return token_ids_list, filtered_lps, ranks_list, decoded_list

    if candidate_strategy != "fill":
        raise ValueError(f"candidate_strategy must be 'fill' or 'null', got {candidate_strategy!r}")

    valid_list = sorted(valid_token_ids)
    new_tids = token_ids_list[:]
    new_lps = logprobs_list[:]
    new_ranks = ranks_list[: len(token_ids_list)] if ranks_list is not None else None
    new_decoded = decoded_list[: len(token_ids_list)] if decoded_list is not None else None
    if new_ranks is not None and len(new_ranks) < len(token_ids_list):
        new_ranks.extend([None] * (len(token_ids_list) - len(new_ranks)))
    if new_decoded is not None and len(new_decoded) < len(token_ids_list):
        new_decoded.extend([None] * (len(token_ids_list) - len(new_decoded)))

    fill_used = 0
    for index, token_id in enumerate(token_ids_list):
        if token_id in valid_token_ids:
            continue
        replacement = valid_list[fill_used % len(valid_list)]
        new_tids[index] = replacement
        new_lps[index] = -10000.0 - (fill_used % len(valid_list)) * 0.1
        fill_used += 1
        if new_ranks is not None:
            new_ranks[index] = None
        if new_decoded is not None:
            new_decoded[index] = None

    return new_tids, new_lps, new_ranks, new_decoded
