# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Derive structural candidate validity from canonical constraint tables."""

from __future__ import annotations

import torch


def constraint_candidate_counts(
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    *,
    current_step: int,
    vocab_size: int,
) -> torch.Tensor:
    """Return the legal candidate count for every flattened operator row."""

    rows = sequence_group.reshape(-1, sequence_group.shape[-1])
    row_count = rows.shape[0]
    device = sequence_group.device
    if current_step == 0:
        return torch.full(
            (row_count,),
            first_token_ids.numel(),
            dtype=torch.int64,
            device=device,
        )

    first = rows[:, 0].to(torch.int64)
    first_valid = (first >= 0) & (first < vocab_size)
    safe_first = first.clamp(0, max(vocab_size - 1, 0))
    if current_step == 1:
        counts = prefix1_offsets[safe_first + 1] - prefix1_offsets[safe_first]
        return torch.where(first_valid, counts, torch.zeros_like(counts)).to(torch.int64)

    second = rows[:, 1].to(torch.int64)
    prefix_valid = first_valid & (second >= 0) & (second < vocab_size)
    if prefix1_pair_keys.numel() == 0:
        return torch.zeros(row_count, dtype=torch.int64, device=device)

    queries = first * vocab_size + second
    positions = torch.searchsorted(prefix1_pair_keys, queries, right=False)
    in_range = positions < prefix1_pair_keys.numel()
    safe_positions = positions.clamp(0, prefix1_pair_keys.numel() - 1)
    found = in_range & (prefix1_pair_keys[safe_positions] == queries)
    counts = prefix2_value_offsets[safe_positions + 1] - prefix2_value_offsets[safe_positions]
    return torch.where(prefix_valid & found, counts, torch.zeros_like(counts)).to(torch.int64)


def valid_mask_from_counts(counts: torch.Tensor, top_k: int) -> torch.Tensor:
    """Expand per-row counts into the fixed-width valid-prefix mask."""

    if counts.dim() != 1:
        raise ValueError(f"candidate counts must be 1-D, got shape {tuple(counts.shape)}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    positions = torch.arange(top_k, dtype=torch.int64, device=counts.device)
    return positions.unsqueeze(0).lt(counts.clamp(min=0, max=top_k).unsqueeze(1)).contiguous()


__all__ = ["constraint_candidate_counts", "valid_mask_from_counts"]
