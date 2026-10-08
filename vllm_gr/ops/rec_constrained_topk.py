# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PyTorch implementation of the composite constrained Top-K operator.

The operator consumes raw logits and the compact three-level constraint-table
representation used by OneRec.  It returns the top-k valid token ids and their
log probabilities for every input row.  This implementation intentionally uses
ordinary PyTorch operators so it can serve as a portable fallback when the NPU
ACLNN fused operator is unavailable.
"""

from __future__ import annotations

import torch

from vllm_gr.constraints.scoring import INVALID_LOGIT as _INVALID_LOGIT
from vllm_gr.constraints.scoring import score_selected_tokens


def _range_candidates(
    values: torch.Tensor,
    begins: torch.Tensor,
    degrees: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(width, device=begins.device, dtype=torch.long)
    positions = begins.to(torch.long).unsqueeze(1) + offsets.unsqueeze(0)
    valid = offsets.unsqueeze(0).lt(degrees.to(torch.long).unsqueeze(1))
    max_position = max(values.numel() - 1, 0)
    safe_positions = torch.where(valid, positions, torch.zeros_like(positions)).clamp(
        0, max_position
    )
    candidates = values.index_select(0, safe_positions.reshape(-1)).view_as(safe_positions)
    return candidates, valid


def _finish_topk(
    logits: torch.Tensor,
    candidate_tokens: torch.Tensor,
    valid_candidates: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_logits = logits.gather(1, candidate_tokens.to(torch.long)).float()
    candidate_logits = candidate_logits.masked_fill(~valid_candidates, float("-inf"))
    # Stable descending order freezes the lower constraint-table offset as
    # the exact-tie winner and serves as the deterministic CUDA oracle.
    # NaN candidates must sort last (mirroring the fused CUDA kernel), not
    # first, so finite candidates win selection on poisoned logits.
    safe_logits = torch.nan_to_num(candidate_logits, nan=float("-inf"))
    top_indices = torch.argsort(safe_logits, dim=1, descending=True, stable=True)[:, :top_k]
    top_valid = valid_candidates.gather(1, top_indices)
    top_tokens = candidate_tokens.gather(1, top_indices).to(torch.int64)
    # Every device scores the selection on the shared full-window contract so
    # the reference stays interchangeable with the CPU backend, the CUDA
    # fused kernel, and the NPU wrapper (which re-scores with the same
    # helper) alike.
    top_logprobs = score_selected_tokens(
        logits,
        top_tokens,
        top_valid,
        temperatures,
        invalid_logprob=_INVALID_LOGIT,
    )
    return top_tokens, top_logprobs


def _bucketed_topk(
    logits: torch.Tensor,
    values: torch.Tensor,
    begins: torch.Tensor,
    degrees: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    candidate_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process wide degree ranges in geometric buckets."""
    rows = logits.size(0)
    tokens = torch.empty((rows, top_k), dtype=torch.int64, device=logits.device)
    probs = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
    remaining = torch.ones(rows, dtype=torch.bool, device=logits.device)
    widths: list[int] = []
    width = top_k
    while width < candidate_width:
        widths.append(width)
        if width > candidate_width // 2:
            break
        width *= 2

    for width in (*widths, candidate_width):
        mask = remaining & degrees.le(width)
        row_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if row_indices.numel() > 0:
            row_candidates, row_valid = _range_candidates(
                values,
                begins.index_select(0, row_indices),
                degrees.index_select(0, row_indices),
                width,
            )
            row_temps = (
                temperatures.index_select(0, row_indices)
                if temperatures.dim() > 0 and temperatures.numel() == rows
                else temperatures
            )
            row_tokens, row_probs = _finish_topk(
                logits.index_select(0, row_indices),
                row_candidates,
                row_valid,
                row_temps,
                top_k,
            )
            tokens.index_copy_(0, row_indices, row_tokens)
            probs.index_copy_(0, row_indices, row_probs)
        remaining = remaining & ~mask
    return tokens, probs


def _degree_width(degrees: torch.Tensor, top_k: int, configured: int) -> int:
    if configured > 0:
        return max(top_k, configured)
    if degrees.numel() == 0:
        return top_k
    return max(top_k, int(degrees.max().item()))


def rec_constrained_topk_torch(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int = 0,
    max_prefix2_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute constrained per-row top-k candidates using PyTorch operators."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D, got {tuple(logits.shape)}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows = logits.size(0)
    if current_step == 0:
        allowed = first_token_ids.reshape(-1)
        width = max(top_k, allowed.numel())
        begins = torch.zeros(rows, dtype=torch.long, device=allowed.device)
        degrees = torch.full_like(begins, allowed.numel())
        candidates, valid = _range_candidates(allowed, begins, degrees, width)
        return _finish_topk(logits, candidates, valid, temperatures, top_k)

    if sequence_group is None or sequence_group.size(-1) <= current_step:
        raise ValueError("sequence_group does not contain the requested prefix")

    flat = sequence_group.reshape(-1, sequence_group.size(-1))
    if flat.size(0) != rows:
        raise ValueError("sequence_group row count must match logits")

    if current_step == 1:
        t0 = flat[:, 0].to(torch.long)
        begins = prefix1_offsets.index_select(0, t0)
        ends = prefix1_offsets.index_select(0, t0 + 1)
        degrees = ends - begins
        width = _degree_width(degrees, top_k, max_prefix1_degree)
        if width <= 2 * top_k:
            candidates, valid = _range_candidates(prefix1_values, begins, degrees, width)
            return _finish_topk(logits, candidates, valid, temperatures, top_k)
        return _bucketed_topk(logits, prefix1_values, begins, degrees, temperatures, top_k, width)

    if current_step == 2:
        if prefix2_value_offsets.numel() < 2 or prefix2_values.numel() == 0:
            width = max(top_k, max_prefix2_degree)
            candidates = torch.zeros(
                (rows, width), dtype=prefix2_values.dtype, device=logits.device
            )
            valid = torch.zeros((rows, width), dtype=torch.bool, device=logits.device)
            return _finish_topk(logits, candidates, valid, temperatures, top_k)
        # Pair keys are encoded as t0 * vocab_size + t1. The prefix1 table has
        # one offset per vocabulary token, so its size provides vocab_size.
        vocab_size = prefix1_offsets.numel() - 1
        query = flat[:, 0].to(torch.long) * vocab_size + flat[:, 1].to(torch.long)
        pair_indices = torch.searchsorted(prefix1_pair_keys, query, right=False)
        safe = pair_indices.clamp(0, max(prefix1_pair_keys.numel() - 1, 0))
        found = prefix1_pair_keys.index_select(0, safe)
        pair_valid = pair_indices.lt(prefix1_pair_keys.numel()) & found.eq(query)
        safe_rows = torch.where(pair_valid, pair_indices, torch.zeros_like(pair_indices)).clamp(
            0, max(prefix2_value_offsets.numel() - 2, 0)
        )
        begins = prefix2_value_offsets.index_select(0, safe_rows)
        ends = prefix2_value_offsets.index_select(0, safe_rows + 1)
        degrees = torch.where(pair_valid, ends - begins, torch.zeros_like(begins))
        width = _degree_width(degrees, top_k, max_prefix2_degree)
        if width <= 2 * top_k:
            candidates, valid = _range_candidates(prefix2_values, begins, degrees, width)
            return _finish_topk(logits, candidates, valid, temperatures, top_k)
        return _bucketed_topk(logits, prefix2_values, begins, degrees, temperatures, top_k, width)

    raise ValueError(f"unsupported current_step={current_step}; expected 0, 1, or 2")


def rec_constrained_topk(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int = 0,
    max_prefix2_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch constrained top-k to a device implementation or Torch fallback."""
    if logits.device.type == "cuda":
        from vllm_gr.ops.cuda.rec_constrained_topk import rec_constrained_topk_cuda

        tokens, logprobs, _, _ = rec_constrained_topk_cuda(
            logits,
            sequence_group,
            first_token_ids,
            prefix1_offsets,
            prefix1_values,
            prefix1_pair_keys,
            prefix2_value_offsets,
            prefix2_values,
            temperatures,
            current_step,
            top_k,
            max_prefix1_degree,
            max_prefix2_degree,
        )
        return tokens, logprobs
    if logits.device.type == "npu":
        from vllm_gr.ops.npu import (
            rec_constrained_topk_npu,
            rec_constrained_topk_npu_available,
        )

        if rec_constrained_topk_npu_available():
            return rec_constrained_topk_npu(
                logits,
                sequence_group,
                first_token_ids,
                prefix1_offsets,
                prefix1_values,
                prefix1_pair_keys,
                prefix2_value_offsets,
                prefix2_values,
                temperatures,
                current_step,
                top_k,
                max_prefix1_degree,
                max_prefix2_degree,
            )
    return rec_constrained_topk_torch(
        logits,
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        current_step,
        top_k,
        max_prefix1_degree,
        max_prefix2_degree,
    )


__all__ = ["rec_constrained_topk", "rec_constrained_topk_torch"]
