# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scoring for candidates selected by a Constraint Table.

Constraint Tables restrict which tokens may survive Top-K, but Beam Search
compares candidates from different parent rows.  Their local scores must
therefore use the same vocabulary domain.  Normalizing separately over each
prefix's legal set makes those scores incomparable (a row with one legal token
would always return logprob zero, regardless of model confidence).
"""

from __future__ import annotations

import torch

INVALID_LOGIT = -1.0e20


def score_selected_tokens(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    temperatures: torch.Tensor,
    *,
    vocab_offset: int = 0,
    invalid_logprob: float = INVALID_LOGIT,
) -> torch.Tensor:
    """Return selected-token logprobs over the complete logits window.

    ``token_ids`` remain global table IDs.  ``vocab_offset`` maps them into a
    sliced LM-head projection when one is active.  The available logits window
    is identical for every row, so the returned FP32 values are comparable
    across parent beams while the explicit structural mask remains authoritative.
    """

    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D, got shape {tuple(logits.shape)}")
    if token_ids.dim() != 2 or valid_mask.shape != token_ids.shape:
        raise ValueError("token_ids and valid_mask must be equal-shape 2-D tensors")
    if token_ids.shape[0] != logits.shape[0]:
        raise ValueError("token_ids row count must match logits")
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must use bool, got {valid_mask.dtype}")
    if temperatures.dim() > 1 or temperatures.numel() not in (0, 1, logits.shape[0]):
        raise ValueError("temperatures must be empty or contain one value or one value per row")
    if isinstance(vocab_offset, bool) or not isinstance(vocab_offset, int):
        raise TypeError("vocab_offset must be an int")
    if vocab_offset < 0:
        raise ValueError("vocab_offset must be non-negative")

    row_temperatures = temperatures.to(device=logits.device, dtype=torch.float32).reshape(-1)
    if row_temperatures.numel() == 0:
        row_temperatures = torch.ones(logits.shape[0], dtype=torch.float32, device=logits.device)
    elif row_temperatures.numel() == 1:
        row_temperatures = row_temperatures.expand(logits.shape[0])
    divisors = torch.where(
        row_temperatures == 0,
        torch.ones_like(row_temperatures),
        row_temperatures,
    )

    # Keep the original raw logits untouched even when they are already FP32.
    scaled_logits = logits.to(dtype=torch.float32, copy=True)
    scaled_logits.div_(divisors[:, None])

    local_ids = token_ids.to(torch.long) - vocab_offset
    safe_local_ids = local_ids.clamp(0, logits.shape[1] - 1)
    selected_scores = scaled_logits.gather(1, safe_local_ids)
    log_denominators = torch.logsumexp(scaled_logits, dim=1, keepdim=True)
    selected_logprobs = selected_scores - log_denominators

    in_window = local_ids.ge(0) & local_ids.lt(logits.shape[1])
    selected_logprobs.masked_fill_(~(valid_mask & in_window), invalid_logprob)
    return selected_logprobs


__all__ = ["INVALID_LOGIT", "score_selected_tokens"]
