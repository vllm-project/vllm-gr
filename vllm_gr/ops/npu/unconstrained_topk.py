# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native full-vocabulary candidate baseline for Ascend Beam V1.

Selection runs outside the model graph. Log probabilities retain the complete
finite vocabulary denominator, independent of the requested candidate count.
Stable sorting gives equal logits the same ascending-token-ID order as CUDA.
"""

from __future__ import annotations

import torch

from vllm_gr.constraints.backend import ConstraintOutput


def select_unconstrained_candidates(
    logits: torch.Tensor, temperatures: torch.Tensor, top_k: int
) -> ConstraintOutput:
    """Return device-resident strict TopK without querying a catalog.

    A zero temperature has the existing Beam meaning of unscaled logits.
    Request validation owns finite, non-negative temperature values; this
    function checks tensor geometry without synchronizing device values.
    NaN and infinite logits have no probability mass, matching the CUDA
    unconstrained candidate implementation. Missing candidates are explicit
    invalid slots, never sampled-token side-channel entries.
    """
    if logits.ndim != 2 or min(logits.shape) < 1:
        raise ValueError("logits must be a non-empty two-dimensional tensor")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("logits must use float16, bfloat16, or float32")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    rows, vocab_size = logits.shape
    if temperatures.device != logits.device:
        raise ValueError("temperatures must share logits.device")
    if temperatures.ndim != 1 or temperatures.numel() not in (1, rows):
        raise ValueError("temperatures must contain one value or one value per row")
    if not temperatures.is_floating_point():
        raise TypeError("temperatures must use a floating-point dtype")

    temperature = temperatures.float().reshape(-1, 1)
    divisors = torch.where(temperature == 0, torch.ones_like(temperature), temperature)
    scores = logits.float() / divisors
    scores.masked_fill_(~torch.isfinite(scores), -float("inf"))

    # Original column order is tokenizer order, so stable ties select the
    # lower global token ID even at the TopK boundary.
    selected_scores, selected_ids = torch.sort(scores, dim=-1, descending=True, stable=True)
    count = min(top_k, vocab_size)
    selected_scores = selected_scores[:, :count]
    selected_ids = selected_ids[:, :count]
    normalizer = torch.logsumexp(scores, dim=-1, keepdim=True)
    normalizer = torch.where(torch.isfinite(normalizer), normalizer, torch.zeros_like(normalizer))
    valid = torch.isfinite(selected_scores)
    logprobs = (selected_scores - normalizer).masked_fill(~valid, -float("inf"))

    token_ids = torch.zeros((rows, top_k), dtype=torch.int32, device=logits.device)
    output_scores = torch.full(
        (rows, top_k), -float("inf"), dtype=torch.float32, device=logits.device
    )
    valid_mask = torch.zeros((rows, top_k), dtype=torch.bool, device=logits.device)
    token_ids[:, :count].copy_(torch.where(valid, selected_ids, 0).to(torch.int32))
    output_scores[:, :count].copy_(logprobs)
    valid_mask[:, :count].copy_(valid)
    return ConstraintOutput(token_ids, output_scores, valid_mask)
