# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
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
    Temperature dtypes are float16, bfloat16, or float32, matching native
    NPU arithmetic; float64 is rejected rather than silently underflowing or
    overflowing during conversion to float32.
    NaN and infinite logits have no probability mass. Log probabilities that
    cannot be represented in float32 are also invalid. Missing candidates are
    explicit invalid slots, never sampled-token side-channel entries.
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
    if temperatures.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("temperatures must use float16, bfloat16, or float32")

    temperature = temperatures.float().reshape(-1, 1)
    divisors = torch.where(temperature == 0, torch.ones_like(temperature), temperature)
    scores = logits.float()
    scores = scores.masked_fill(~torch.isfinite(scores), -float("inf"))

    # Original column order is tokenizer order, so stable ties select the
    # lower global token ID even at the TopK boundary.
    selected_scores, selected_ids = torch.sort(scores, dim=-1, descending=True, stable=True)

    # Center before dividing by a small temperature: finite logits such as
    # [2, 1] must not both overflow to infinity at temperature 1e-40. For a
    # large temperature, divide first so mixed-sign logits do not overflow
    # during subtraction (e.g. [-3e38, 3e38] / 3e38). Sorting the original
    # logits also preserves their order when scaling rounds distinct values
    # to the same float32 score. All operations remain device-resident.
    maximum = selected_scores[:, :1]
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    large_divisors = divisors.clamp_min(1.0)
    small_divisors = divisors.clamp_max(1.0)
    centered = selected_scores / large_divisors - maximum / large_divisors
    # Preserve the maxima even on devices that flush a subnormal divisor to
    # zero; 0 / 0 must not invalidate the highest-probability candidates.
    scores = torch.where(centered == 0, torch.zeros_like(centered), centered / small_divisors)
    normalizer = torch.logsumexp(scores, dim=-1, keepdim=True)
    normalizer = torch.where(torch.isfinite(normalizer), normalizer, torch.zeros_like(normalizer))
    count = min(top_k, vocab_size)
    selected_scores = scores[:, :count]
    selected_ids = selected_ids[:, :count]
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
