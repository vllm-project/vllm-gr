# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Unconstrained candidate selection for NPU Beam V1.

Selection runs outside the model graph. The full-vocabulary selector preserves
the complete finite vocabulary denominator and breaks score ties by token ID.
The native TopK selector normalizes only its selected scores.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_gr.constraints.backend import ConstraintOutput

# An early finite mask needs one additional logits-sized buffer. Limit the
# device-only path to the smaller tensors where it wins on NPU.
_EARLY_FINITE_MASK_MAX_BYTES = 192 * 1024**2


def _select_and_normalize_topk(
    logits: torch.Tensor,
    divisors: torch.Tensor,
    top_k: int,
    *,
    mask_nonfinite_before_topk: bool,
) -> ConstraintOutput:
    count = min(top_k, logits.shape[-1])
    check_selected_values = not mask_nonfinite_before_topk
    if mask_nonfinite_before_topk:
        logits = logits.masked_fill(~torch.isfinite(logits), -float("inf"))
    elif (
        logits.device.type == "npu"
        and logits.is_contiguous()
        and logits.numel() * logits.element_size() <= _EARLY_FINITE_MASK_MAX_BYTES
    ):
        logits = torch.nan_to_num(
            logits, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf")
        )
        check_selected_values = False
    values, ids = torch.topk(logits, count, dim=-1, sorted=True)
    # On larger tensors, allocate a vocabulary-sized mask only when an
    # invalid value displaced a finite candidate.
    if check_selected_values and (~torch.isfinite(values) & (values != -float("inf"))).any():
        finite_logits = logits.masked_fill(~torch.isfinite(logits), -float("inf"))
        values, ids = torch.topk(finite_logits, count, dim=-1, sorted=True)

    scores = values.float()
    maximum = scores[:, :1]
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    large = divisors.clamp_min(1.0)
    small = divisors.clamp_max(1.0)
    centered = scores / large - maximum / large
    scaled = torch.where(centered == 0, 0.0, centered / small)
    logprobs = torch.log_softmax(scaled, dim=-1)
    valid = torch.isfinite(logprobs)
    token_ids = torch.where(valid, ids.to(torch.int32), 0)
    logprobs = logprobs.masked_fill(~valid, -float("inf"))
    if count < top_k:
        padding = (0, top_k - count)
        token_ids = F.pad(token_ids, padding, value=0)
        logprobs = F.pad(logprobs, padding, value=-float("inf"))
        valid = F.pad(valid, padding, value=False)
    return ConstraintOutput(token_ids, logprobs, valid)


def select_unconstrained_candidates(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    *,
    normalize_selected: bool = False,
    mask_nonfinite_before_topk: bool = False,
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
    With normalize_selected, native TopK determines the tie order and FP32
    normalization uses the selected logits. Prefill and small contiguous
    NPU Decode tensors apply the finite mask before native TopK to avoid a
    device-to-host scalar check.
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
    if normalize_selected:
        return _select_and_normalize_topk(
            logits, divisors, top_k, mask_nonfinite_before_topk=mask_nonfinite_before_topk
        )
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
