# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary candidates for the native, zero-temperature Beam contract.

This operation runs after the model Forward, outside the Forward graph. The
candidate width is independent of the number of live Beams: native vLLM
requests twice as many candidates so EOS completions do not consume the live
Beam slots. Token IDs are global model IDs; no catalog or vocabulary window
participates in either selection or normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_gr.config.models import MAX_BEAM_WIDTH


@dataclass(frozen=True, slots=True)
class FullVocabCandidates:
    token_ids: torch.Tensor
    logprobs: torch.Tensor
    valid_mask: torch.Tensor
    valid_counts: torch.Tensor


def full_vocab_topk(logits: torch.Tensor, top_k: int) -> FullVocabCandidates:
    """Return fixed-width candidates scored over the complete effective vocab.

    Scores match native vLLM's raw FP32 log-softmax, before sampling transforms.
    The caller admits only the official zero-temperature Beam profile. As with
    native ``torch.topk``, the order of exactly tied tokens is unspecified.

    Rows with no finite probability have zero valid candidates. Padding and
    non-finite probabilities are marked invalid and carry ``-inf`` scores; they
    can never be promoted to live Beams. No device-to-host synchronization is
    needed to compute validity.
    """
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError("full-vocabulary logits must have shape [rows, vocab] with nonzero axes")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("full-vocabulary logits must be FP16, BF16 or FP32")
    if type(top_k) is not int or not 1 <= top_k <= 2 * MAX_BEAM_WIDTH:
        raise ValueError(f"full-vocabulary candidate width must be in [1, {2 * MAX_BEAM_WIDTH}]")

    rows, vocab_size = logits.shape
    width = min(top_k, vocab_size)
    scores = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    values, indices = torch.topk(scores, width, dim=-1, sorted=True)
    valid = torch.isfinite(values)
    values = values.masked_fill(~valid, float("-inf"))
    token_ids = indices.to(torch.int32)
    counts = valid.sum(dim=-1, dtype=torch.int32)
    if width < top_k:
        padding = (rows, top_k - width)
        token_ids = torch.cat(
            (token_ids, torch.zeros(padding, dtype=torch.int32, device=logits.device)), dim=-1
        )
        values = torch.cat(
            (values, torch.full(padding, float("-inf"), dtype=torch.float32, device=logits.device)),
            dim=-1,
        )
        valid = torch.cat(
            (valid, torch.zeros(padding, dtype=torch.bool, device=logits.device)), dim=-1
        )
    return FullVocabCandidates(
        token_ids.contiguous(), values.contiguous(), valid.contiguous(), counts.contiguous()
    )
