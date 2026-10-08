# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU dispatch wrapper for the fused constrained Top-K operator."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

from vllm_gr.constraints.scoring import (
    INVALID_LOGIT as _INVALID_LOGIT,
)
from vllm_gr.constraints.scoring import (
    score_selected_tokens,
)
from vllm_gr.ops.npu._xllm_loader import custom_ops as _custom_ops

MAX_TOP_K = 256
NPUConstraintOp = Callable[..., tuple[torch.Tensor, torch.Tensor]]

# Chunk widths for the tie-order offset scan; together they bound the
# transient comparison buffers at
# rows * _TIE_SCAN_CHUNK * _TIE_TOPK_CHUNK * 5 bytes (bool mask + int32 key).
_TIE_SCAN_CHUNK = 2048
_TIE_TOPK_CHUNK = 32


def is_available() -> bool:
    """Return whether the NPU extension exposes the fused operator."""
    custom_ops = _custom_ops()
    return custom_ops is not None and callable(getattr(custom_ops, "rec_constrained_topk", None))


def unavailable_reason() -> str:
    """Describe why the fused NPU operator cannot be called."""
    custom_ops = _custom_ops()
    if custom_ops is None:
        return "custom_ops_lib cannot be imported"
    if not callable(getattr(custom_ops, "rec_constrained_topk", None)):
        return "custom_ops_lib does not export rec_constrained_topk"
    return ""


def resolve_rec_constrained_topk_npu() -> NPUConstraintOp:
    """Resolve the real fused callable once, without a Torch fallback."""

    custom_ops = _custom_ops()
    if custom_ops is None:
        raise RuntimeError(
            "NPU rec_constrained_topk is unavailable: custom_ops_lib cannot be imported"
        )
    op = getattr(custom_ops, "rec_constrained_topk", None)
    if not callable(op):
        raise RuntimeError(
            "NPU rec_constrained_topk is unavailable: "
            "custom_ops_lib does not export rec_constrained_topk"
        )
    return cast(NPUConstraintOp, op)


def _row_candidate_slices(
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    current_step: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve each row's candidate slice as ``(values, begins, degrees)``.

    Mirrors the lookup semantics of ``rec_constrained_topk_torch`` so the
    offsets recomputed here match the reference's candidate enumeration order.
    """
    device = first_token_ids.device
    if current_step == 0:
        allowed = first_token_ids.reshape(-1)
        begins = torch.zeros(rows, dtype=torch.long, device=device)
        degrees = torch.full_like(begins, allowed.numel())
        return allowed, begins, degrees

    flat = sequence_group.reshape(-1, sequence_group.size(-1))
    if current_step == 1:
        t0 = flat[:, 0].to(torch.long)
        begins = prefix1_offsets.index_select(0, t0).to(torch.long)
        ends = prefix1_offsets.index_select(0, t0 + 1).to(torch.long)
        return prefix1_values, begins, ends - begins

    if prefix2_value_offsets.numel() < 2 or prefix2_values.numel() == 0:
        zeros = torch.zeros(rows, dtype=torch.long, device=device)
        return prefix2_values, zeros, zeros
    # Pair keys are encoded as t0 * vocab_size + t1 (same encoding as the
    # Torch reference, which derives vocab_size from the prefix1 table size).
    pair_vocab_size = prefix1_offsets.numel() - 1
    query = flat[:, 0].to(torch.long) * pair_vocab_size + flat[:, 1].to(torch.long)
    pair_indices = torch.searchsorted(prefix1_pair_keys, query, right=False)
    safe = pair_indices.clamp(0, max(prefix1_pair_keys.numel() - 1, 0))
    found = prefix1_pair_keys.index_select(0, safe)
    pair_valid = pair_indices.lt(prefix1_pair_keys.numel()) & found.eq(query)
    safe_rows = torch.where(pair_valid, pair_indices, torch.zeros_like(pair_indices)).clamp(
        0, max(prefix2_value_offsets.numel() - 2, 0)
    )
    begins = prefix2_value_offsets.index_select(0, safe_rows).to(torch.long)
    ends = prefix2_value_offsets.index_select(0, safe_rows + 1).to(torch.long)
    degrees = torch.where(pair_valid, ends - begins, torch.zeros_like(begins))
    return prefix2_values, begins, degrees


def _stable_oracle_tie_order(
    tokens: torch.Tensor,
    logprobs: torch.Tensor,
    values: torch.Tensor,
    begins: torch.Tensor,
    degrees: torch.Tensor,
    scan_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder wrapper columns into the deterministic oracle order.

    The CUDA oracle and the Torch reference freeze exact-tie order as "score
    descending, then lower constraint-table offset" (a stable descending
    argsort over each row's candidate slice).  The fused operator's heap makes
    no such promise for bit-equal tie scores, so this recomputes every
    returned token's first position inside its row's candidate slice and
    reorders with a double stable argsort on (score desc, offset asc).

    The scan is chunked over the slice and blocked over the top-k columns so
    the transient comparison buffers stay bounded regardless of table degree
    or top_k; no host synchronization is introduced (``scan_width`` is derived
    from host-side metadata only).
    """
    rows, top_k = tokens.shape
    device = tokens.device
    long_tokens = tokens.to(torch.long)
    not_found = scan_width
    # Offsets are bounded by scan_width < 2**31 (constraint tables are int32
    # indexed), so int32 carries them and halves the transient below.
    offsets = torch.full((rows, top_k), not_found, dtype=torch.int32, device=device)
    for start in range(0, scan_width, _TIE_SCAN_CHUNK):
        stop = min(start + _TIE_SCAN_CHUNK, scan_width)
        local = torch.arange(start, stop, device=device, dtype=torch.long)
        positions = begins.unsqueeze(1) + local.unsqueeze(0)
        valid = local.unsqueeze(0).lt(degrees.unsqueeze(1))
        max_position = max(values.numel() - 1, 0)
        safe_positions = torch.where(valid, positions, torch.zeros_like(positions)).clamp(
            0, max_position
        )
        slice_tokens = (
            values.to(torch.long).index_select(0, safe_positions.reshape(-1)).view_as(safe_positions)
        )
        local_i32 = local.to(torch.int32)
        for k_start in range(0, top_k, _TIE_TOPK_CHUNK):
            k_stop = min(k_start + _TIE_TOPK_CHUNK, top_k)
            # First occurrence wins: `local` ascends within the chunk, so amin
            # keeps the lowest table offset among duplicate-token writes.
            matched = (
                slice_tokens.unsqueeze(2).eq(long_tokens[:, k_start:k_stop].unsqueeze(1))
                & valid.unsqueeze(2)
            )
            chunk_offsets = torch.where(matched, local_i32.view(1, -1, 1), not_found).amin(dim=1)
            offsets[:, k_start:k_stop] = torch.minimum(
                offsets[:, k_start:k_stop], chunk_offsets
            )

    scores = logprobs.to(torch.float32)
    sort_scores = torch.where(
        torch.isfinite(scores), scores, torch.full_like(scores, _INVALID_LOGIT)
    )
    # Double stable argsort: sort by the secondary key (offset ascending)
    # first, then stably by the primary key (score descending) so exact ties
    # keep the offset-ascending order. Offsets are sorted as fp32 (exact up to
    # 2**24) because the NPU argsort kernel rejects integer inputs and would
    # fall back to the much slower AiCPU path.
    sort_offsets = offsets.to(torch.float32) if values.numel() <= 2**24 else offsets
    by_offset = torch.argsort(sort_offsets, dim=1, stable=True)
    staged = sort_scores.gather(1, by_offset)
    by_score = torch.argsort(-staged, dim=1, stable=True)
    order = by_offset.gather(1, by_score)
    return tokens.gather(1, order), logprobs.gather(1, order)


def _slice_scan_width(
    current_step: int,
    first_token_ids: torch.Tensor,
    max_prefix1_degree: int,
    max_prefix2_degree: int,
    vocab_size: int,
) -> int:
    """Host-side upper bound for any row's candidate degree (no D2H sync)."""
    if current_step == 0:
        return max(1, first_token_ids.numel())
    configured = max_prefix1_degree if current_step == 1 else max_prefix2_degree
    if configured > 0:
        return max(1, configured)
    return max(1, vocab_size)


def rec_constrained_topk_npu(
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
    """Call the fused NPU operator using the shared operator signature.

    bf16 logits are widened to fp32 before dispatch. The fused kernel's bf16
    load path reads stale unified-buffer data on ascend910b and produces
    garbage logprobs, while its fp32 path is correct; bf16 -> fp32 is a
    lossless bit extension, so results match the reference exactly.

    The fused kernel only supplies the top-k selection (and its internal
    candidate-set scores, which are discarded). The wrapper re-scores the
    selection over the shared full-window domain with ``score_selected_tokens``
    and restores the deterministic oracle tie order, so its output is
    byte-compatible with the Torch reference.
    """
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(getattr(custom_ops, "rec_constrained_topk", None)):
        raise RuntimeError(
            f"NPU rec_constrained_topk is unavailable ({unavailable_reason()}); "
            "use the Torch implementation instead"
        )
    if logits.dtype == torch.bfloat16:
        logits = logits.to(torch.float32)
    # The operator returns its internally normalized (candidate-set) logprobs
    # alongside the selection; they are discarded below — the wrapper re-scores
    # the selected tokens over the shared full-window domain.
    token_ids, logprobs = cast(
        tuple[torch.Tensor, torch.Tensor],
        custom_ops.rec_constrained_topk(
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
        ),
    )
    # The fused kernel selects each row's top-k under its internal
    # candidate-set scores, but the NPU scoring domain is the shared
    # full-window contract (#389): CPU and CUDA normalize over the complete
    # logits window so scores are comparable across parent beams, and the NPU
    # path must land on the same domain.  Re-score the operator's selection
    # with score_selected_tokens — the identical helper the CPU backend and
    # the Torch reference use — discarding the operator's candidate-set
    # logprobs.  The operator's valid-before-invalid packing supplies the
    # structural mask (the per-row legal degree, mirrored from
    # rec_constrained_topk_torch); the backend still derives the same mask
    # independently for output validation.
    values, begins, degrees = _row_candidate_slices(
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        current_step,
        rows=token_ids.shape[0],
    )
    valid_mask = (
        torch.arange(top_k, device=token_ids.device).view(1, -1).lt(degrees.view(-1, 1))
    )
    logprobs = score_selected_tokens(logits, token_ids, valid_mask, temperatures)
    scan_width = _slice_scan_width(
        current_step,
        first_token_ids,
        max_prefix1_degree,
        max_prefix2_degree,
        vocab_size=logits.shape[1],
    )
    # The fused kernel's heap makes no exact-tie promise; restore the
    # deterministic oracle order (score descending, then lower
    # constraint-table offset) over the re-scored window.  Every step above
    # is contract-level and there is deliberately no soft fallback anywhere:
    # a failure propagates exactly like the CUDA and CPU paths, the only
    # alternative being a silent return to the operator's candidate-set
    # scores on an output that claims the shared domain.
    token_ids, logprobs = _stable_oracle_tie_order(
        token_ids, logprobs, values, begins, degrees, scan_width
    )
    return token_ids, logprobs


__all__ = [
    "MAX_TOP_K",
    "is_available",
    "rec_constrained_topk_npu",
    "resolve_rec_constrained_topk_npu",
    "unavailable_reason",
]
