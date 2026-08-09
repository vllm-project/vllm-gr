# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def extract_suffix_kv_pytorch(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping_out: torch.Tensor,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch equivalent of the paged_kv_to_contig_suffix_kernel.
    Replaces indirect pointer math with vectorized advanced indexing.

    Args:
        k_cache: Tensor of shape [num_blocks, block_size, num_heads, head_dim]
        v_cache: Tensor of shape [num_blocks, block_size, num_heads, head_dim]
        slot_mapping_out: Indices indicating which slots to extract
        out_k: Optional pre-allocated output key tensor
        out_v: Optional pre-allocated output value tensor
    """

    # 1. Collapse paged blocks [num_blocks, block_size, heads, dim]
    # into a flat token space [total_physical_slots, heads, dim]
    num_heads, head_dim = k_cache.shape[2], k_cache.shape[3]
    k_flat = k_cache.view(-1, num_heads, head_dim)
    v_flat = v_cache.view(-1, num_heads, head_dim)

    # 2. Vectorized gather: Selects all matching token rows across all heads/dimensions
    if out_k is not None:
        out_k.copy_(k_flat[slot_mapping_out])
    else:
        out_k = k_flat[slot_mapping_out]

    if out_v is not None:
        out_v.copy_(v_flat[slot_mapping_out])
    else:
        out_v = v_flat[slot_mapping_out]

    return out_k, out_v


def merge_tnd_attention_outputs(
    out1: torch.Tensor, lse1: torch.Tensor, out2: torch.Tensor, lse2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes and merges two TND attention output pairs.

    Expected input dimensions:
        out1, out2: [T, N, D]  (Token, NumHead, HeadDim)
        lse1, lse2: [T, N, 1]  (output of _npu_normalize_lse)
    """
    # LSE is [T, N, 1], matching out1/out2 layout [T, N, D]

    # 1. Extract max values for numerical stability
    max_lse = torch.maximum(lse1, lse2)  # [T, N, 1]
    # Clamping to a finite negative floor
    max_lse = torch.clamp(max_lse, min=-10000.0)

    # 2. Compute exponential weights (relative to max for numerical stability)
    w1 = torch.exp(lse1 - max_lse)  # [T, N, 1]
    w2 = torch.exp(lse2 - max_lse)  # [T, N, 1]
    w_total = w1 + w2  # [T, N, 1]
    is_zero = w_total == 0.0
    w_total = torch.where(is_zero, 1.0, w_total)

    # 3. Compute weighted attention sum - shapes align: [T, N, D] * [T, N, 1] broadcasts
    normalized_out = ((w1 * out1 + w2 * out2) / w_total).to(dtype=out1.dtype)

    # 4. Compute merged LSE
    normalized_lse = max_lse + torch.log(w_total)  # [T, N, 1]

    return normalized_out, normalized_lse
