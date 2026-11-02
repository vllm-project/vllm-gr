# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def interleaved_pool_base(k_cache, v_cache):
    """The contiguous ``[num_blocks, 2, block, H, D]`` pool behind K/V views.

    vllm-ascend 0.30 allocates each layer's KV pool as one tensor with K and V
    interleaved per block; the per-layer views handed to attention are the
    non-contiguous halves ``base[:, 0]`` / ``base[:, 1]``. Both
    ``index_select`` and advanced indexing on those views materialize the
    entire pool before selecting, and flattening them with ``.view`` raises —
    so gathers must run against this reconstructed contiguous base instead.

    Returns None for any other layout (contiguous caches, mismatched shapes):
    callers fall back to their original per-tensor path unchanged.
    """
    if k_cache.dim() != 4 or k_cache.shape != v_cache.shape:
        return None
    num_blocks, block, num_heads, head_dim = k_cache.shape
    block_elems = block * num_heads * head_dim
    expected_stride = (2 * block_elems, num_heads * head_dim, head_dim, 1)
    if k_cache.stride() != expected_stride or v_cache.stride() != expected_stride:
        return None
    if v_cache.data_ptr() - k_cache.data_ptr() != block_elems * k_cache.element_size():
        return None
    try:
        return k_cache.as_strided(
            (num_blocks, 2, block, num_heads, head_dim),
            (2 * block_elems, block_elems, num_heads * head_dim, head_dim, 1),
        )
    except RuntimeError:
        # View escaping the underlying storage: not an interleaved pool.
        return None


def extract_suffix_kv_pytorch(k_cache, v_cache, slot_mapping_out):
    """
    Pure PyTorch equivalent of the paged_kv_to_contig_suffix_kernel.
    Replaces indirect pointer math with vectorized advanced indexing.

    Args:
        k_cache: Tensor of shape [num_blocks, block_size, num_heads, head_dim]
        v_cache: Tensor of shape [num_blocks, block_size, num_heads, head_dim]
        slot_mapping_out: Indices indicating which slots to extract
    """
    num_heads, head_dim = k_cache.shape[2], k_cache.shape[3]

    base = interleaved_pool_base(k_cache, v_cache)
    if base is not None:
        # K/V-interleaved pool (vllm-ascend 0.30): the flat ``.view`` below
        # is not expressible on the strided per-layer views. Gather by
        # (block, half, offset) directly from the contiguous base instead.
        block = k_cache.shape[1]
        flat = slot_mapping_out.reshape(-1)
        block_idx = torch.div(flat, block, rounding_mode="floor")
        token_idx = flat - block_idx * block
        return base[block_idx, 0, token_idx], base[block_idx, 1, token_idx]

    # 1. Collapse paged blocks [num_blocks, block_size, heads, dim]
    # into a flat token space [total_physical_slots, heads, dim]
    k_flat = k_cache.view(-1, num_heads, head_dim)
    v_flat = v_cache.view(-1, num_heads, head_dim)

    # 2. Vectorized gather: Selects all matching token rows across all heads/dimensions
    out_k = k_flat[slot_mapping_out]
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
