# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

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
