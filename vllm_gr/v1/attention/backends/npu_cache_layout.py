# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NPU Beam attention cache layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def restore_npu_cache_pool_layout(tensor: Any, request_width: int) -> Any:
    """Restore ``[B, W, ...]`` before the xllm cache-write operator."""

    if request_width < 1:
        raise ValueError("request_width must be positive")
    if tensor.shape[0] != 1 or tensor.shape[1] == request_width:
        return tensor
    if tensor.shape[1] % request_width:
        raise RuntimeError("flattened NPU pool width is not divisible by Beam width")
    batch = tensor.shape[1] // request_width
    return tensor.view(batch, request_width, *tensor.shape[2:])


def interleaved_pool_base(k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor | None:
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
