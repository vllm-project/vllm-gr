# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Adapt vLLM's FlashAttention varlen call contract to ROCm FlashAttention.

On ROCm, ``vllm.v1.attention.backends.fa_utils`` exports the upstream
``flash_attn`` (Composable Kernel) ``flash_attn_varlen_func``, whose signature
differs from ``vllm.vllm_flash_attn``. Beam attention is written against the
vLLM contract, so this adapter maps it onto the upstream FA2 API:

* ``seqused_k`` becomes ``cu_seqlens_k``; with ``block_table`` the CK kernel
  reads per-sequence key lengths from the ``cu_seqlens_k`` differences.
* ``return_softmax_lse`` returns ``(out, lse)`` with ``lse`` shaped
  ``[num_heads, total_q]``, matching the vLLM FA2 varlen layout.
* ``fa_version``, ``num_splits`` and ``scheduler_metadata`` tune CUDA
  kernels only and are ignored. Sinks (``s_aux``) are rejected.

The CK paged-KV kernel requires a KV block size divisible by
``ROCM_PAGED_KV_BLOCK_SIZE``; ``BeamAttentionBackend`` selects it on ROCm.
The Triton AMD backend (``FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE``, used where
CK is unavailable, e.g. RDNA) has no paged varlen path: paged KV is gathered
into a contiguous buffer first, with static shapes so graph capture stays valid.
"""

from __future__ import annotations

from typing import Any

import torch

#: Paged-KV block granularity required by the ROCm CK FlashAttention kernel.
ROCM_PAGED_KV_BLOCK_SIZE = 128


def _cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    cu = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths.to(torch.int32), dim=0, out=cu[1:])
    return cu


def _uses_triton_backend() -> bool:
    """Whether upstream ``flash_attn`` dispatches to its Triton AMD backend."""
    from flash_attn import flash_attn_interface

    return bool(getattr(flash_attn_interface, "USE_TRITON_ROCM", False))


def gather_paged_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack paged ``[blocks, block_size, heads, dim]`` KV into varlen rows.

    Returns ``(key, value, cu_seqlens_k)``. Shapes depend only on the batch
    size and ``max_seqlen_k``; lengths are read on the device, so the copy is
    valid inside a captured graph. Rows past each length go to a scratch row.
    """
    batch = seqused_k.numel()
    block_size = key_cache.shape[1]
    device = key_cache.device
    lengths = seqused_k.to(torch.int64)
    cu_seqlens_k = _cu_seqlens(seqused_k)
    token = torch.arange(max_seqlen_k, device=device)
    column = (token // block_size).clamp_(max=block_table.shape[1] - 1)
    blocks = block_table.to(torch.int64)[:, column]
    num_slots = key_cache.shape[0] * block_size
    source = (blocks * block_size + token % block_size).clamp_(0, num_slots - 1)
    valid = token.unsqueeze(0) < lengths.unsqueeze(1)
    scratch = batch * max_seqlen_k
    target = torch.where(
        valid, cu_seqlens_k[:-1].to(torch.int64).unsqueeze(1) + token, scratch
    ).flatten()
    source = source.flatten()
    packed = []
    for cache in (key_cache, value_cache):
        flat = cache.reshape(num_slots, *cache.shape[2:])
        out = cache.new_empty(scratch + 1, *cache.shape[2:])
        out.index_copy_(0, target, flat.index_select(0, source))
        packed.append(out[:scratch])
    return packed[0], packed[1], cu_seqlens_k


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: Any = None,
    softcap: float | None = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    out: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
    fa_version: int | None = None,
    num_splits: int = 0,
    scheduler_metadata: Any = None,
    **unsupported: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run ROCm FlashAttention with the vLLM ``flash_attn_varlen_func`` contract."""
    del fa_version, num_splits, scheduler_metadata
    if unsupported:
        raise TypeError(f"ROCm FlashAttention does not support {sorted(unsupported)}")
    if s_aux is not None:
        raise NotImplementedError("ROCm FlashAttention does not support attention sinks")
    from flash_attn import flash_attn_varlen_func as _rocm_flash_attn_varlen_func

    if block_table is not None and _uses_triton_backend():
        if seqused_k is None:
            raise ValueError("Paged ROCm Triton FlashAttention requires seqused_k")
        k, v, cu_seqlens_k = gather_paged_kv(k, v, block_table, seqused_k, max_seqlen_k)
        block_table = seqused_k = None
    if cu_seqlens_k is None:
        if seqused_k is None:
            raise ValueError("cu_seqlens_k or seqused_k is required")
        cu_seqlens_k = _cu_seqlens(seqused_k)
    elif seqused_k is not None and block_table is None:
        raise ValueError("ROCm FlashAttention cannot combine cu_seqlens_k with seqused_k")
    if block_table is not None and k.shape[1] % ROCM_PAGED_KV_BLOCK_SIZE:
        raise ValueError(
            f"ROCm paged FlashAttention requires a KV block size divisible by "
            f"{ROCM_PAGED_KV_BLOCK_SIZE}, got {k.shape[1]}"
        )
    window = (-1, -1) if window_size is None else tuple(int(x) for x in window_size)
    result = _rocm_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window,
        softcap=float(softcap or 0.0),
        alibi_slopes=alibi_slopes,
        return_attn_probs=return_softmax_lse,
        block_table=block_table,
    )
    if return_softmax_lse:
        attn_out, lse = result[0], result[1]
    else:
        attn_out, lse = result, None
    if out is not None:
        out.copy_(attn_out)
        attn_out = out
    if return_softmax_lse:
        return attn_out, lse
    return attn_out
