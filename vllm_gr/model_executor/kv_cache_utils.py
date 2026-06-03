# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility helpers for reading and writing paged KV cache tensors."""

from __future__ import annotations

from typing import Any

import torch


def _resolve_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported torch dtype string: {dtype}")
        resolved = getattr(torch, dtype)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unsupported torch dtype string: {dtype}")
        return resolved
    return torch.bfloat16


def _complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    if lengths.numel() == 0:
        return torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    try:
        return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    except (RuntimeError, AttributeError):
        cumsum = torch.zeros(
            lengths.size(0) + 1,
            dtype=lengths.dtype,
            device=lengths.device,
        )
        cumsum[1:] = torch.cumsum(lengths, dim=0)
        return cumsum


def write_to_paged_cache(
    *,
    runtime_kv_cache: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
) -> None:
    if runtime_kv_cache is None or runtime_kv_cache.numel() == 0:
        return

    flat_slot_mapping = slot_mapping.reshape(-1).to(torch.long)
    valid_mask = flat_slot_mapping >= 0
    if not torch.any(valid_mask):
        return

    packed_kv = torch.cat([key, value], dim=-1)
    flat_cache = runtime_kv_cache.view(
        -1,
        num_kv_heads,
        head_size + head_size_v,
    )
    valid_token_indices = torch.nonzero(valid_mask, as_tuple=False).reshape(-1)
    slot_indices = flat_slot_mapping.index_select(0, valid_token_indices)
    packed_kv = packed_kv.index_select(0, valid_token_indices)
    flat_cache.index_copy_(0, slot_indices, packed_kv)


def read_from_paged_cache(
    *,
    runtime_kv_cache: torch.Tensor | None,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    kv_cache_torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    if runtime_kv_cache is None or runtime_kv_cache.numel() == 0:
        empty_k = torch.empty(
            0,
            num_kv_heads,
            head_size,
            dtype=kv_cache_torch_dtype,
            device=seq_lens.device,
        )
        empty_v = torch.empty(
            0,
            num_kv_heads,
            head_size_v,
            dtype=kv_cache_torch_dtype,
            device=seq_lens.device,
        )
        return empty_k, empty_v, 0, _complete_cumsum(seq_lens)

    block_size = int(runtime_kv_cache.shape[1])
    packed_per_req: list[torch.Tensor] = []
    for req_idx in range(seq_lens.shape[0]):
        seq_len = int(seq_lens[req_idx].item())
        if seq_len <= 0:
            continue
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_table_tensor[req_idx, :num_blocks].to(
            device=runtime_kv_cache.device,
            dtype=torch.long,
        )
        req_cache = runtime_kv_cache.index_select(0, block_ids).reshape(
            -1,
            num_kv_heads,
            head_size + head_size_v,
        )[:seq_len]
        packed_per_req.append(req_cache)

    if packed_per_req:
        packed = torch.cat(packed_per_req, dim=0)
    else:
        packed = runtime_kv_cache.new_empty((0, num_kv_heads, head_size + head_size_v))

    return (
        packed[..., :head_size],
        packed[..., head_size:],
        int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0,
        _complete_cumsum(seq_lens.to(device=packed.device, dtype=torch.int64)),
    )


__all__ = [
    "_complete_cumsum",
    "_resolve_torch_dtype",
    "read_from_paged_cache",
    "write_to_paged_cache",
]
