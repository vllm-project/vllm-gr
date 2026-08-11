# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementation of the xllm-compatible KV cache write operator."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm_gr.ops.cache_unshared_kv import _validate_inputs


@triton.jit
def _cache_unshared_kv_kernel(
    key_cache,
    value_cache,
    curr_key,
    curr_value,
    block_table,
    decode_step,
    key_stride_block,
    key_stride_beam,
    key_stride_head,
    key_stride_step,
    key_stride_dim,
    value_stride_block,
    value_stride_beam,
    value_stride_head,
    value_stride_step,
    value_stride_dim,
    curr_key_stride_token,
    curr_key_stride_head,
    curr_key_stride_dim,
    curr_value_stride_token,
    curr_value_stride_head,
    curr_value_stride_dim,
    block_table_stride,
    beam_size: tl.constexpr,
    head_dim,
    BLOCK_SIZE_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    request = token // beam_size
    beam = token % beam_size
    block = tl.load(block_table + request * block_table_stride)
    step = tl.load(decode_step) - 1
    dim = tl.arange(0, BLOCK_SIZE_D)
    mask = dim < head_dim

    curr_key_ptrs = (
        curr_key
        + token * curr_key_stride_token
        + head * curr_key_stride_head
        + dim * curr_key_stride_dim
    )
    curr_value_ptrs = (
        curr_value
        + token * curr_value_stride_token
        + head * curr_value_stride_head
        + dim * curr_value_stride_dim
    )
    key_cache_ptrs = (
        key_cache
        + block * key_stride_block
        + beam * key_stride_beam
        + head * key_stride_head
        + step * key_stride_step
        + dim * key_stride_dim
    )
    value_cache_ptrs = (
        value_cache
        + block * value_stride_block
        + beam * value_stride_beam
        + head * value_stride_head
        + step * value_stride_step
        + dim * value_stride_dim
    )

    key = tl.load(curr_key_ptrs, mask=mask)
    value = tl.load(curr_value_ptrs, mask=mask)
    tl.store(key_cache_ptrs, key, mask=mask)
    tl.store(value_cache_ptrs, value, mask=mask)


def cache_unshared_kv_triton(
    x_key_block: torch.Tensor,
    x_value_block: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    block_table: torch.Tensor,
    decode_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write current K/V with the same ABI as ``cache_unshared_kv_torch``."""

    batch_size, beam_size, num_heads, _, head_dim = _validate_inputs(
        x_key_block,
        x_value_block,
        curr_key,
        curr_value,
        block_table,
        decode_step,
    )
    if x_key_block.device.type != "cuda":
        raise ValueError("Triton operators require CUDA tensors")
    if batch_size == 0:
        return x_key_block, x_value_block

    block_size_d = triton.next_power_of_2(head_dim)
    grid = (batch_size * beam_size, num_heads)
    _cache_unshared_kv_kernel[grid](
        x_key_block,
        x_value_block,
        curr_key,
        curr_value,
        block_table,
        decode_step,
        x_key_block.stride(0),
        x_key_block.stride(1),
        x_key_block.stride(2),
        x_key_block.stride(3),
        x_key_block.stride(4),
        x_value_block.stride(0),
        x_value_block.stride(1),
        x_value_block.stride(2),
        x_value_block.stride(3),
        x_value_block.stride(4),
        curr_key.stride(0),
        curr_key.stride(1),
        curr_key.stride(2),
        curr_value.stride(0),
        curr_value.stride(1),
        curr_value.stride(2),
        block_table.stride(0),
        beam_size=beam_size,
        head_dim=head_dim,
        BLOCK_SIZE_D=block_size_d,
    )
    return x_key_block, x_value_block


__all__ = ["cache_unshared_kv_triton"]
