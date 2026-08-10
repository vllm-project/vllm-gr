# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementations of incremental BeamSearch KV operators."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from vllm_gr.v1.attention.backends.beam_kernels import (
    _validate_cache_inputs,
    _validate_select_inputs,
)


@triton.jit
def _cache_unshared_kv_kernel(
    unshared_key,
    unshared_value,
    curr_key,
    curr_value,
    pool_key_stride_beam,
    pool_key_stride_head,
    pool_key_stride_step,
    pool_key_stride_dim,
    pool_value_stride_beam,
    pool_value_stride_head,
    pool_value_stride_step,
    pool_value_stride_dim,
    curr_key_stride_beam,
    curr_key_stride_head,
    curr_key_stride_dim,
    curr_value_stride_beam,
    curr_value_stride_head,
    curr_value_stride_dim,
    head_dim,
    step,
    BLOCK_SIZE_D: tl.constexpr,
):
    beam = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, BLOCK_SIZE_D)
    mask = dim < head_dim

    curr_key_ptrs = (
        curr_key
        + beam * curr_key_stride_beam
        + head * curr_key_stride_head
        + dim * curr_key_stride_dim
    )
    curr_value_ptrs = (
        curr_value
        + beam * curr_value_stride_beam
        + head * curr_value_stride_head
        + dim * curr_value_stride_dim
    )
    pool_key_ptrs = (
        unshared_key
        + beam * pool_key_stride_beam
        + head * pool_key_stride_head
        + step * pool_key_stride_step
        + dim * pool_key_stride_dim
    )
    pool_value_ptrs = (
        unshared_value
        + beam * pool_value_stride_beam
        + head * pool_value_stride_head
        + step * pool_value_stride_step
        + dim * pool_value_stride_dim
    )

    key = tl.load(curr_key_ptrs, mask=mask)
    value = tl.load(curr_value_ptrs, mask=mask)
    tl.store(pool_key_ptrs, key, mask=mask)
    tl.store(pool_value_ptrs, value, mask=mask)


def cache_unshared_kv_triton(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    step: int,
) -> None:
    """Write one decode step into fixed-address CUDA KV pools using Triton."""

    beam_width = _validate_cache_inputs(unshared_key, unshared_value, curr_key, curr_value, step)
    if unshared_key.device.type != "cuda":
        raise ValueError("Triton operators require CUDA tensors")

    num_kv_heads = unshared_key.shape[2]
    head_dim = unshared_key.shape[4]
    block_size_d = triton.next_power_of_2(head_dim)
    grid = (beam_width, num_kv_heads)
    _cache_unshared_kv_kernel[grid](
        unshared_key,
        unshared_value,
        curr_key,
        curr_value,
        unshared_key.stride(1),
        unshared_key.stride(2),
        unshared_key.stride(3),
        unshared_key.stride(4),
        unshared_value.stride(1),
        unshared_value.stride(2),
        unshared_value.stride(3),
        unshared_value.stride(4),
        curr_key.stride(0),
        curr_key.stride(1),
        curr_key.stride(2),
        curr_value.stride(0),
        curr_value.stride(1),
        curr_value.stride(2),
        head_dim,
        step,
        BLOCK_SIZE_D=block_size_d,
    )


@triton.jit
def _select_unshared_kv_kernel(
    parent,
    source_key,
    source_value,
    destination_key,
    destination_value,
    parent_stride,
    source_key_stride_beam,
    source_key_stride_head,
    source_key_stride_step,
    source_key_stride_dim,
    source_value_stride_beam,
    source_value_stride_head,
    source_value_stride_step,
    source_value_stride_dim,
    destination_key_stride_beam,
    destination_key_stride_head,
    destination_key_stride_step,
    destination_key_stride_dim,
    destination_value_stride_beam,
    destination_value_stride_head,
    destination_value_stride_step,
    destination_value_stride_dim,
    head_dim,
    BLOCK_SIZE_D: tl.constexpr,
):
    new_beam = tl.program_id(0)
    head = tl.program_id(1)
    history_step = tl.program_id(2)
    dim = tl.arange(0, BLOCK_SIZE_D)
    mask = dim < head_dim
    old_beam = tl.load(parent + new_beam * parent_stride)

    source_key_ptrs = (
        source_key
        + old_beam * source_key_stride_beam
        + head * source_key_stride_head
        + history_step * source_key_stride_step
        + dim * source_key_stride_dim
    )
    source_value_ptrs = (
        source_value
        + old_beam * source_value_stride_beam
        + head * source_value_stride_head
        + history_step * source_value_stride_step
        + dim * source_value_stride_dim
    )
    destination_key_ptrs = (
        destination_key
        + new_beam * destination_key_stride_beam
        + head * destination_key_stride_head
        + history_step * destination_key_stride_step
        + dim * destination_key_stride_dim
    )
    destination_value_ptrs = (
        destination_value
        + new_beam * destination_value_stride_beam
        + head * destination_value_stride_head
        + history_step * destination_value_stride_step
        + dim * destination_value_stride_dim
    )

    key = tl.load(source_key_ptrs, mask=mask)
    value = tl.load(source_value_ptrs, mask=mask)
    tl.store(destination_key_ptrs, key, mask=mask)
    tl.store(destination_value_ptrs, value, mask=mask)


def select_unshared_kv_triton(
    parent: torch.Tensor,
    unshared_key_caches: Sequence[torch.Tensor],
    unshared_value_caches: Sequence[torch.Tensor],
    step: int,
) -> None:
    """Select executed KV histories in place using snapshot-backed Triton gathers."""

    beam_width = _validate_select_inputs(parent, unshared_key_caches, unshared_value_caches, step)
    if parent.device.type != "cuda":
        raise ValueError("Triton operators require CUDA tensors")

    num_kv_heads = unshared_key_caches[0].shape[2]
    history_length = step + 1
    head_dim = unshared_key_caches[0].shape[4]
    block_size_d = triton.next_power_of_2(head_dim)
    grid = (beam_width, num_kv_heads, history_length)

    for key, value in zip(unshared_key_caches, unshared_value_caches):
        source_key = key[0, :beam_width, :, :history_length, :].clone()
        source_value = value[0, :beam_width, :, :history_length, :].clone()
        _select_unshared_kv_kernel[grid](
            parent,
            source_key,
            source_value,
            key,
            value,
            parent.stride(0),
            source_key.stride(0),
            source_key.stride(1),
            source_key.stride(2),
            source_key.stride(3),
            source_value.stride(0),
            source_value.stride(1),
            source_value.stride(2),
            source_value.stride(3),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            key.stride(4),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            value.stride(4),
            head_dim,
            BLOCK_SIZE_D=block_size_d,
        )


__all__ = ["cache_unshared_kv_triton", "select_unshared_kv_triton"]
