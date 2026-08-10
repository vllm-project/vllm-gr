# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU implementation of the ``cache_unshared_kv`` operator.

The cache layout follows xllm-ops:

* key/value cache: ``[num_blocks, beam, num_heads, max_decode_steps, head_dim]``
* current key/value: ``[batch * beam, num_heads, head_dim]``
* block table: ``[batch]``
* decode step: one-element, 1-based integer tensor

The implementation updates the two cache tensors in place and returns them,
matching the NPU operator contract.
"""

from __future__ import annotations

import torch

_SUPPORTED_CACHE_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def _validate_inputs(
    x_key_block: torch.Tensor,
    x_value_block: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    block_table: torch.Tensor,
    decode_step: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if x_key_block.ndim != 5:
        raise ValueError(
            "x_key_block must have shape [num_blocks, beam, num_heads, max_decode_steps, head_dim]"
        )
    if x_value_block.shape != x_key_block.shape:
        raise ValueError("x_value_block must have the same shape as x_key_block")
    if curr_key.ndim != 3:
        raise ValueError("curr_key must have shape [batch * beam, num_heads, head_dim]")
    if curr_value.shape != curr_key.shape:
        raise ValueError("curr_value must have the same shape as curr_key")
    if block_table.ndim != 1:
        raise ValueError("block_table must be one-dimensional")
    if decode_step.numel() != 1:
        raise ValueError("decode_step must contain exactly one element")

    if x_key_block.dtype not in _SUPPORTED_CACHE_DTYPES:
        raise TypeError("key/value tensors must use torch.float16 or torch.bfloat16")
    for name, tensor in (
        ("x_value_block", x_value_block),
        ("curr_key", curr_key),
        ("curr_value", curr_value),
    ):
        if tensor.dtype != x_key_block.dtype:
            raise TypeError(f"{name} must have the same dtype as x_key_block")
    if block_table.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError("block_table must use torch.int32 or torch.int64")
    if decode_step.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError("decode_step must use torch.int32 or torch.int64")

    device = x_key_block.device
    for name, tensor in (
        ("x_value_block", x_value_block),
        ("curr_key", curr_key),
        ("curr_value", curr_value),
        ("block_table", block_table),
        ("decode_step", decode_step),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")

    num_blocks, beam_size, num_heads, max_decode_steps, head_dim = x_key_block.shape
    batch_size = block_table.numel()
    expected_curr_shape = (batch_size * beam_size, num_heads, head_dim)
    if curr_key.shape != expected_curr_shape:
        raise ValueError(
            f"curr_key must have shape {expected_curr_shape}, got {tuple(curr_key.shape)}"
        )
    if num_blocks == 0 or beam_size == 0 or num_heads == 0:
        raise ValueError("cache block, beam, and head dimensions must be non-zero")
    if max_decode_steps == 0 or head_dim == 0:
        raise ValueError("decode and head dimensions must be non-zero")

    return batch_size, beam_size, num_heads, max_decode_steps, head_dim


def cache_unshared_kv_torch(
    x_key_block: torch.Tensor,
    x_value_block: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    block_table: torch.Tensor,
    decode_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write the current KV tensors using ordinary PyTorch tensor operators.

    ``block_table`` entries must be unique and in range, and ``decode_step``
    must be in ``[1, max_decode_steps]``. These are device-side runtime values,
    so the launcher deliberately avoids reading them on the CPU.
    """

    batch_size, beam_size, num_heads, _, head_dim = _validate_inputs(
        x_key_block,
        x_value_block,
        curr_key,
        curr_value,
        block_table,
        decode_step,
    )
    if batch_size == 0:
        return x_key_block, x_value_block

    cache_blocks = block_table.to(torch.long)
    decode_index = (decode_step - 1).to(torch.long)
    current_shape = (batch_size, beam_size, num_heads, head_dim)

    x_key_block[cache_blocks, :, :, decode_index, :] = curr_key.reshape(current_shape)
    x_value_block[cache_blocks, :, :, decode_index, :] = curr_value.reshape(current_shape)
    return x_key_block, x_value_block


def cache_unshared_kv(
    x_key_block: torch.Tensor,
    x_value_block: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    block_table: torch.Tensor,
    decode_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache current K/V through the backend-neutral public entry point."""

    return cache_unshared_kv_torch(
        x_key_block,
        x_value_block,
        curr_key,
        curr_value,
        block_table,
        decode_step,
    )


__all__ = ["cache_unshared_kv", "cache_unshared_kv_torch"]
