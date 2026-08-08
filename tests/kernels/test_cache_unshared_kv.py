# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests derived from xllm-ops' cache_unshared_kv NPU test."""

from __future__ import annotations

import pytest
import torch

from vllm_gr.ops.cache_unshared_kv import cache_unshared_kv_torch


def _reference_write(
    x_key_block: torch.Tensor,
    x_value_block: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    block_table: torch.Tensor,
    decode_step: int,
) -> None:
    beam_size = x_key_block.shape[1]
    for batch_idx, block_idx in enumerate(block_table.cpu().tolist()):
        token_start = batch_idx * beam_size
        token_end = token_start + beam_size
        x_key_block[block_idx, :, :, decode_step - 1, :] = curr_key[token_start:token_end]
        x_value_block[block_idx, :, :, decode_step - 1, :] = curr_value[token_start:token_end]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_cache_unshared_kv_torch_matches_xllm_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(1234)

    batch_size = 3
    max_decode_steps = 3
    beam_size = 5
    num_heads = 2
    head_dim = 128
    num_blocks = 7
    cache_shape = (
        num_blocks,
        beam_size,
        num_heads,
        max_decode_steps,
        head_dim,
    )

    key_cache = torch.zeros(cache_shape, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    expected_key_cache = torch.zeros_like(key_cache)
    expected_value_cache = torch.zeros_like(value_cache)
    block_table = torch.tensor([5, 1, 3], dtype=torch.int32)

    for decode_step in range(1, max_decode_steps + 1):
        current_shape = (batch_size * beam_size, num_heads, head_dim)
        curr_key = torch.rand(current_shape, dtype=dtype)
        curr_value = torch.rand(current_shape, dtype=dtype)
        step_tensor = torch.tensor([decode_step], dtype=torch.int32)

        _reference_write(
            expected_key_cache,
            expected_value_cache,
            curr_key,
            curr_value,
            block_table,
            decode_step,
        )
        returned_key, returned_value = cache_unshared_kv_torch(
            key_cache,
            value_cache,
            curr_key,
            curr_value,
            block_table,
            step_tensor,
        )

        assert returned_key is key_cache
        assert returned_value is value_cache
        torch.testing.assert_close(key_cache, expected_key_cache, rtol=0, atol=0)
        torch.testing.assert_close(value_cache, expected_value_cache, rtol=0, atol=0)


def test_cache_unshared_kv_torch_validates_current_shape() -> None:
    key_cache = torch.zeros((4, 3, 2, 2, 8), dtype=torch.float16)
    value_cache = torch.zeros_like(key_cache)
    curr_key = torch.zeros((5, 2, 8), dtype=torch.float16)
    curr_value = torch.zeros_like(curr_key)
    block_table = torch.tensor([0, 1], dtype=torch.int32)
    decode_step = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(ValueError, match="curr_key must have shape"):
        cache_unshared_kv_torch(
            key_cache,
            value_cache,
            curr_key,
            curr_value,
            block_table,
            decode_step,
        )
