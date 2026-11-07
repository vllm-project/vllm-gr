# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright 2025 The xLLM Authors. All Rights Reserved.
# Adapted from https://github.com/xLLM-AI/xllm-ops

"""NPU precision guard tests for the fused ``cache_unshared_kv`` op.

Derived from xllm-ops' ``test_cache_unshared_kv.py``.  The op writes the
current step's K/V for every beam into the fixed unshared beam pool using
``pool[block_table[b], :, :, decode_step - 1, :] = curr[b*beam:(b+1)*beam]``
(in-place, 1-based decode step).
"""

from __future__ import annotations

import pytest
import torch

# pytest parametrization decorators are untyped in the strict mypy environment.
# mypy: disable-error-code="misc"


pytestmark = pytest.mark.npu_test


try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is not available", allow_module_level=True)

try:
    import custom_ops_lib as custom_ops
except ImportError as exc:
    pytest.skip(f"custom_ops_lib is not installed: {exc}", allow_module_level=True)


DEVICE = torch.device("npu")
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("max_decode_steps", [3, 4, 6])
def test_cache_unshared_kv_npu_inplace_write(dtype: torch.dtype, max_decode_steps: int) -> None:
    """Every decode step writes its per-beam K/V slice; nothing else changes."""
    torch.manual_seed(1234)

    batch = 2
    beam_size = 128
    kv_heads = 8
    head_dim = 128
    num_blocks = 4
    pool_shape = (num_blocks, beam_size, kv_heads, max_decode_steps, head_dim)

    key_pool_npu = torch.zeros(pool_shape, dtype=dtype, device=DEVICE)
    value_pool_npu = torch.zeros(pool_shape, dtype=dtype, device=DEVICE)
    key_pool_golden = torch.zeros(pool_shape, dtype=dtype)
    value_pool_golden = torch.zeros(pool_shape, dtype=dtype)
    block_table = torch.tensor([3, 1], dtype=torch.int32)

    for decode_step in range(1, max_decode_steps + 1):
        curr_shape = (batch * beam_size, kv_heads, head_dim)
        curr_key = torch.rand(curr_shape, dtype=dtype)
        curr_value = torch.rand(curr_shape, dtype=dtype)

        for batch_idx, block_idx in enumerate(block_table.tolist()):
            start = batch_idx * beam_size
            stop = start + beam_size
            key_pool_golden[block_idx, :, :, decode_step - 1, :] = curr_key[start:stop]
            value_pool_golden[block_idx, :, :, decode_step - 1, :] = curr_value[start:stop]

        returned_key, returned_value = custom_ops.cache_unshared_kv(
            key_pool_npu,
            value_pool_npu,
            curr_key.to(DEVICE),
            curr_value.to(DEVICE),
            block_table.to(DEVICE),
            torch.tensor([decode_step], dtype=torch.int32, device=DEVICE),
        )
        torch.npu.synchronize()

        # The op mutates the pools in place and returns them.
        assert returned_key is key_pool_npu
        assert returned_value is value_pool_npu
        torch.testing.assert_close(key_pool_npu.cpu(), key_pool_golden, rtol=0, atol=0)
        torch.testing.assert_close(value_pool_npu.cpu(), value_pool_golden, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("max_decode_steps", [3, 4, 6])
def test_cache_unshared_kv_npu_preserves_previous_steps(
    dtype: torch.dtype, max_decode_steps: int
) -> None:
    """A later write must not clobber slices written at earlier steps."""
    torch.manual_seed(4321)

    batch = 1
    beam_size = 128
    kv_heads = 8
    head_dim = 128
    pool_shape = (2, beam_size, kv_heads, max_decode_steps, head_dim)
    block_table = torch.tensor([0], dtype=torch.int32)

    key_pool_npu = torch.zeros(pool_shape, dtype=dtype, device=DEVICE)
    value_pool_npu = torch.zeros(pool_shape, dtype=dtype, device=DEVICE)
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []

    for decode_step in range(1, max_decode_steps + 1):
        curr_key = torch.rand((batch * beam_size, kv_heads, head_dim), dtype=dtype)
        curr_value = torch.rand((batch * beam_size, kv_heads, head_dim), dtype=dtype)
        custom_ops.cache_unshared_kv(
            key_pool_npu,
            value_pool_npu,
            curr_key.to(DEVICE),
            curr_value.to(DEVICE),
            block_table.to(DEVICE),
            torch.tensor([decode_step], dtype=torch.int32, device=DEVICE),
        )
        torch.npu.synchronize()
        snapshots.append((key_pool_npu.clone().cpu(), value_pool_npu.clone().cpu()))

    # After the final step, every earlier snapshot must still be a prefix of
    # the pool's time axis.
    final_key, final_value = snapshots[-1]
    for step_index, (key_snapshot, value_snapshot) in enumerate(snapshots[:-1]):
        torch.testing.assert_close(
            final_key[:, :, :, : step_index + 1, :],
            key_snapshot[:, :, :, : step_index + 1, :],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            final_value[:, :, :, : step_index + 1, :],
            value_snapshot[:, :, :, : step_index + 1, :],
            rtol=0,
            atol=0,
        )
