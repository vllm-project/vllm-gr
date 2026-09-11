# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


pytestmark = pytest.mark.npu_integration


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
@pytest.mark.parametrize(
    "block_ids,num_blocks,seed",
    [
        pytest.param((3, 1), 4, 1234, id="multi-batch-noncontiguous-blocks"),
        pytest.param((0,), 2, 4321, id="single-batch-block-zero"),
    ],
)
def test_cache_unshared_kv_npu_inplace_write(
    dtype: torch.dtype,
    block_ids: tuple[int, ...],
    num_blocks: int,
    seed: int,
) -> None:
    """Each write preserves earlier steps and blocks outside the batch."""
    torch.manual_seed(seed)

    batch = len(block_ids)
    beam_size = 128
    kv_heads = 8
    head_dim = 128
    max_decode_steps = 3
    pool_shape = (num_blocks, beam_size, kv_heads, max_decode_steps, head_dim)

    key_pool_npu = torch.zeros(pool_shape, dtype=dtype, device=DEVICE)
    value_pool_npu = torch.zeros(pool_shape, dtype=dtype, device=DEVICE)
    key_pool_golden = torch.zeros(pool_shape, dtype=dtype)
    value_pool_golden = torch.zeros(pool_shape, dtype=dtype)
    block_table = torch.tensor(block_ids, dtype=torch.int32)

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
        # Comparing the whole pool after every write also checks all previous
        # steps and untouched blocks without keeping device snapshots.
        torch.testing.assert_close(key_pool_npu.cpu(), key_pool_golden, rtol=0, atol=0)
        torch.testing.assert_close(value_pool_npu.cpu(), value_pool_golden, rtol=0, atol=0)
