# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright 2025 The xLLM Authors. All Rights Reserved.
# Adapted from https://github.com/xLLM-AI/xllm-ops

"""NPU precision guard tests for the fused ``select_unshared_kv`` op.

Derived from xllm-ops' ``test_select_unshared_kv.py``.  The op rearranges the
unshared beam KV pools of every layer in place after a beam selection, driven
by ``group_token_num`` prefix sums.  The CPU golden is the package's own
``_select_unshared_kv_torch`` reference, whose semantics were already locked
down by ``test_onerec_decode_ops.py``.
"""

from __future__ import annotations

import pytest
import torch

from tools.hardware_marks import hardware_marks
from vllm_gr.ops.onerec_decode import _select_unshared_kv_torch


# mypy: disable-error-code="misc"


pytestmark = hardware_marks(res={"npu": "A3"}, num_cards=1)


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


def _sorted_parents(batch: int, beam_size: int) -> torch.Tensor:
    """Beam sources per target slot, sorted ascending within each batch."""
    parents = torch.randint(0, beam_size, (batch, beam_size), dtype=torch.int32)
    return torch.sort(parents, dim=1).values


def _prefix_from_parents(parents: torch.Tensor, beam_size: int) -> torch.Tensor:
    """Request-local prefix sums consumed by select_unshared_kv."""
    counts = torch.stack([torch.bincount(row, minlength=beam_size).cumsum(0) for row in parents])
    return counts.to(torch.int32).reshape(-1)


@pytest.mark.parametrize("batch,layer_num", [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("max_decode_steps", [3, 4, 6])
def test_select_unshared_kv_npu_matches_torch_reference(
    batch: int,
    layer_num: int,
    max_decode_steps: int,
) -> None:
    """Multi-step cache -> select pipeline must match the CPU reference."""
    torch.manual_seed(1234)
    dtype = torch.bfloat16

    beam_size = 128
    kv_heads = 8
    head_dim = 128
    num_blocks = batch
    pool_shape = (num_blocks, beam_size, kv_heads, max_decode_steps, head_dim)
    block_table = torch.arange(batch, dtype=torch.int32)

    golden_keys = [torch.zeros(pool_shape, dtype=dtype) for _ in range(layer_num)]
    golden_values = [torch.zeros(pool_shape, dtype=dtype) for _ in range(layer_num)]
    npu_keys = [torch.zeros(pool_shape, dtype=dtype, device=DEVICE) for _ in range(layer_num)]
    npu_values = [torch.zeros(pool_shape, dtype=dtype, device=DEVICE) for _ in range(layer_num)]

    for decode_step in range(1, max_decode_steps + 1):
        # --- cache stage: fill this step's slice for every beam (CPU fill,
        # mirrored into both pools; mirrors cache_unshared_kv semantics).
        for layer in range(layer_num):
            curr_key = torch.rand((batch * beam_size, kv_heads, head_dim), dtype=dtype)
            curr_value = torch.rand((batch * beam_size, kv_heads, head_dim), dtype=dtype)
            for batch_idx, block_idx in enumerate(block_table.tolist()):
                start = batch_idx * beam_size
                stop = start + beam_size
                golden_keys[layer][block_idx, :, :, decode_step - 1, :] = curr_key[start:stop]
                golden_values[layer][block_idx, :, :, decode_step - 1, :] = curr_value[start:stop]
            npu_keys[layer].copy_(golden_keys[layer].to(DEVICE))
            npu_values[layer].copy_(golden_values[layer].to(DEVICE))

        # --- select stage: same selection applied to both pools.
        parents = _sorted_parents(batch, beam_size)
        beam_index = parents.reshape(-1)
        group_token_num = _prefix_from_parents(parents, beam_size)

        _select_unshared_kv_torch(
            beam_index,
            golden_keys,
            golden_values,
            block_table,
            group_token_num,
            decode_step,
            beam_size,
            layer_num,
        )
        result = custom_ops.select_unshared_kv(
            beam_index.to(DEVICE),
            npu_keys,
            npu_values,
            block_table.to(DEVICE),
            group_token_num.to(DEVICE),
            decode_step,
            beam_size,
            layer_num,
        )
        torch.npu.synchronize()
        assert result is None  # in-place contract

        # Only the referenced pool slots and the already-written time steps
        # are governed by the op contract.
        for layer in range(layer_num):
            expected_key = golden_keys[layer][block_table.to(torch.int64)]
            expected_value = golden_values[layer][block_table.to(torch.int64)]
            torch.testing.assert_close(
                npu_keys[layer].cpu()[block_table.to(torch.int64)],
                expected_key,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                npu_values[layer].cpu()[block_table.to(torch.int64)],
                expected_value,
                rtol=0,
                atol=0,
            )


def test_select_unshared_kv_npu_untouched_blocks_stay_intact() -> None:
    """Blocks not referenced by block_table must not be modified."""
    torch.manual_seed(4321)
    dtype = torch.bfloat16

    batch, beam_size, kv_heads, head_dim = 1, 128, 8, 128
    max_decode_steps, num_blocks, layer_num = 2, 3, 1
    pool_shape = (num_blocks, beam_size, kv_heads, max_decode_steps, head_dim)
    block_table = torch.tensor([1], dtype=torch.int32)

    key_pool = torch.rand(pool_shape, dtype=dtype)
    value_pool = torch.rand(pool_shape, dtype=dtype)
    npu_key = key_pool.clone().to(DEVICE)
    npu_value = value_pool.clone().to(DEVICE)

    parents = _sorted_parents(batch, beam_size)
    group_token_num = _prefix_from_parents(parents, beam_size)

    custom_ops.select_unshared_kv(
        parents.reshape(-1).to(DEVICE),
        [npu_key],
        [npu_value],
        block_table.to(DEVICE),
        group_token_num.to(DEVICE),
        1,
        beam_size,
        layer_num,
    )
    torch.npu.synchronize()

    untouched = [block for block in range(num_blocks) if block not in block_table.tolist()]
    for block in untouched:
        torch.testing.assert_close(npu_key.cpu()[block], key_pool[block], rtol=0, atol=0)
        torch.testing.assert_close(npu_value.cpu()[block], value_pool[block], rtol=0, atol=0)
