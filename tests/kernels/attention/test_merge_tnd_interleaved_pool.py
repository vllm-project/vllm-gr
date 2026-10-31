# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disallow-untyped-decorators=False
"""Paged-KV gathers on the vllm-ascend 0.30 interleaved cache layout.

vllm-ascend 0.30 allocates each layer's KV pool as one
``[num_blocks, 2, block, H, D]`` tensor and hands attention the strided
halves ``base[:, 0]`` / ``base[:, 1]``. Gathers must handle that layout
without materializing the whole pool:

  * ``interleaved_pool_base`` reconstructs the contiguous base from the two
    per-layer views, so ``index_select``/advanced indexing copy only the
    referenced blocks/tokens instead of contiguous-izing the source first.
  * ``extract_suffix_kv_pytorch`` (the beam recompute-path suffix gather)
    flattens the cache with ``.view(-1, H, D)``, which is not expressible
    on the strided halves at all -- it raised "view size is not compatible"
    before the base-aware path existed.

Runs on CPU (layout math is pure torch) and on NPU when available.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    extract_suffix_kv_pytorch,
    interleaved_pool_base,
)

DEVICES = ["cpu"]
try:
    if torch.npu.is_available():
        DEVICES.append("npu")
except AttributeError:
    pass


def _make_interleaved(
    num_blocks: int,
    block: int,
    heads: int,
    head_dim: int,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strided per-layer K/V views plus the CPU pool they alias."""
    torch.manual_seed(seed)
    pool = torch.randn(num_blocks, 2, block, heads, head_dim, dtype=torch.float32)
    pool_dev = pool.to(device)
    return pool_dev[:, 0], pool_dev[:, 1], pool


@pytest.mark.parametrize("device", DEVICES)
def test_interleaved_pool_base_detects_and_rejects(device: str) -> None:
    num_blocks, block, heads, head_dim = 5, 4, 2, 8
    k, v, _ = _make_interleaved(num_blocks, block, heads, head_dim, device)

    base = interleaved_pool_base(k, v)
    assert base is not None
    assert base.is_contiguous()
    assert tuple(base.shape) == (num_blocks, 2, block, heads, head_dim)
    assert torch.equal(base[:, 0], k)
    assert torch.equal(base[:, 1], v)

    # Contiguous caches (CUDA layout, pre-0.30 NPU, synthetic tests): no base.
    kc = torch.randn(num_blocks, block, heads, head_dim, device=device)
    vc = torch.randn(num_blocks, block, heads, head_dim, device=device)
    assert interleaved_pool_base(kc, vc) is None
    # Mixed layouts (one contiguous, one strided): no base.
    assert interleaved_pool_base(kc, v) is None
    # Mismatched shapes: no base.
    assert interleaved_pool_base(k, vc[:2]) is None


@pytest.mark.parametrize("device", DEVICES)
def test_extract_suffix_kv_on_interleaved_layout(device: str) -> None:
    num_blocks, block, heads, head_dim = 6, 4, 2, 8
    k, v, pool_cpu = _make_interleaved(num_blocks, block, heads, head_dim, device, seed=3)

    slots = torch.tensor([0, 5, 17, 23, num_blocks * block - 1], device=device, dtype=torch.long)
    out_k, out_v = extract_suffix_kv_pytorch(k, v, slots)

    # Reference: flat token t lives in block t // block at offset t % block,
    # K in half 0 and V in half 1 of the contiguous CPU pool.
    flat = pool_cpu.permute(0, 2, 1, 3, 4).reshape(num_blocks * block, 2, heads, head_dim)
    slots_cpu = slots.cpu()
    assert torch.equal(out_k.cpu(), flat[slots_cpu, 0])
    assert torch.equal(out_v.cpu(), flat[slots_cpu, 1])


@pytest.mark.parametrize("device", DEVICES)
def test_extract_suffix_kv_contiguous_unchanged(device: str) -> None:
    num_blocks, block, heads, head_dim = 4, 8, 2, 16
    k = torch.randn(num_blocks, block, heads, head_dim, device=device)
    v = torch.randn(num_blocks, block, heads, head_dim, device=device)
    slots = torch.tensor([3, 10, 31], device=device, dtype=torch.long)

    out_k, out_v = extract_suffix_kv_pytorch(k, v, slots)
    assert torch.equal(out_k, k.view(-1, heads, head_dim)[slots])
    assert torch.equal(out_v, v.view(-1, heads, head_dim)[slots])
