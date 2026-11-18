# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disallow-untyped-decorators=False
"""Paged-KV gathers on the vllm-ascend 0.30 interleaved cache layout.

vllm-ascend 0.30 allocates each layer's KV pool as one
``[num_blocks, 2, block, H, D]`` tensor and hands attention the strided
halves ``base[:, 0]`` / ``base[:, 1]``. Gathers must handle that layout
without materializing the whole pool:

  * ``interleaved_pool_base`` reconstructs the contiguous base from the two
    per-layer views, so ``index_select``/advanced indexing copy only the
    referenced blocks/tokens instead of contiguous-izing the source first.

Runs on CPU (layout math is pure torch) and on NPU when available.
"""

from __future__ import annotations

import pytest
import torch

from tools.hardware_marks import hardware_marks
from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    interleaved_pool_base,
)


DEVICES = ["cpu"]
try:
    if torch.npu.is_available():
        DEVICES.append(pytest.param("npu", marks=hardware_marks(res={"npu": "A3"}, num_cards=1)))
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

    # Contiguous CUDA and synthetic cache layouts have no interleaved base.
    kc = torch.randn(num_blocks, block, heads, head_dim, device=device)
    vc = torch.randn(num_blocks, block, heads, head_dim, device=device)
    assert interleaved_pool_base(kc, vc) is None
    # Mixed layouts (one contiguous, one strided): no base.
    assert interleaved_pool_base(kc, v) is None
    # Mismatched shapes: no base.
    assert interleaved_pool_base(k, vc[:2]) is None


pytestmark = pytest.mark.cpu_test
