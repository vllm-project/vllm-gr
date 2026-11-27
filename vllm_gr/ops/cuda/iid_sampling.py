# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Sample independent uniforms against shared parent CDFs."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["total", "width"])
def _search_parent_cdfs(cdf, parents, uniforms, columns, total, width, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parent = tl.load(parents + lane, lane < total, 0)
    u = tl.load(uniforms + lane, lane < total, 0)
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), 0, tl.int32) + width - 1
    # Last CDF entry is exactly one. Find the first entry strictly above u.
    span = width
    while span > 1:
        mid = (low + high) // 2
        left = u < tl.load(cdf + parent * width + mid)
        low = tl.where(left, low, tl.minimum(mid + 1, high))
        high = tl.where(left, mid, high)
        span = (span + 1) // 2
    tl.store(columns + lane, low, lane < total)


def search_parent_cdfs(
    cdf: torch.Tensor, parents: torch.Tensor, uniforms: torch.Tensor
) -> torch.Tensor:
    columns = torch.empty_like(parents)
    _search_parent_cdfs[(triton.cdiv(parents.numel(), 128),)](
        cdf, parents, uniforms, columns, parents.numel(), cdf.shape[1], BLOCK=128
    )
    return columns
