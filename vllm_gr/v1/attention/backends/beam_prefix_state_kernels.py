# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe repair of backend-specific empty prefix states."""

import torch
import triton
import triton.language as tl


@triton.jit
def _clear_empty_prefix(
    Out,
    LSE,
    CuQ,
    Lengths,
    out_strides: tl.constexpr,
    lse_strides: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    if tl.load(Lengths + request) == 0:
        start = tl.load(CuQ + request)
        end = tl.load(CuQ + request + 1)
        for block in range(tl.cdiv((end - start) * DIM, BLOCK)):
            offsets = block * BLOCK + tl.arange(0, BLOCK)
            token = start + offsets // DIM
            d = offsets % DIM
            valid = token < end
            tl.store(
                Out + token * out_strides[0] + head * out_strides[1] + d * out_strides[2],
                0.0,
                valid,
            )
            tl.store(
                LSE + head * lse_strides[0] + token * lse_strides[1],
                -float("inf"),
                valid & (d == 0),
            )


def clear_empty_prefix(
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_query_lens: torch.Tensor,
    kv_lens: torch.Tensor,
) -> None:
    if not out.numel():
        return
    heads, dim = out.shape[1:]
    _clear_empty_prefix[(kv_lens.numel(), heads)](
        out,
        lse,
        cu_query_lens,
        kv_lens,
        out.stride(),
        lse.stride(),
        dim,
        256,
    )
