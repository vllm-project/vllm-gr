# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA kernels for paged prefix attention: empty-state repair and segmented execution."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm_gr.v1.attention.backends.beam_attention_state import AttentionState

# Elements per merge program; the launch grid and the kernel must agree.
_MERGE_BLOCK = 256


@triton.jit
def _prepare(
    CuQ,
    Lengths,
    Table,
    SplitCuQ,
    SplitLengths,
    SplitTable,
    B: tl.constexpr,
    T: tl.constexpr,
    PAGE: tl.constexpr,
    COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    split, request = tl.program_id(0), tl.program_id(1)
    virtual = split * B + request
    start = tl.load(CuQ + request)
    tl.store(SplitCuQ + virtual, split * T + start)
    if (split == tl.num_programs(0) - 1) & (request == B - 1):
        tl.store(SplitCuQ + virtual + 1, (split + 1) * T)
    length = tl.load(Lengths + request)
    length = tl.minimum(tl.maximum(length - split * CHUNK * PAGE, 0), CHUNK * PAGE)
    tl.store(SplitLengths + virtual, length)
    col = tl.arange(0, BLOCK)
    source = split * CHUNK + col
    block = tl.load(
        Table + request * TABLE_STRIDE + source,
        (col < CHUNK) & (source < COLS) & (col * PAGE < length),
        other=0,
    )
    tl.store(SplitTable + virtual * CHUNK + col, block, col < CHUNK)


@triton.jit
def _merge(
    Partial,
    LSE,
    CuQ,
    Lengths,
    Out,
    OutLSE,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    PS2: tl.constexpr,
    LS0: tl.constexpr,
    LS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per (request, head, tile): the former serial tile loop left
    # only batch*heads CTAs (16 for B=1), making the merge as slow as the FA call.
    request, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = tl.load(CuQ + request), tl.load(CuQ + request + 1)
    length = tl.load(Lengths + request)
    offsets = tl.arange(0, BLOCK)
    if tile * BLOCK < (end - start) * D:
        flat = tile * BLOCK + offsets
        token, dim = start + flat // D, flat % D
        valid = token < end
        maximum = tl.full((BLOCK,), -float("inf"), tl.float32)
        denominator = tl.full((BLOCK,), 0, tl.float32)
        accumulator = tl.full((BLOCK,), 0, tl.float32)
        for split in range(S):
            # Use lengths, not FA2's potentially uninitialized empty LSE.
            live = valid & (length > split * CHUNK_TOKENS)
            lse = tl.load(LSE + head * LS0 + (split * T + token) * LS1, live, other=-float("inf"))
            present = live & (lse != float("inf")) & (lse != -float("inf"))
            lse = tl.where(present, lse, -float("inf"))
            next_max = tl.maximum(maximum, lse)
            safe_max = tl.where(next_max == -float("inf"), 0, next_max)
            old_weight = tl.exp(maximum - safe_max)
            new_weight = tl.exp(lse - safe_max)
            value = tl.load(
                Partial + (split * T + token) * PS0 + head * PS1 + dim * PS2, present, other=0
            ).to(tl.float32)
            accumulator = accumulator * old_weight + value * new_weight
            denominator = denominator * old_weight + new_weight
            maximum = next_max
        result = accumulator / tl.where(denominator > 0, denominator, 1)
        tl.store(Out + (token * H + head) * D + dim, result, valid)
        tl.store(OutLSE + head * T + token, maximum + tl.log(denominator), valid & (dim == 0))


def segmented_prefix(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_query: torch.Tensor,
    lengths: torch.Tensor,
    block_table: torch.Tensor,
    max_query: int,
    max_length: int,
    scale: float,
    fa_version: int,
    segments: int,
    flash,
) -> AttentionState:
    """Plain noncausal prefix only; KV pages are shared, never copied.

    Query/state workspace grows by S. Split-major virtual requests preserve
    arbitrary request query lengths; all segments run in ONE FA invocation.
    Empty and partial last segments are masked during merge before LSE loads.
    """
    if type(segments) is not int or segments < 2:
        raise ValueError("segments must be an integer >= 2")
    batch, columns = block_table.shape
    tokens, heads, dim = query.shape
    chunk = max(1, triton.cdiv(triton.cdiv(max_length, key.shape[1]), segments))
    # A segment layout may only cover pages the caller actually allocated.
    # Otherwise the last virtual request would read block-table entries that do
    # not exist (silently clamped to page 0 by _prepare) and produce wrong
    # outputs instead of an error.
    if chunk * segments > columns:
        raise ValueError(
            f"segment layout needs {chunk * segments} pages but the block table "
            f"only holds {columns}"
        )
    split_query = query.repeat(segments, 1, 1)
    split_cu = torch.empty(segments * batch + 1, dtype=torch.int32, device=query.device)
    split_lengths = torch.empty(segments * batch, dtype=torch.int32, device=query.device)
    split_table = torch.empty((segments * batch, chunk), dtype=torch.int32, device=query.device)
    _prepare[(segments, batch)](
        cu_query,
        lengths,
        block_table,
        split_cu,
        split_lengths,
        split_table,
        batch,
        tokens,
        key.shape[1],
        columns,
        block_table.stride(0),
        chunk,
        triton.next_power_of_2(chunk),
    )
    out, lse = flash(
        q=split_query,
        k=key,
        v=value,
        cu_seqlens_q=split_cu,
        seqused_k=split_lengths,
        max_seqlen_q=max_query,
        max_seqlen_k=chunk * key.shape[1],
        block_table=split_table,
        softmax_scale=scale,
        causal=False,
        return_softmax_lse=True,
        fa_version=fa_version,
        num_splits=1,
    )
    result = torch.empty_like(query, memory_format=torch.contiguous_format)
    result_lse = torch.empty((heads, tokens), dtype=torch.float32, device=query.device)
    _merge[(batch, heads, triton.cdiv(max_query * dim, _MERGE_BLOCK))](
        out,
        lse,
        cu_query,
        lengths,
        result,
        result_lse,
        tokens,
        heads,
        dim,
        segments,
        chunk * key.shape[1],
        *out.stride(),
        *lse.stride(),
        _MERGE_BLOCK,
    )
    return AttentionState(result, result_lse)


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
