# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA Graph-compatible KV operators for incremental beam decode.

cache_unshared_kv_kernel: Triton JIT kernel that gathers the current step's
    beam rows and writes K and V directly into the pool's
    ``[slot, beam, head, decode_step, :]`` slice.

This makes the kernel compatible with CUDA Graph capture/replay: the
``decode_step`` value can change between replays via ``decode_step_buf.fill_()``
without re-capturing the graph.

The original pure-Python ``cache_unshared_kv`` (in beam_kernels.py) is
retained as a fallback and for non-Graph eager paths.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def copy_suffix_kv_kernel(
    pool_ptr,  # pool base pointer (unshared_key or unshared_value)
    buf_ptr,  # suffix buffer base pointer [W*max_steps, H, D]
    W: tl.constexpr,  # session beam width (NOT the pool's max_beams)
    H: tl.constexpr,  # num_kv_heads
    D: tl.constexpr,  # head_dim
    max_steps: tl.constexpr,  # max_decode_steps (pool dim 3)
    slot: tl.constexpr,  # fixed int: which session slot in pool
    decode_step_buf,  # device tensor [1] int32: decode_step_buf
    stride_b,  # pool stride for batch dim
    stride_w,  # pool stride for beam dim
    stride_h,  # pool stride for head dim
    stride_s,  # pool stride for step dim
    active_mask,
    MASKED: tl.constexpr,
):
    """Copy the valid suffix KV rows from the pool into the packed suffix buffer.

    Pool layout:  [max_batch, max_beams, H, max_steps, D]
    Buffer layout (packed): [W*(step+1), H, D] — beam b's rows occupy
    [b*(step+1), (b+1)*(step+1)), W = the session's beam width.

    ``step`` is read from the device-side ``decode_step_buf`` so the same
    captured kernel works for every decode step.  The grid always covers all
    W*max_steps*H (beam, step_idx, head) rows; rows with step_idx > step are
    skipped at runtime.

    This replaces the eager host-side copy
    ``suffix_buf[:W*(step+1)] = pool[slot,:W,:,:step+1,:].permute(0,2,1,3)``,
    whose slice bounds are host Python ints that CUDA Graph capture would
    bake in (a graph captured at step=1 would forever copy 2 rows).
    """
    pid = tl.program_id(0)
    total = W * max_steps * H
    if pid >= total:
        return

    head = pid % H
    s = (pid // H) % max_steps
    beam = pid // (H * max_steps)

    step = tl.load(decode_step_buf).to(tl.int32)
    if MASKED:
        live = tl.load(active_mask + beam) & (step >= 0)
        extent = tl.maximum(step + 1, 1)
    else:
        live = True
        extent = step + 1
    if s >= extent:
        return
    # Defense-in-depth: a stale decode_step_buf under CUDA Graph replay
    # could exceed max_steps; skip to avoid OOB writes.
    if step >= max_steps:
        return

    src_off = slot * stride_b + beam * stride_w + head * stride_h + s * stride_s + tl.arange(0, D)
    src = tl.load(pool_ptr + src_off, mask=live, other=0)

    dst_row = beam * extent + s
    dst_off = (dst_row * H + head) * D + tl.arange(0, D)
    tl.store(buf_ptr + dst_off, src)


def copy_suffix_kv_graph(
    pool_key: torch.Tensor,  # [max_batch, max_beams, H, max_steps, D]
    pool_value: torch.Tensor,  # [max_batch, max_beams, H, max_steps, D]
    suffix_k_buf: torch.Tensor,  # [beam_width*max_steps, H, D]
    suffix_v_buf: torch.Tensor,  # [beam_width*max_steps, H, D]
    slot: int,  # session slot index
    decode_step_buf: torch.Tensor,  # [1] int32 device tensor
    beam_width: int | None = None,  # session beam width; None -> infer from buffer
    active_mask: torch.Tensor | None = None,
) -> None:
    """Graph-compatible pool→suffix_buf packed copy using a Triton kernel.

    ``W`` passed to the kernel is the SESSION's beam width, not the pool's
    max_beams: the grid and packed destination range must match the session's
    suffix buffer view (``beam_width * max_steps`` rows).  Copying the full
    pool width would both waste work (width=4 in a 128-wide pool copies 32x
    too much, also on the eager fallback path) and, once max_batch > 1, write
    into a NEIGHBOURING slot's suffix region.
    """
    _, _, H, max_steps, D = pool_key.shape
    if beam_width is None:
        beam_width = suffix_k_buf.shape[0] // max_steps
    grid = (beam_width * max_steps * H,)

    for pool, buf in ((pool_key, suffix_k_buf), (pool_value, suffix_v_buf)):
        copy_suffix_kv_kernel[grid](
            pool,
            buf,
            W=beam_width,
            H=H,
            D=D,
            max_steps=max_steps,
            slot=slot,
            decode_step_buf=decode_step_buf,
            stride_b=pool.stride(0),
            stride_w=pool.stride(1),
            stride_h=pool.stride(2),
            stride_s=pool.stride(3),
            active_mask=active_mask,
            MASKED=active_mask is not None,
        )


@triton.jit(do_not_specialize=["slot"])
def cache_unshared_kv_kernel(
    pool_key,
    pool_value,
    curr_key,
    curr_value,
    row_indices,
    slot,
    decode_step_buf,
    active_mask,
    key_strides: tl.constexpr,
    value_strides: tl.constexpr,
    pool_key_strides: tl.constexpr,
    pool_value_strides: tl.constexpr,
    index_stride: tl.constexpr,
    mask_stride: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    INDEXED: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    step = tl.load(decode_step_buf).to(tl.int32)
    if step < 0 or step >= MAX_STEPS:
        return

    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    beam, head = row // HEADS, row % HEADS
    live = beam < WIDTH
    if MASKED:
        live = live & tl.load(active_mask + beam * mask_stride, live, other=False)
    source_row = beam
    if INDEXED:
        source_row = tl.load(row_indices + beam * index_stride, live, other=0)
        live = live & (source_row >= 0) & (source_row < SOURCE_ROWS)
    source_row = source_row.to(tl.int64)
    slot = slot.to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    valid = live[:, None] & (d[None, :] < DIM)

    key_offset = source_row * key_strides[0] + head * key_strides[1]
    value_offset = source_row * value_strides[0] + head * value_strides[1]
    key = tl.load(curr_key + key_offset[:, None] + d[None, :] * key_strides[2], valid)
    value = tl.load(curr_value + value_offset[:, None] + d[None, :] * value_strides[2], valid)

    dst_key = (
        slot * pool_key_strides[0]
        + beam * pool_key_strides[1]
        + head * pool_key_strides[2]
        + step * pool_key_strides[3]
    )
    dst_value = (
        slot * pool_value_strides[0]
        + beam * pool_value_strides[1]
        + head * pool_value_strides[2]
        + step * pool_value_strides[3]
    )
    tl.store(pool_key + dst_key[:, None] + d[None, :] * pool_key_strides[4], key, valid)
    tl.store(pool_value + dst_value[:, None] + d[None, :] * pool_value_strides[4], value, valid)


def cache_unshared_kv_graph(
    pool_key: torch.Tensor,
    pool_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    slot: int,
    decode_step_buf: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    *,
    row_indices: torch.Tensor | None = None,
) -> None:
    """Gather the beam rows of K and V and write both to the private pool.

    Source K/V are [rows, heads, dim]. Row indices, when provided, map each
    destination beam to a source row. Step, row indices and active mask are
    read on device so their values can change between graph replays.
    """
    source_rows, heads, dim = curr_key.shape
    width = row_indices.numel() if row_indices is not None else source_rows
    block_rows = 4
    cache_unshared_kv_kernel[(triton.cdiv(width * heads, block_rows),)](
        pool_key,
        pool_value,
        curr_key,
        curr_value,
        row_indices,
        slot,
        decode_step_buf,
        active_mask,
        curr_key.stride(),
        curr_value.stride(),
        pool_key.stride(),
        pool_value.stride(),
        row_indices.stride(0) if row_indices is not None else 0,
        active_mask.stride(0) if active_mask is not None else 0,
        width,
        heads,
        dim,
        source_rows,
        pool_key.shape[3],
        row_indices is not None,
        active_mask is not None,
        block_rows,
        triton.next_power_of_2(dim),
    )
