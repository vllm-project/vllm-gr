# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA Graph-compatible KV writes and packed suffix copies for Beam decode.

cache_unshared_kv_kernel: Triton JIT kernel that writes the current step's
    beam KV into the pool's ``[slot, beam, head, decode_step, :]`` slice,
    reading ``decode_step`` from a device-side buffer (not a host Python int).

This makes the kernel compatible with CUDA Graph capture/replay: the
``decode_step`` value can change between replays via ``decode_step_buf.fill_()``
without re-capturing the graph.

The eager path uses ``write_eager_step`` in beam_attention_pool.py.
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


@triton.jit
def cache_unshared_kv_kernel(
    pool_ptr,  # pool base pointer (unshared_key or unshared_value)
    curr_ptr,  # current step KV [W, H, D]
    W: tl.constexpr,  # beam width
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
    # NOTE: the last (head_dim) pool dimension is assumed contiguous
    # (stride == 1); the pool is allocated as one contiguous 5D tensor.
):
    """Write current step's beam KV into the pool at [slot, beam, head, decode_step, :].

    Each program instance handles one (beam, head) pair, writing D elements.
    Grid size = W * H.
    """
    pid = tl.program_id(0)
    total = W * H
    if pid >= total:
        return

    beam = pid // H
    head = pid % H

    # Device-side read of decode_step (key enabler for CUDA Graph replay)
    decode_step = tl.load(decode_step_buf).to(tl.int32)
    # Defense-in-depth: a stale decode_step_buf under CUDA Graph replay
    # could exceed max_steps; skip to avoid OOB pool writes.
    if decode_step < 0 or decode_step >= max_steps:
        return

    if MASKED:
        if not tl.load(active_mask + beam):
            return

    # Source: curr[beam, head, :] — [D] contiguous
    src_off = beam * H * D + head * D + tl.arange(0, D)
    src = tl.load(curr_ptr + src_off)

    # Destination: pool[slot, beam, head, decode_step, :]
    dst_off = (
        slot * stride_b
        + beam * stride_w
        + head * stride_h
        + decode_step * stride_s
        + tl.arange(0, D)
    )
    tl.store(pool_ptr + dst_off, src)


def cache_unshared_kv_graph(
    pool_key: torch.Tensor,  # [max_batch, W, H, max_steps, D]
    pool_value: torch.Tensor,  # [max_batch, W, H, max_steps, D]
    curr_key: torch.Tensor,  # [W, H, D]
    curr_value: torch.Tensor,  # [W, H, D]
    slot: int,  # session slot index
    decode_step_buf: torch.Tensor,  # [1] int32 device tensor
    active_mask: torch.Tensor | None = None,
) -> None:
    """Graph-compatible cache_unshared_kv using Triton kernels.

    Writes current step's beam KV into the pool at [slot, :, :, decode_step, :],
    where decode_step is read from the device buffer at replay time.

    This function is safe to call both in eager mode and during CUDA Graph
    capture/replay — no host-side Python ints are baked into kernel args
    (except `slot`, which is constant across all steps of a session).
    """
    W, H, D = curr_key.shape

    grid = (W * H,)

    # Launch key write
    cache_unshared_kv_kernel[grid](
        pool_key,
        curr_key,
        W=W,
        H=H,
        D=D,
        max_steps=pool_key.shape[3],
        slot=slot,
        decode_step_buf=decode_step_buf,
        stride_b=pool_key.stride(0),
        stride_w=pool_key.stride(1),
        stride_h=pool_key.stride(2),
        stride_s=pool_key.stride(3),
        active_mask=active_mask,
        MASKED=active_mask is not None,
    )

    # Launch value write
    cache_unshared_kv_kernel[grid](
        pool_value,
        curr_value,
        W=W,
        H=H,
        D=D,
        max_steps=pool_value.shape[3],
        slot=slot,
        decode_step_buf=decode_step_buf,
        stride_b=pool_value.stride(0),
        stride_w=pool_value.stride(1),
        stride_h=pool_value.stride(2),
        stride_s=pool_value.stride(3),
        active_mask=active_mask,
        MASKED=active_mask is not None,
    )
