# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Duplicate-safe, bit-exact all-layer CUDA KV gather/scatter primitives."""

import triton
import triton.language as tl

# mypy: disable-error-code="misc,no-untyped-def"


@triton.jit
def _select_kv_ping_pong_kernel(
    pool_k_ptrs,
    pool_v_ptrs,
    standby_k_ptrs,
    standby_v_ptrs,
    pool_slot_offset,
    n,
    steps,
    beam_index_ptr,
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D,
    stride_pw,
    stride_ph,
    stride_ps,
    active_mask=None,
    execute=None,
    DEVICE_STEPS: tl.constexpr = False,
):
    """Gather all-layer parent histories directly into the next Decode bank.

    Banks have identical layouts and disjoint, persistent addresses. Selecting
    them from the existing step counter needs neither a pointer-table update
    nor a bank-switch kernel, including under CUDA graph replay.
    """
    pid = tl.program_id(0)
    per_layer = n * H
    layer = pid // per_layer
    i = (pid % per_layer) // H
    h = pid % H
    if DEVICE_STEPS:
        step = tl.load(steps)
        enabled = tl.load(execute) & tl.load(active_mask + i)
        steps = tl.where(enabled, step + 1, 0)
    else:
        step = steps - 1
    bank = step & 1
    k0 = tl.load(pool_k_ptrs + layer)
    v0 = tl.load(pool_v_ptrs + layer)
    k1 = tl.load(standby_k_ptrs + layer)
    v1 = tl.load(standby_v_ptrs + layer)
    k_src = (tl.where(bank == 0, k0, k1) + pool_slot_offset).to(tl.pointer_type(tl.uint16))
    v_src = (tl.where(bank == 0, v0, v1) + pool_slot_offset).to(tl.pointer_type(tl.uint16))
    k_dst = (tl.where(bank == 0, k1, k0) + pool_slot_offset).to(tl.pointer_type(tl.uint16))
    v_dst = (tl.where(bank == 0, v1, v0) + pool_slot_offset).to(tl.pointer_type(tl.uint16))
    parent = tl.load(beam_index_ptr + i)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    src_off = parent * stride_pw + h * stride_ph
    dst_off = i * stride_pw + h * stride_ph
    for s in range(steps):
        src = src_off + s * stride_ps + d
        dst = dst_off + s * stride_ps + d
        tl.store(k_dst + dst, tl.load(k_src + src, mask=mask), mask=mask)
        tl.store(v_dst + dst, tl.load(v_src + src, mask=mask), mask=mask)


@triton.jit
def _select_kv_gather_kernel(
    pool_k_ptrs,  # int64 [L]: per-layer key pool base byte addresses
    pool_v_ptrs,  # int64 [L]: per-layer value pool base byte addresses
    aux_k_addr,  # int64: aux key scratch base byte address ([L, W, H, S, D])
    aux_v_addr,  # int64: aux value scratch base byte address
    pool_slot_offset,  # byte offset of the active batch slot
    n,  # live beam count (<= pool width)
    steps,  # reorder range = decode_step + 1
    beam_index_ptr,  # [n] source beam for each destination beam
    H: tl.constexpr,  # num_kv_heads
    BLOCK_D: tl.constexpr,  # next_power_of_2(head_dim)
    D,  # head_dim (masked against BLOCK_D)
    stride_pw,
    stride_ph,
    stride_ps,  # pool strides (beam/head/step)
    stride_al,
    stride_aw,
    stride_ah,
    stride_as,  # aux strides (layer-first)
    active_mask=None,
    execute=None,
    DEVICE_STEPS: tl.constexpr = False,
):
    pid = tl.program_id(0)  # layer * (n * H) + i * H + h
    per_layer = n * H
    layer = pid // per_layer
    rem = pid % per_layer
    i = rem // H
    h = rem % H
    j = tl.load(beam_index_ptr + i)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    if DEVICE_STEPS:
        enabled = tl.load(execute) & tl.load(active_mask + i)
        steps = tl.where(enabled, tl.load(steps) + 1, 0)
    # int64 byte addresses → uint16 pointers: element offsets map 1:1 for
    # any 2-byte dtype, so the copy is bit-exact without a dtype anchor
    # (a typed aux pointer would silently CONVERT uint16→bf16 values).
    k_src = (tl.load(pool_k_ptrs + layer) + pool_slot_offset).to(tl.pointer_type(tl.uint16))
    v_src = (tl.load(pool_v_ptrs + layer) + pool_slot_offset).to(tl.pointer_type(tl.uint16))
    k_dst = aux_k_addr.to(tl.pointer_type(tl.uint16))
    v_dst = aux_v_addr.to(tl.pointer_type(tl.uint16))
    src_off = j * stride_pw + h * stride_ph
    dst_off = layer * stride_al + i * stride_aw + h * stride_ah
    for s in range(steps):
        src = src_off + s * stride_ps + d
        dst = dst_off + s * stride_as + d
        tl.store(k_dst + dst, tl.load(k_src + src, mask=mask), mask=mask)
        tl.store(v_dst + dst, tl.load(v_src + src, mask=mask), mask=mask)


@triton.jit
def _select_kv_scatter_kernel(
    pool_k_ptrs,  # int64 [L]: per-layer key pool base byte addresses
    pool_v_ptrs,  # int64 [L]: per-layer value pool base byte addresses
    aux_k_addr,  # int64: aux key scratch base byte address ([L, W, H, S, D])
    aux_v_addr,  # int64: aux value scratch base byte address
    pool_slot_offset,  # byte offset of the active batch slot
    n,  # live beam count (<= pool width)
    steps,  # reorder range = decode_step + 1
    H: tl.constexpr,  # num_kv_heads
    BLOCK_D: tl.constexpr,  # next_power_of_2(head_dim)
    D,  # head_dim (masked against BLOCK_D)
    stride_pw,
    stride_ph,
    stride_ps,  # pool strides (beam/head/step)
    stride_al,
    stride_aw,
    stride_ah,
    stride_as,  # aux strides (layer-first)
    active_mask=None,
    execute=None,
    DEVICE_STEPS: tl.constexpr = False,
):
    pid = tl.program_id(0)  # layer * (n * H) + i * H + h
    per_layer = n * H
    layer = pid // per_layer
    rem = pid % per_layer
    i = rem // H
    h = rem % H
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    if DEVICE_STEPS:
        enabled = tl.load(execute) & tl.load(active_mask + i)
        steps = tl.where(enabled, tl.load(steps) + 1, 0)
    k_dst = (tl.load(pool_k_ptrs + layer) + pool_slot_offset).to(tl.pointer_type(tl.uint16))
    v_dst = (tl.load(pool_v_ptrs + layer) + pool_slot_offset).to(tl.pointer_type(tl.uint16))
    k_src = aux_k_addr.to(tl.pointer_type(tl.uint16))
    v_src = aux_v_addr.to(tl.pointer_type(tl.uint16))
    src_off = layer * stride_al + i * stride_aw + h * stride_ah
    dst_off = i * stride_pw + h * stride_ph
    for s in range(steps):
        src = src_off + s * stride_as + d
        dst = dst_off + s * stride_ps + d
        tl.store(k_dst + dst, tl.load(k_src + src, mask=mask), mask=mask)
        tl.store(v_dst + dst, tl.load(v_src + src, mask=mask), mask=mask)
