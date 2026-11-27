# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Duplicate-safe, bit-exact all-layer CUDA KV gather/scatter primitives."""

import torch
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


def select_unshared_kv_ping_pong(
    beam_index: torch.Tensor,
    slot: int,
    unshared_key_caches: list[torch.Tensor],
    unshared_value_caches: list[torch.Tensor],
    standby_key_caches: list[torch.Tensor | None],
    standby_value_caches: list[torch.Tensor | None],
    decode_step: int,
    pool_key_ptrs: torch.Tensor,
    pool_value_ptrs: torch.Tensor,
    standby_key_ptrs: torch.Tensor,
    standby_value_ptrs: torch.Tensor,
    *,
    validate_layout: bool = True,
) -> None:
    """Write selected history into the bank consumed by the next Decode.

    The step selects both source and destination, so callers retain the same
    tensor objects and pointer tables for every session and graph replay.
    """
    if not unshared_key_caches:
        raise ValueError("KV ping-pong requires at least one layer")
    first = unshared_key_caches[0]
    layers = len(unshared_key_caches)
    banks = (unshared_key_caches, unshared_value_caches, standby_key_caches, standby_value_caches)
    if not first.is_cuda or first.element_size() != 2:
        raise ValueError("KV ping-pong requires two-byte CUDA pools")
    if any(len(bank) != layers for bank in banks):
        raise ValueError("KV ping-pong layer counts must agree")
    if validate_layout:
        ranges = []
        for bank in banks:
            for tensor in bank:
                if (
                    tensor is None
                    or tensor.shape != first.shape
                    or tensor.stride() != first.stride()
                    or tensor.dtype != first.dtype
                    or tensor.device != first.device
                ):
                    raise ValueError("KV ping-pong banks must have identical layouts")
                start = tensor.data_ptr()
                ranges.append((start, start + tensor.numel() * tensor.element_size()))
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("KV ping-pong banks must not overlap")
        for ptrs in (pool_key_ptrs, pool_value_ptrs, standby_key_ptrs, standby_value_ptrs):
            if (
                ptrs.shape != (layers,)
                or ptrs.dtype != torch.int64
                or ptrs.device != first.device
                or not ptrs.is_contiguous()
            ):
                raise ValueError("KV ping-pong requires matching persistent pointer tables")
    n = beam_index.numel()
    if n == 0:
        return
    if (
        beam_index.ndim != 1
        or beam_index.dtype not in (torch.int32, torch.int64)
        or beam_index.device != first.device
        or not beam_index.is_contiguous()
        or n > first.shape[1]
        or not first.is_contiguous()
    ):
        raise ValueError("KV ping-pong parent layout or width is incompatible")
    if not 0 <= decode_step < first.shape[3] or not 0 <= slot < first.shape[0]:
        raise ValueError("KV ping-pong step or slot exceeds capacity")
    H, D = first.shape[2], first.shape[4]
    _select_kv_ping_pong_kernel[(layers * n * H,)](
        pool_key_ptrs,
        pool_value_ptrs,
        standby_key_ptrs,
        standby_value_ptrs,
        slot * first.stride(0) * first.element_size(),
        n,
        decode_step + 1,
        beam_index,
        H=H,
        BLOCK_D=triton.next_power_of_2(D),
        D=D,
        stride_pw=first.stride(1),
        stride_ph=first.stride(2),
        stride_ps=first.stride(3),
    )


def select_unshared_kv_fused(
    beam_index: torch.Tensor,  # [n] source beam index for each beam
    slot: int,
    unshared_key_caches: list[torch.Tensor],
    unshared_value_caches: list[torch.Tensor],
    decode_step: int,
    aux_key: torch.Tensor,  # [L, W, H, max_steps, D] per-layer scratch
    aux_value: torch.Tensor,
    pool_key_ptrs: torch.Tensor | None = None,
    pool_value_ptrs: torch.Tensor | None = None,
    *,
    validate_layout: bool = True,
) -> bool:
    """Triton fast path for select_unshared_kv; returns False when unavailable.

    Same semantics as select_unshared_kv's aux-buf path — reorder the first
    ``n`` beams' ``[0:decode_step+1]`` KV through the scratch buffer — as one
    gather + one scatter launch across all layers (per-layer pool base
    addresses passed as an int64 pointer table; uint16 pointer casts make
    the copy bit-exact for any 2-byte dtype). Requires the per-layer aux
    buffers: every layer stages through its own slice, so the single
    all-layer launch cannot cross-clobber staging data.
    """
    if not unshared_key_caches or len(unshared_value_caches) != len(unshared_key_caches):
        return False
    n = beam_index.shape[0]
    first = unshared_key_caches[0]
    if not first.is_cuda or first.element_size() != 2:
        return False
    if aux_key.dim() != 5 or aux_key.shape[0] != len(unshared_key_caches):
        return False  # unexpected aux layout → Python fallback
    _, _, H, _, D = aux_key.shape
    steps = decode_step + 1
    # All layers must share layout (they do: same-shape contiguous pools) and
    # the head_dim axis must be unit-stride in both pool and aux.
    ref_strides = first.stride()
    if ref_strides[4] != 1 or aux_key.stride(4) != 1:
        return False
    if validate_layout and (
        any(p.stride() != ref_strides for p in unshared_key_caches)
        or any(v.stride() != ref_strides for v in unshared_value_caches)
    ):
        return False

    if not beam_index.is_contiguous():
        return False
    device = first.device
    if pool_key_ptrs is None or pool_value_ptrs is None:
        k_ptrs = torch.tensor(
            [p.data_ptr() for p in unshared_key_caches], dtype=torch.int64, device=device
        )
        v_ptrs = torch.tensor(
            [p.data_ptr() for p in unshared_value_caches], dtype=torch.int64, device=device
        )
    else:
        k_ptrs = pool_key_ptrs
        v_ptrs = pool_value_ptrs
        if validate_layout:
            expected = (len(unshared_key_caches),)
            for ptrs in (k_ptrs, v_ptrs):
                if (
                    ptrs.device != device
                    or ptrs.dtype != torch.int64
                    or tuple(ptrs.shape) != expected
                    or not ptrs.is_contiguous()
                ):
                    return False
    # The batch-slot offset advances the byte addresses directly instead of
    # passing a stride_pb argument into the kernels.
    slot_off = slot * ref_strides[0] * first.element_size()
    grid = (len(unshared_key_caches) * n * H,)
    base_args = (
        k_ptrs,
        v_ptrs,
        aux_key.data_ptr(),
        aux_value.data_ptr(),
        slot_off,
        n,
        steps,
    )
    kernel_kwargs = {
        "H": H,
        "BLOCK_D": triton.next_power_of_2(D),
        "D": D,
        "stride_pw": ref_strides[1],
        "stride_ph": ref_strides[2],
        "stride_ps": ref_strides[3],
        "stride_al": aux_key.stride(0),
        "stride_aw": aux_key.stride(1),
        "stride_ah": aux_key.stride(2),
        "stride_as": aux_key.stride(3),
    }
    _select_kv_gather_kernel[grid](*base_args, beam_index, **kernel_kwargs)
    _select_kv_scatter_kernel[grid](*base_args, **kernel_kwargs)
    return True
