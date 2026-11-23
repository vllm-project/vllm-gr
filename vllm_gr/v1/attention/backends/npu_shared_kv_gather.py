# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""One-launch, bit-preserving shared-prefix KV gather on NPU.

The paged cache remains authoritative. Every call/replay reads its current
contents and block table; there is no data cache or cross-session scratch
reuse. Two fresh contiguous outputs are owned by the caller/ACL graph, just
as on the existing Torch path. This changes the copy implementation only.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any

import torch

_ENV = "VLLM_GR_NPU_FUSED_SHARED_KV_GATHER"
_MASK_ENV = "VLLM_GR_NPU_SHARED_KV_LENGTH_MASK"
_COUNTS: Counter[str] = Counter()
_FORMATS: Counter[str] = Counter()
_WARNED: set[str] = set()
_LOGGER = logging.getLogger(__name__)
_KERNEL: Any = None
tl: Any = None


def fused_shared_kv_gather_stats() -> dict[str, Any]:
    """Host diagnostics; replay counts belong to the graph owner's counters."""
    return {
        "requested": os.environ.get(_ENV, "1"),
        "default_enabled": "1",
        "length_mask_requested": os.environ.get(_MASK_ENV, os.environ.get(_ENV, "1")),
        "length_mask_default": "inherit_fused_request",
        "invalid_device_length_policy": "copy_full_capacity_without_masking",
        "host_launch_counts": dict(_COUNTS),
        "observed_input_formats_key_value_table": dict(_FORMATS),
        "counter_scope": "Python invocations only; ACL graph replay does not re-enter Python.",
        "data_cache_retained": False,
    }


def _unsupported_reason(key: torch.Tensor, value: torch.Tensor, table: torch.Tensor) -> str | None:
    if key.device.type != "npu" or value.device != key.device or table.device != key.device:
        return "device"
    if key.ndim != 4 or tuple(value.shape) != tuple(key.shape) or table.ndim != 2:
        return "shape"
    if any(size <= 0 for size in key.shape) or table.shape[0] <= 0 or table.shape[1] <= 0:
        return "empty"
    if key.dtype not in (torch.float16, torch.bfloat16, torch.float32) or value.dtype != key.dtype:
        return "kv_dtype"
    if table.dtype not in (torch.int32, torch.int64):
        return "table_dtype"
    if any(stride <= 0 for tensor in (key, value, table) for stride in tensor.stride()):
        return "stride"
    # dtype view is a metadata-only bit reinterpretation with unchanged item
    # width, but PyTorch requires a contiguous innermost dimension. Ordinary
    # KV pools and their interleaved halves satisfy this; other layouts keep
    # the explicitly observable Torch fallback.
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        return "last_stride"
    _, _, heads, dim = key.shape
    if any(tuple(tensor.stride()[1:]) != (heads * dim, dim, 1) for tensor in (key, value)):
        # Real paged KV blocks are internally dense; only the outer block
        # stride differs for interleaved pools. Other layouts must not enter
        # a scalarized gather that is slower than the original Torch path.
        return "inner_layout"
    # Both NCHW (0) and ND (2) describe unblocked storage in logical dimension
    # order. Tensor data_ptr() already includes a view's storage offset, and
    # the explicit positive strides below describe interleaved/strided views.
    # Do not admit NC1HWC0/FRACTAL_NZ or other internal blocked formats: their
    # physical offsets cannot be reconstructed from these logical strides.
    import torch_npu

    formats = tuple(torch_npu.get_npu_format(tensor) for tensor in (key, value, table))
    _FORMATS[",".join(str(value) for value in formats)] += 1
    if any(value not in (0, 2) for value in formats):
        return "blocked_format"
    return None


def _unsupported_length_reason(
    key: torch.Tensor, table: torch.Tensor, lengths: torch.Tensor | None
) -> str | None:
    """Validate only metadata: capture/replay must not inspect device values."""
    if lengths is None:
        return "prefix_lengths_missing"
    if lengths.device != key.device:
        return "prefix_lengths_device"
    if lengths.dtype != torch.int32:
        return "prefix_lengths_dtype"
    if lengths.ndim != 1 or lengths.shape[0] != table.shape[0]:
        return "prefix_lengths_shape"
    if lengths.stride(0) != 1:
        return "prefix_lengths_stride"
    import torch_npu

    if torch_npu.get_npu_format(lengths) not in (0, 2):
        return "prefix_lengths_format"
    return None


def _kernel():
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    # Disabled/CPU imports do not require Triton. Compile/launch errors
    # propagate: a failed kernel must never be counted as a successful copy.
    import triton
    import triton.language as tl

    @triton.jit
    def copy_kv(
        KEY,
        VALUE,
        TABLE,
        OUT_KEY,
        OUT_VALUE,
        OUT_TABLE,
        LENGTHS,
        COLS: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        KS0: tl.constexpr,
        KS1: tl.constexpr,
        KS2: tl.constexpr,
        KS3: tl.constexpr,
        VS0: tl.constexpr,
        VS1: tl.constexpr,
        VS2: tl.constexpr,
        VS3: tl.constexpr,
        TS0: tl.constexpr,
        TS1: tl.constexpr,
        TILE: tl.constexpr,
        MASK_LENGTHS: tl.constexpr,
        ALIGN_ELEMS: tl.constexpr,
        K_ALIGNED: tl.constexpr,
        V_ALIGNED: tl.constexpr,
    ):
        reference = tl.program_id(0)
        tile = tl.program_id(1)
        row = reference // COLS
        column = reference % COLS
        offset = tile * TILE + tl.arange(0, TILE)
        valid = offset < BLOCK_ELEMS
        if MASK_LENGTHS:
            length = tl.load(LENGTHS + row).to(tl.int64)
            tokens_per_block: tl.constexpr = BLOCK_ELEMS // (HEADS * DIM)
            # Invalid device values cannot be checked through Python during
            # graph replay. Conservatively copy ALL columns in that row,
            # preserving the previous unmasked gather instead of hiding KV.
            valid_length = (length > 0) & (length <= COLS * tokens_per_block)
            active = (column * tokens_per_block < length) | (~valid_length)
            physical = tl.load(TABLE + row * TS0 + column * TS1, mask=active, other=0).to(tl.int64)
            valid = valid & active
        else:
            physical = tl.load(TABLE + row * TS0 + column * TS1).to(tl.int64)
        # Do not leave quotient/remainder reconstruction in the real KV
        # branch. On Ascend that expression scalarized into one GM load per
        # element despite being algebraically contiguous (about600us/layer).
        # Explicit affine offsets and truthful alignment/contiguity hints
        # allow a vector DMA load of each tile from a selected dense block.
        if KS1 == HEADS * DIM and KS2 == DIM and KS3 == 1:
            key_base = physical * KS0
            if K_ALIGNED and KS0 % ALIGN_ELEMS == 0:
                key_base = tl.multiple_of(key_base, ALIGN_ELEMS)
            key_offsets = tl.max_contiguous(key_base + offset, TILE)
        else:
            token = offset // (HEADS * DIM)
            head = (offset // DIM) % HEADS
            dim = offset % DIM
            key_offsets = physical * KS0 + token * KS1 + head * KS2 + dim * KS3
        if VS1 == HEADS * DIM and VS2 == DIM and VS3 == 1:
            value_base = physical * VS0
            if V_ALIGNED and VS0 % ALIGN_ELEMS == 0:
                value_base = tl.multiple_of(value_base, ALIGN_ELEMS)
            value_offsets = tl.max_contiguous(value_base + offset, TILE)
        else:
            token = offset // (HEADS * DIM)
            head = (offset // DIM) % HEADS
            dim = offset % DIM
            value_offsets = physical * VS0 + token * VS1 + head * VS2 + dim * VS3
        # Python passes same-width integer views. Keep every load/store an
        # integer operation: no in-kernel pointer bitcast (unsupported by the
        # current BiSheng allocation-scope pass), no floating conversion.
        key = tl.load(KEY + key_offsets, mask=valid, other=0)
        value = tl.load(VALUE + value_offsets, mask=valid, other=0)
        output_offsets = reference.to(tl.int64) * BLOCK_ELEMS + offset
        tl.store(OUT_KEY + output_offsets, key, mask=valid)
        tl.store(OUT_VALUE + output_offsets, value, mask=valid)
        if tile == 0:
            tl.store(OUT_TABLE + reference, reference)

    _KERNEL = copy_kv
    return _KERNEL


def try_fused_shared_kv_gather(
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    max_valid_cols: int,
    prefix_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return the fused result, or an explicit observable unsupported fallback.

    The shape bound is identical to the existing caller: capture passes zero
    and retains its complete fixed table width; Eager may truncate to its
    validated actual-prefix bound. The device length mask skips
    only unread K/V tail blocks, without changing output geometry or remap.
    Block IDs retain the upstream paged-cache validity contract. No device
    values are read by Python; invalid lengths copy that row's full capacity.
    """
    requested = os.environ.get(_ENV, "1")
    mask_requested = os.environ.get(_MASK_ENV, requested)
    if requested not in ("0", "1"):
        raise ValueError(f"{_ENV} must be 0 or 1, got {requested!r}")
    if mask_requested not in ("0", "1"):
        raise ValueError(f"{_MASK_ENV} must be 0 or 1, got {mask_requested!r}")
    if mask_requested == "1" and requested != "1":
        raise ValueError(f"{_MASK_ENV}=1 requires {_ENV}=1")
    if requested == "0":
        return None
    reason = _unsupported_reason(key, value, block_table)
    if reason is None and mask_requested == "1":
        reason = _unsupported_length_reason(key, block_table, prefix_seqlens)
    if reason is not None:
        _COUNTS[f"fallback:{reason}"] += 1
        if reason not in _WARNED:
            _WARNED.add(reason)
            _LOGGER.warning(
                "Fused NPU shared-KV gather requested but unsupported (%s); using Torch gather",
                reason,
            )
        return None
    columns = block_table.shape[1]
    if 0 < max_valid_cols < columns:
        columns = max_valid_cols
    rows = block_table.shape[0]
    block, heads, dim = key.shape[1:]
    references = rows * columns
    output_shape = (references, block, heads, dim)
    out_key = torch.empty(output_shape, dtype=key.dtype, device=key.device)
    out_value = torch.empty(output_shape, dtype=value.dtype, device=value.device)
    remapped = torch.empty((rows, columns), dtype=block_table.dtype, device=block_table.device)
    tile = 4096
    block_elems = block * heads * dim
    integer_dtype = torch.int16 if key.element_size() == 2 else torch.int32
    key_bits, value_bits = key.view(integer_dtype), value.view(integer_dtype)
    out_key_bits, out_value_bits = out_key.view(integer_dtype), out_value.view(integer_dtype)
    for original, bits in (
        (key, key_bits),
        (value, value_bits),
        (out_key, out_key_bits),
        (out_value, out_value_bits),
    ):
        if (
            bits.data_ptr() != original.data_ptr()
            or tuple(bits.shape) != tuple(original.shape)
            or bits.stride() != original.stride()
        ):
            raise RuntimeError("NPU shared-KV integer dtype view changed storage geometry")
    _kernel()[(references, (block_elems + tile - 1) // tile)](
        key_bits,
        value_bits,
        block_table,
        out_key_bits,
        out_value_bits,
        remapped,
        prefix_seqlens if mask_requested == "1" else block_table,
        columns,
        block_elems,
        heads,
        dim,
        *key.stride(),
        *value.stride(),
        *block_table.stride(),
        tile,
        mask_requested == "1",
        32 // key.element_size(),
        key.data_ptr() % 32 == 0,
        value.data_ptr() % 32 == 0,
    )
    _COUNTS["fused"] += 1
    if mask_requested == "1":
        _COUNTS["fused_masked"] += 1
    return out_key, out_value, remapped


__all__ = ["try_fused_shared_kv_gather", "fused_shared_kv_gather_stats"]
