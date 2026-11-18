# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA-only FA2-compatible short-suffix attention.

Keep Gluon and NVIDIA MMA imports here so the shared dispatch module remains
importable on platforms whose Triton build does not support Gluon.
"""

import math
import struct

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.extra import libdevice
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit(do_not_specialize=["SLOT"])
def _short_suffix_fa2_merge_kernel(
    Q,
    K,
    V,
    Prefix,
    LSE,
    Out,
    Step,
    Active,
    q_strides: gl.constexpr,
    pool_strides: gl.constexpr,
    prefix_strides: gl.constexpr,
    lse_strides: gl.constexpr,
    out_strides: gl.constexpr,
    WIDTH: gl.constexpr,
    HEADS: gl.constexpr,
    KV_HEADS: gl.constexpr,
    DIM: gl.constexpr,
    STEPS: gl.constexpr,
    SLOT,
    SCALE: gl.constexpr,
    SCALE_LOG2: gl.constexpr,
    MASKED: gl.constexpr,
):
    q_layout: gl.constexpr = BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    k_layout: gl.constexpr = BlockedLayout([8, 1], [16, 2], [1, 4], [0, 1])
    mma_layout: gl.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8]
    )

    group: gl.constexpr = HEADS // KV_HEADS
    beams_per_tile: gl.constexpr = 16 // group
    padded_steps: gl.constexpr = triton.next_power_of_2(STEPS)
    block_n: gl.constexpr = max(16, triton.next_power_of_2(beams_per_tile * padded_steps))

    first_beam = gl.program_id(0) * beams_per_tile
    kv_head = gl.program_id(1)
    row = gl.arange(0, 16, layout=SliceLayout(1, q_layout))
    d = gl.arange(0, DIM, layout=SliceLayout(0, q_layout))
    beam = first_beam + row // group
    head = kv_head * group + row % group
    valid = (beam < WIDTH) & (row < beams_per_tile * group)

    query = gl.load(
        Q + beam[:, None] * q_strides[0] + head[:, None] * q_strides[1] + d[None, :] * q_strides[2],
        valid[:, None],
        other=0,
    )

    step = gl.load(Step)
    extent = gl.maximum(step + 1, 1)

    kd = gl.arange(0, DIM, layout=SliceLayout(1, k_layout))
    kn = gl.arange(0, block_n, layout=SliceLayout(0, k_layout))
    k_beam = first_beam + kn // padded_steps
    k_step = kn % padded_steps
    valid_k = (
        (k_beam < WIDTH)
        & (kn < beams_per_tile * padded_steps)
        & (k_step < extent)
        & (k_step < STEPS)
    )

    live = valid_k & (step >= 0)
    if MASKED:
        live = live & gl.load(Active + k_beam, valid_k, other=False)

    base = (
        SLOT.to(gl.int64) * pool_strides[0]
        + k_beam * pool_strides[1]
        + kv_head * pool_strides[2]
        + k_step * pool_strides[3]
    )
    key = gl.load(K + base[None, :] + kd[:, None] * pool_strides[4], live[None, :], other=0)

    # FA2 uses kWidth=2 MMA operands. A different operand layout changes the
    # FP32 addition order even though the BF16/FP16 dot operands are identical.
    query_mma = gl.convert_layout(query, DotOperandLayout(0, mma_layout, 2))
    key_mma = gl.convert_layout(key, DotOperandLayout(1, mma_layout, 2))
    scores = mma_v2(query_mma, key_mma, gl.full((16, block_n), 0, gl.float32, mma_layout))

    mr = gl.arange(0, 16, layout=SliceLayout(1, mma_layout))
    mn = gl.arange(0, block_n, layout=SliceLayout(0, mma_layout))

    # One MMA tile shares a KV head across several beams. Cross-beam QK
    # products must not contribute to either softmax or P@V.
    same_beam = mr[:, None] // group == mn[None, :] // padded_steps
    valid_score = same_beam & (mn[None, :] % padded_steps < extent)
    valid_score = valid_score & (mn[None, :] % padded_steps < STEPS)
    scores = gl.where(valid_score, scores, -float("inf"))

    maximum = gl.max(scores, 1)
    probabilities = gl.exp2(gl.fma(scores, SCALE_LOG2, -(maximum * SCALE_LOG2)[:, None]))
    normalizer = gl.sum(probabilities, 1)
    suffix_lse = gl.fma(maximum, SCALE, libdevice.fast_logf(normalizer))
    rounded_probabilities = probabilities.to(query.dtype)

    vn = gl.arange(0, block_n, layout=SliceLayout(1, q_layout))
    v_beam = first_beam + vn // padded_steps
    v_step = vn % padded_steps
    valid_v = (
        (v_beam < WIDTH)
        & (vn < beams_per_tile * padded_steps)
        & (v_step < extent)
        & (v_step < STEPS)
    )

    v_live = valid_v & (step >= 0)
    if MASKED:
        v_live = v_live & gl.load(Active + v_beam, valid_v, other=False)

    v_base = (
        SLOT.to(gl.int64) * pool_strides[0]
        + v_beam * pool_strides[1]
        + kv_head * pool_strides[2]
        + v_step * pool_strides[3]
    )
    value = gl.load(V + v_base[:, None] + d[None, :] * pool_strides[4], v_live[:, None], other=0)

    probability_mma = gl.convert_layout(rounded_probabilities, DotOperandLayout(0, mma_layout, 2))
    value_mma = gl.convert_layout(value, DotOperandLayout(1, mma_layout, 2))
    suffix_acc = mma_v2(probability_mma, value_mma, gl.full((16, DIM), 0, gl.float32, mma_layout))

    # FA2 rounds the normalized suffix output before the separate CUDA merge.
    suffix = (suffix_acc * (1.0 / normalizer)[:, None]).to(query.dtype)
    suffix = gl.convert_layout(suffix, q_layout).to(gl.float32)
    suffix_lse = gl.convert_layout(suffix_lse, SliceLayout(1, q_layout))

    prefix = gl.load(
        Prefix
        + beam[:, None] * prefix_strides[0]
        + head[:, None] * prefix_strides[1]
        + d[None, :] * prefix_strides[2],
        valid[:, None],
        other=0,
    ).to(gl.float32)
    prefix_lse = gl.load(
        LSE + head * lse_strides[0] + beam * lse_strides[1], valid, other=-float("inf")
    )

    # An absent FA2 prefix uses +inf LSE and must not read stale output data.
    has_prefix = gl.abs(prefix_lse) != float("inf")
    prefix = gl.where(has_prefix[:, None], prefix, 0.0)
    prefix_lse = gl.where(has_prefix, prefix_lse, -float("inf"))

    maximum_lse = gl.maximum(prefix_lse, suffix_lse)
    # CUDA merge uses expf, round-to-nearest division, and this FMA ordering.
    prefix_mass = libdevice.exp(prefix_lse - maximum_lse)
    suffix_mass = libdevice.exp(suffix_lse - maximum_lse)
    total_mass = prefix_mass + suffix_mass
    prefix_weight = libdevice.div_rn(prefix_mass, total_mass)
    suffix_weight = libdevice.div_rn(suffix_mass, total_mass)

    result = gl.fma(prefix, prefix_weight[:, None], suffix * suffix_weight[:, None])
    gl.store(
        Out
        + beam[:, None] * out_strides[0]
        + head[:, None] * out_strides[1]
        + d[None, :] * out_strides[2],
        result,
        valid[:, None],
    )


def _float32(value: float) -> float:
    return float(struct.unpack("f", struct.pack("f", value))[0])


def merge_short_suffix_fa2(
    output: torch.Tensor,
    query: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    pool_key: torch.Tensor,
    pool_value: torch.Tensor,
    slot: int,
    decode_step: torch.Tensor,
    scale: float,
    active_mask: torch.Tensor | None,
) -> None:
    width, heads, dim = query.shape
    kv_heads = pool_key.shape[2]
    scale = _float32(scale)

    _short_suffix_fa2_merge_kernel[(triton.cdiv(width, 16 // (heads // kv_heads)), kv_heads)](
        query,
        pool_key,
        pool_value,
        prefix_output,
        prefix_lse,
        output,
        decode_step,
        active_mask,
        query.stride(),
        pool_key.stride(),
        prefix_output.stride(),
        prefix_lse.stride(),
        output.stride(),
        width,
        heads,
        kv_heads,
        dim,
        pool_key.shape[3],
        slot,
        scale,
        _float32(scale * math.log2(math.e)),
        active_mask is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
