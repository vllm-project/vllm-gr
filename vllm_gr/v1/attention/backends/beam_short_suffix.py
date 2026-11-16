# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Short incremental suffix attention, fused with the prefix-state merge.

Read the persistent KV pool directly: no packed suffix, intermediate suffix
output/LSE, or FA2 GQA LSE conversion. The decode step and active mask remain
on device so a single CUDA graph can replay different steps and sessions.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["SLOT"])
def _short_suffix_merge_kernel(
    Q,
    K,
    V,
    Prefix,
    LSE,
    Out,
    Step,
    Active,
    q_strides: tl.constexpr,
    pool_strides: tl.constexpr,
    prefix_strides: tl.constexpr,
    lse_strides: tl.constexpr,
    out_strides: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    STEPS: tl.constexpr,
    SLOT,
    SCALE: tl.constexpr,
    MASKED: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    StandbyK=None,
    StandbyV=None,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    beam, head = row // HEADS, row % HEADS
    kv_head = head // (HEADS // KV_HEADS)
    d = tl.arange(0, BLOCK_D)
    valid = beam < WIDTH

    q = tl.load(
        Q + beam[:, None] * q_strides[0] + head[:, None] * q_strides[1] + d[None, :] * q_strides[2],
        valid[:, None] & (d[None, :] < DIM),
        other=0,
    ).to(tl.float32)

    prefix = tl.load(
        Prefix
        + beam[:, None] * prefix_strides[0]
        + head[:, None] * prefix_strides[1]
        + d[None, :] * prefix_strides[2],
        valid[:, None] & (d[None, :] < DIM),
        other=0,
    ).to(tl.float32)

    maximum = tl.load(
        LSE + head * lse_strides[0] + beam * lse_strides[1], valid, other=-float("inf")
    ).to(tl.float32)

    # FA2 represents empty prefix sequences with +inf LSE. Match the
    # generic merge by treating these states as absent, not as infinite mass.
    maximum = tl.where(maximum == float("inf"), -float("inf"), maximum)
    has_prefix = maximum != -float("inf")
    normalizer = has_prefix.to(tl.float32)
    accumulator = tl.where(has_prefix[:, None], prefix, 0.0)

    step = tl.load(Step)
    if StandbyK is not None:
        K = tl.where((step & 1) != 0, StandbyK, K)
        V = tl.where((step & 1) != 0, StandbyV, V)
    live = valid & (step >= 0)
    if MASKED:
        live = live & tl.load(Active + beam, valid, other=False)

    # Match the existing masked packed-copy path: inactive beams have zero
    # suffix K/V, not stale pool contents. They still have the same extent.
    extent = tl.maximum(step + 1, 1)

    # Widen before multiplying: a valid slot offset can exceed signed int32.
    base = SLOT.to(tl.int64) * pool_strides[0] + beam * pool_strides[1] + kv_head * pool_strides[2]

    # Keep the existing tiny-pool unroll. Larger pools loop only over the
    # current device-side extent, avoiding code growth and work for unused
    # capacity while preserving CUDA graph replay at different decode steps.
    loop_end = STEPS
    if STEPS > 3:
        loop_end = tl.minimum(extent, STEPS)

    for s in tl.range(0, loop_end, loop_unroll_factor=STEPS if STEPS <= 3 else 1):
        offset = base[:, None] + s * pool_strides[3] + d[None, :] * pool_strides[4]
        load_mask = live[:, None] & (s < extent) & (d[None, :] < DIM)
        key = tl.load(K + offset, load_mask, other=0).to(tl.float32)
        value = tl.load(V + offset, load_mask, other=0).to(tl.float32)

        score = tl.sum(q * key, axis=1) * SCALE
        score = tl.where(valid & (s < extent), score, -float("inf"))

        new_max = tl.maximum(maximum, score)
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp(maximum - safe_max)
        beta = tl.exp(score - safe_max)

        accumulator = accumulator * alpha[:, None] + value * beta[:, None]
        normalizer = normalizer * alpha + beta
        maximum = new_max

    result = accumulator / tl.where(normalizer > 0, normalizer, 1.0)[:, None]
    tl.store(
        Out
        + beam[:, None] * out_strides[0]
        + head[:, None] * out_strides[1]
        + d[None, :] * out_strides[2],
        result,
        valid[:, None] & (d[None, :] < DIM),
    )


def merge_short_suffix_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    pool_key: torch.Tensor,
    pool_value: torch.Tensor,
    slot: int,
    decode_step: torch.Tensor,
    scale: float,
    active_mask: torch.Tensor | None = None,
    *,
    standby_key: torch.Tensor | None = None,
    standby_value: torch.Tensor | None = None,
) -> None:
    """Merge one query/beam with the valid pool suffix and a prefix attention state.

    Consumes the backend-independent AttentionState contract: prefix output is
    [beams, query_heads, head_dim]; LSE is FP32 natural-log [query_heads, beams]
    in canonical head order. Either infinite LSE sentinel denotes no prefix.
    The prefix may come from FA2, FA3 or FA4; the tiny-pool suffix arithmetic
    remains FA2-compatible independently of the prefix producer.
    The caller dispatches only plain, unwindowed attention without softcap,
    ALiBi or sinks. Key/value pools have identical strides and independent
    beam histories. Tiny CUDA FP16/BF16 pools preserve FA2's intermediate rounding
    stages. The general kernel, including the independent FP32 validation path,
    accumulates in FP32 without TF32 or reduced-precision dot products.
    """
    width, heads, dim = query.shape
    steps = pool_key.shape[3]

    assert steps >= 1
    assert pool_key.shape == pool_value.shape and pool_key.stride() == pool_value.stride()
    assert heads % pool_key.shape[2] == 0
    assert (standby_key is None) == (standby_value is None)
    if standby_key is not None:
        assert standby_value is not None
        assert standby_key.shape == pool_key.shape and standby_key.stride() == pool_key.stride()
        assert (
            standby_value.shape == pool_value.shape
            and standby_value.stride() == pool_value.stride()
        )
        assert standby_key.dtype == pool_key.dtype and standby_key.device == pool_key.device
        assert standby_value.dtype == pool_value.dtype and standby_value.device == pool_value.device

    if (
        query.is_cuda
        and steps <= 3
        and dim in (64, 128, 256)
        and heads // pool_key.shape[2] <= 16
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.dtype == prefix_output.dtype == output.dtype == pool_key.dtype == pool_value.dtype
    ):
        # Gluon depends on NVIDIA-specific Triton APIs unavailable on NPU.
        from vllm_gr.v1.attention.backends.beam_short_suffix_fa2 import (
            merge_short_suffix_fa2,
        )

        merge_short_suffix_fa2(
            output,
            query,
            prefix_output,
            prefix_lse,
            pool_key,
            pool_value,
            slot,
            decode_step,
            scale,
            active_mask,
            **(
                {"standby_key": standby_key, "standby_value": standby_value}
                if standby_key is not None
                else {}
            ),
        )
        return

    _short_suffix_merge_kernel[(triton.cdiv(width * heads, 4),)](
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
        pool_key.shape[2],
        dim,
        steps,
        slot,
        scale,
        active_mask is not None,
        4,
        triton.next_power_of_2(dim),
        StandbyK=standby_key,
        StandbyV=standby_value,
        num_warps=4,
    )
