# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Triton ordering keys for descending scores and ascending indices."""

import triton
import triton.language as tl

# mypy: disable-error-code="misc,no-untyped-def"


@triton.jit
def pack_score_index(score, index, valid):
    min_i32: tl.constexpr = -2147483648
    score = tl.where(valid & (score == score), score, -float("inf"))
    # Collapse -0.0 onto +0.0 so the packed order is a total order.
    score = tl.where(score == 0.0, 0.0, score)
    score_bits = score.to(tl.int32, bitcast=True)
    sign = score_bits >> 31
    sort_key = tl.where(sign == 0, score_bits ^ -1, score_bits ^ min_i32)
    high = sort_key.to(tl.int64) & 0x00000000FFFFFFFF
    low = tl.where(valid, index.to(tl.int64), 0x00000000FFFFFFFF)
    return (high << 32) | low


@triton.jit
def unpack_score(packed):
    min_i32: tl.constexpr = -2147483648
    sort_key = ((packed >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    sign = sort_key >> 31
    score_bits = tl.where(sign < 0, sort_key ^ -1, sort_key ^ min_i32)
    return score_bits.to(tl.float32, bitcast=True)
