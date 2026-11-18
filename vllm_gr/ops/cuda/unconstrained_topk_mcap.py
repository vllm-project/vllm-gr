# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Bounded-retention full-vocabulary Beam Top-K.

The legacy path keeps ``top_k`` candidates in every vocabulary tile (173 tiles
x 256 = 44,288 candidates per row) and merges them back down to ``top_k``.  This
module keeps ``keep`` candidates per tile plus the ``keep + 1``-th best key (the
"marginal"): if every tile's marginal is worse than the merged ``top_k``-th key,
no dropped candidate can enter the global Top-K.  Packed keys are a strict total
order, so the check is exact, and a failing row is rebuilt by re-scanning only
its flagged tiles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# Triton JIT functions intentionally use its untyped DSL values.
# mypy: disable-error-code="misc,no-untyped-def"

LOCAL_TILE_SIZE = 1024
# Per-tile retention cap.  A tile holds 1024 logits and contributes ~top_k /
# tiles candidates to the global Top-K, so a small cap keeps the local
# write-back and the merge traffic proportional to the read instead of to
# ``tiles * top_k``.  Lowering it trades local ``tl.topk`` work for a higher
# chance of the (exact) repair pass, so it stays tunable per deployment.
LOCAL_KEEP_MAX = int(os.environ.get("VLLM_GR_TOPK_KEEP", "8"))
if not 1 <= LOCAL_KEEP_MAX <= 256:
    raise ValueError(f"VLLM_GR_TOPK_KEEP must be in [1, 256], got {LOCAL_KEEP_MAX}")
# Width of one chunk in the first merge stage.  The stage is a plain
# "sort this chunk, keep its best top_k" pass, so the chunk count (not the tile
# count) bounds the second stage.
MERGE_CHUNK = 1024

_MAX_I64 = tl.constexpr(0x7FFFFFFFFFFFFFFF)


@dataclass(frozen=True, slots=True)
class MCarryTopKScratch:
    """Persistent storage for the bounded-retention Top-K pipeline."""

    local_keys: torch.Tensor
    local_marginal: torch.Tensor
    partial_max: torch.Tensor
    partial_sum: torch.Tensor
    merge_keys: torch.Tensor
    final_keys: torch.Tensor
    repair_flag: torch.Tensor
    repair_running: torch.Tensor
    tile_flag: torch.Tensor
    vocab_size: int
    top_k: int
    keep: int


@triton.jit
def _pack_score_index(score, index, valid):
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
def _unpack_score(packed):
    min_i32: tl.constexpr = -2147483648
    sort_key = ((packed >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    sign = sort_key >> 31
    score_bits = tl.where(sign < 0, sort_key ^ -1, sort_key ^ min_i32)
    return score_bits.to(tl.float32, bitcast=True)


@triton.jit
def _local_keys_kernel(
    logits,
    temperatures,
    local_keys,
    local_marginal,
    partial_max,
    partial_sum,
    vocab_size,
    logits_stride,
    temperature_count,
    keep_stride,
    partial_stride,
    keep_width,
    tile_flag,
    tile_flag_stride,
    tile_offset,
    slice_size,
    BLOCK: tl.constexpr,
    KEEP_BLOCK: tl.constexpr,
    NEED_MARGINAL: tl.constexpr,
    I32_SELECT: tl.constexpr,
    LANE_BITS: tl.constexpr,
):
    """Logsumexp statistics plus the retained keys of one vocabulary tile."""

    row = tl.program_id(0)
    tile = tile_offset + tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    flat = tile * BLOCK + offsets
    # ``logits`` may be one vocabulary slice of the full row, so addressing is
    # relative to the slice while the packed key keeps the global index.
    local = flat - tile_offset * BLOCK
    valid = (flat < vocab_size) & (local < slice_size)
    values = tl.load(logits + row * logits_stride + local, mask=valid, other=-float("inf"))
    values = values.to(tl.float32)
    temperature_index = tl.where(temperature_count == 1, 0, row)
    temperature = tl.load(temperatures + temperature_index)
    temperature = tl.where(temperature == 0.0, 1.0, temperature)
    values = values / temperature
    # Non-finite logits cannot produce valid Beam candidates or probability mass.
    values = tl.where((values > -float("inf")) & (values < float("inf")), values, -float("inf"))
    # The normalization statistics stay beside the candidates so the final stage
    # never rescans the full vocabulary.
    tile_max = tl.max(values, axis=0)
    tile_sum = tl.sum(tl.exp(values - tl.where(tile_max == -float("inf"), 0.0, tile_max)), axis=0)
    tl.store(partial_max + row * partial_stride + tile, tile_max)
    tl.store(partial_sum + row * partial_stride + tile, tile_sum)

    packed = _pack_score_index(values, flat, valid)
    retained = tl.arange(0, KEEP_BLOCK)
    if KEEP_BLOCK == 1:
        # Smallest packed key wins: highest score, then lowest token index.
        head = tl.min(packed, axis=0)
        tl.store(local_keys + row * keep_stride + tile * keep_width, head)
        marginal = tl.full((1,), _MAX_I64, tl.int64)
        tl.store(tile_flag + row * tile_flag_stride + tile, 0)
    elif I32_SELECT:
        sort_key = ((packed >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
        # 21 score bits above 10 lane bits keeps the surrogate non-negative.
        # The packed key is a *signed* int64: a sort key >= 2**31 wraps into the
        # negative half, which is what places every positive score ahead of
        # every negative one.  XOR the prefix sign bit to reproduce that order.
        unsigned_key = sort_key.to(tl.int64) & 0x00000000FFFFFFFF
        prefix = (unsigned_key >> (LANE_BITS + 1)) ^ 0x100000
        surrogate = ((prefix << LANE_BITS) | offsets).to(tl.int32)
        ranked = ~tl.topk(~surrogate, KEEP_BLOCK)
        lanes = ranked & ((1 << LANE_BITS) - 1)
        selected_flat = tile * BLOCK + lanes
        selected_local = selected_flat - tile_offset * BLOCK
        selected_valid = (
            (selected_flat < vocab_size) & (selected_local < slice_size) & (selected_local >= 0)
        )
        selected_values = tl.load(
            logits + row * logits_stride + selected_local,
            mask=selected_valid,
            other=-float("inf"),
        ).to(tl.float32)
        selected_values = selected_values / temperature
        selected_values = tl.where(
            (selected_values > -float("inf")) & (selected_values < float("inf")),
            selected_values,
            -float("inf"),
        )
        exact = _pack_score_index(selected_values, selected_flat, selected_valid)
        tl.store(
            local_keys + row * keep_stride + tile * keep_width + retained,
            exact,
            mask=retained < keep_width,
        )
        marginal = tl.min(tl.where(retained == keep_width, exact, _MAX_I64), axis=0)
        max_i32: tl.constexpr = 2147483647
        last_in = tl.min(tl.where(retained == keep_width - 1, ranked >> LANE_BITS, max_i32), axis=0)
        first_out = tl.min(tl.where(retained == keep_width, ranked >> LANE_BITS, max_i32), axis=0)
        # Equal truncated prefixes across the cutoff mean the surrogate cannot
        # prove an order: let the exact repair pass own this tile.
        tl.store(
            tile_flag + row * tile_flag_stride + tile,
            (last_in == first_out).to(tl.int8),
        )
    else:
        # Complementing reverses the signed order, so a descending Top-K of the
        # complement is an ascending Top-K of the original packed keys.
        ranked = ~tl.topk(~packed, KEEP_BLOCK)
        tl.store(
            local_keys + row * keep_stride + tile * keep_width + retained,
            ranked,
            mask=retained < keep_width,
        )
        marginal = tl.min(tl.where(retained == keep_width, ranked, _MAX_I64), axis=0)
        tl.store(tile_flag + row * tile_flag_stride + tile, 0)
    if NEED_MARGINAL:
        # Slot ``keep_width`` holds the (keep + 1)-th best key of this tile.
        tl.store(local_marginal + row * partial_stride + tile, marginal)
    else:
        tl.store(local_marginal + row * partial_stride + tile, _MAX_I64)


@triton.jit
def _merge_chunk_kernel(
    input_keys,
    output_keys,
    candidate_count,
    in_stride,
    out_stride,
    BLOCK: tl.constexpr,
    SELECT_BLOCK: tl.constexpr,
    STORE_BLOCK: tl.constexpr,
):
    """Keep the best ``STORE_BLOCK`` keys of one chunk of candidates."""

    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    source = chunk * BLOCK + offsets
    valid = source < candidate_count
    key = tl.load(input_keys + row * in_stride + source, mask=valid, other=_MAX_I64)
    if STORE_BLOCK == 1:
        tl.store(output_keys + row * out_stride + chunk, tl.min(key, axis=0))
    else:
        ranked = ~tl.topk(~key, SELECT_BLOCK)
        selected = tl.arange(0, SELECT_BLOCK)
        # ``SELECT_BLOCK`` is only the width of the bitonic network (the next
        # power of two), so the slot stride and the store mask follow
        # ``STORE_BLOCK``: a non-power-of-two ``top_k`` would otherwise write
        # past its slot into the next chunk, the next row or the allocation.
        tl.store(
            output_keys + row * out_stride + chunk * STORE_BLOCK + selected,
            ranked,
            mask=selected < STORE_BLOCK,
        )


@triton.jit
def _finalize_kernel(
    keys,
    partial_max,
    partial_sum,
    local_marginal,
    output_ids,
    output_logprobs,
    output_valid,
    output_counts,
    repair_flag,
    tile_flag,
    tile_count,
    # One ``tile_count`` array serves both the statistics and the marginals.
    partial_stride,
    tile_flag_stride,
    key_stride,
    output_stride,
    top_k,
    NEED_VERIFY: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile_offsets = tl.arange(0, SCAN_BLOCK)
    tile_valid = tile_offsets < tile_count
    tile_maxes = tl.load(
        partial_max + row * partial_stride + tile_offsets,
        mask=tile_valid,
        other=-float("inf"),
    )
    tile_sums = tl.load(
        partial_sum + row * partial_stride + tile_offsets,
        mask=tile_valid,
        other=0.0,
    )
    global_max = tl.max(tile_maxes, axis=0)
    has_mass = global_max != -float("inf")
    safe_max = tl.where(has_mass, global_max, 0.0)
    denominator = safe_max + tl.log(tl.sum(tile_sums * tl.exp(tile_maxes - safe_max), axis=0))

    ranks = tl.arange(0, OUTPUT_BLOCK)
    selected = ranks < top_k
    key = tl.load(keys + row * key_stride + ranks, mask=selected, other=_MAX_I64)
    tokens = (key & 0x00000000FFFFFFFF).to(tl.int32)
    scores = _unpack_score(key)
    valid = selected & has_mass & (tokens >= 0) & (scores > -float("inf")) & (scores < float("inf"))
    output_offset = row * output_stride + ranks
    tl.store(output_ids + output_offset, tokens, mask=selected)
    tl.store(
        output_logprobs + output_offset,
        tl.where(valid, scores - denominator, -float("inf")),
        mask=selected,
    )
    tl.store(output_valid + output_offset, valid, mask=selected)
    tl.store(output_counts + row, tl.sum(valid.to(tl.int32), axis=0))

    if NEED_VERIFY:
        marginals = tl.load(
            local_marginal + row * partial_stride + tile_offsets,
            mask=tile_valid,
            other=_MAX_I64,
        )
        theta = tl.min(tl.where(ranks == top_k - 1, key, _MAX_I64), axis=0)
        # A tile is unsafe only if it could contribute more than ``keep`` keys
        # to the global Top-K.  Keys are unique, so the comparison is exact.
        unsafe = tile_valid & (marginals < theta)
        previous = tl.load(
            tile_flag + row * tile_flag_stride + tile_offsets,
            mask=tile_valid,
            other=0,
        )
        flags = unsafe | (previous != 0)
        tl.store(
            tile_flag + row * tile_flag_stride + tile_offsets,
            flags.to(tl.int8),
            mask=tile_valid,
        )
        tl.store(repair_flag + row, (tl.sum(flags.to(tl.int32), axis=0) > 0).to(tl.int32))


@triton.jit
def _repair_row_kernel(
    logits,
    temperatures,
    repair_flag,
    final_keys,
    tile_flag,
    partial_max,
    partial_sum,
    output_ids,
    output_logprobs,
    output_valid,
    output_counts,
    vocab_size,
    logits_stride,
    temperature_count,
    key_stride,
    tile_flag_stride,
    tile_count,
    partial_stride,
    output_stride,
    top_k,
    BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr,
):
    """Recompute the exact Top-K for flagged rows, folding tile by tile."""

    row = tl.program_id(0)
    if tl.load(repair_flag + row) != 0:
        lanes = tl.arange(0, TOPK_BLOCK)
        # The retained Top-K already covers every non-flagged tile, and only a
        # flagged tile can hold a candidate that the marginal check rejected.
        running = tl.load(
            final_keys + row * key_stride + lanes,
            mask=lanes < top_k,
            other=_MAX_I64,
        )
        # Retained entries that belong to a flagged tile are superseded by that
        # tile's full Top-K below; dropping them first keeps the union disjoint
        # (packed keys are unique, so duplicates would steal output slots).
        key_tile = ((running & 0x00000000FFFFFFFF) // BLOCK).to(tl.int32)
        key_tile = tl.where((key_tile >= 0) & (key_tile < tile_count), key_tile, 0)
        superseded = (
            tl.load(
                tile_flag + row * tile_flag_stride + key_tile,
                mask=lanes < top_k,
                other=0,
            )
            != 0
        )
        running = tl.where(superseded, _MAX_I64, running)
        temperature_index = tl.where(temperature_count == 1, 0, row)
        temperature = tl.load(temperatures + temperature_index)
        temperature = tl.where(temperature == 0.0, 1.0, temperature)
        for tile in range(0, tile_count):
            if tl.load(tile_flag + row * tile_flag_stride + tile) != 0:
                offsets = tl.arange(0, BLOCK)
                flat = tile * BLOCK + offsets
                valid = flat < vocab_size
                values = tl.load(
                    logits + row * logits_stride + flat, mask=valid, other=-float("inf")
                ).to(tl.float32)
                values = values / temperature
                values = tl.where(
                    (values > -float("inf")) & (values < float("inf")), values, -float("inf")
                )
                packed = _pack_score_index(values, flat, valid)
                if TOPK_BLOCK == 1:
                    # The i32 cutoff can require exact repair even for K=1.
                    # Avoid Triton's scalar reshape in tl.topk(..., 1).
                    tile_top = tl.min(packed, axis=0, keep_dims=True)
                    running = tl.minimum(running, tile_top)
                else:
                    tile_top = ~tl.topk(~packed, TOPK_BLOCK)
                    both = tl.cat(running, tile_top, can_reorder=True)
                    running = ~tl.topk(~both, TOPK_BLOCK)
        key = running

        tile_offsets = tl.arange(0, SCAN_BLOCK)
        tile_valid = tile_offsets < tile_count
        tile_maxes = tl.load(
            partial_max + row * partial_stride + tile_offsets,
            mask=tile_valid,
            other=-float("inf"),
        )
        tile_sums = tl.load(
            partial_sum + row * partial_stride + tile_offsets,
            mask=tile_valid,
            other=0.0,
        )
        global_max = tl.max(tile_maxes, axis=0)
        has_mass = global_max != -float("inf")
        safe_max = tl.where(has_mass, global_max, 0.0)
        denominator = safe_max + tl.log(tl.sum(tile_sums * tl.exp(tile_maxes - safe_max), axis=0))

        ranks = tl.arange(0, OUTPUT_BLOCK)
        selected = ranks < top_k
        tokens = (key & 0x00000000FFFFFFFF).to(tl.int32)
        scores = _unpack_score(key)
        valid_out = (
            selected & has_mass & (tokens >= 0) & (scores > -float("inf")) & (scores < float("inf"))
        )
        output_offset = row * output_stride + ranks
        tl.store(output_ids + output_offset, tokens, mask=selected)
        tl.store(
            output_logprobs + output_offset,
            tl.where(valid_out, scores - denominator, -float("inf")),
            mask=selected,
        )
        tl.store(output_valid + output_offset, valid_out, mask=selected)
        tl.store(output_counts + row, tl.sum(valid_out.to(tl.int32), axis=0))


@triton.jit
def _repair_fold_kernel(
    logits,
    temperatures,
    repair_flag,
    final_keys,
    running,
    tile_flag,
    vocab_size,
    logits_stride,
    temperature_count,
    key_stride,
    running_stride,
    tile_flag_stride,
    tile_offset,
    slice_size,
    slice_tiles,
    tile_count,
    top_k,
    BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    FIRST: tl.constexpr,
):
    """Refold the flagged tiles of one vocabulary slice into the running set."""

    row = tl.program_id(0)
    if tl.load(repair_flag + row) != 0:
        lanes = tl.arange(0, TOPK_BLOCK)
        if FIRST:
            # Start from the merged Top-K with every entry owned by a flagged
            # tile removed; those tiles are refolded from exact keys below.
            key = tl.load(final_keys + row * key_stride + lanes, mask=lanes < top_k, other=_MAX_I64)
            key_tile = ((key & 0x00000000FFFFFFFF) // BLOCK).to(tl.int32)
            key_tile = tl.where((key_tile >= 0) & (key_tile < tile_count), key_tile, 0)
            superseded = (
                tl.load(
                    tile_flag + row * tile_flag_stride + key_tile,
                    mask=lanes < top_k,
                    other=0,
                )
                != 0
            )
            current = tl.where(superseded, _MAX_I64, key)
        else:
            current = tl.load(
                running + row * running_stride + lanes, mask=lanes < top_k, other=_MAX_I64
            )
        temperature_index = tl.where(temperature_count == 1, 0, row)
        temperature = tl.load(temperatures + temperature_index)
        temperature = tl.where(temperature == 0.0, 1.0, temperature)
        for step in range(0, slice_tiles):
            tile = tile_offset + step
            if tl.load(tile_flag + row * tile_flag_stride + tile) != 0:
                offsets = tl.arange(0, BLOCK)
                global_flat = tile * BLOCK + offsets
                local = global_flat - tile_offset * BLOCK
                valid = (global_flat < vocab_size) & (local < slice_size)
                values = tl.load(
                    logits + row * logits_stride + local,
                    mask=valid,
                    other=-float("inf"),
                ).to(tl.float32)
                values = values / temperature
                values = tl.where(
                    (values > -float("inf")) & (values < float("inf")),
                    values,
                    -float("inf"),
                )
                packed = _pack_score_index(values, global_flat, valid)
                tile_top = ~tl.topk(~packed, TOPK_BLOCK)
                both = tl.cat(current, tile_top, can_reorder=True)
                current = ~tl.topk(~both, TOPK_BLOCK)
        tl.store(running + row * running_stride + lanes, current, mask=lanes < top_k)


@triton.jit
def _stream_slice_kernel(
    logits,
    temperatures,
    local_keys,
    local_marginal,
    tile_flag,
    running,
    repair_flag,
    logits_stride,
    temperature_count,
    candidate_stride,
    tile_stride,
    running_stride,
    vocab_size,
    tile_offset,
    slice_size,
    top_k,
    KEEP: tl.constexpr,
    SLICE_TILES: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Merge and repair a slice before its logits buffer is reused."""

    row = tl.program_id(0)
    lanes = tl.arange(0, TOPK_BLOCK)
    offsets = tl.arange(0, CANDIDATE_BLOCK)
    if FIRST:
        previous = tl.full((CANDIDATE_BLOCK,), _MAX_I64, tl.int64)
    else:
        previous = tl.load(
            running + row * running_stride + offsets, mask=offsets < top_k, other=_MAX_I64
        )
    retained = tl.load(
        local_keys + row * candidate_stride + tile_offset * KEEP + offsets,
        mask=offsets < SLICE_TILES * KEEP,
        other=_MAX_I64,
    )
    combined = tl.cat(previous, retained, can_reorder=True)
    if TOPK_BLOCK == 1:
        current = tl.min(combined, axis=0, keep_dims=True)
    else:
        current = ~tl.topk(~combined, TOPK_BLOCK)
    cutoff = tl.max(tl.where(lanes < top_k, current, -9223372036854775808), axis=0)

    # The retained keys are a subset of the true candidates, so their cutoff
    # is a lower bound on the final score cutoff. Any tile whose marginal is
    # worse can be discarded permanently, including in later slices.
    tiles = tl.arange(0, TILE_BLOCK)
    marginal = tl.load(
        local_marginal + row * tile_stride + tile_offset + tiles,
        mask=tiles < SLICE_TILES,
        other=_MAX_I64,
    )
    ambiguous = tl.load(
        tile_flag + row * tile_stride + tile_offset + tiles,
        mask=tiles < SLICE_TILES,
        other=0,
    )
    flagged = (tiles < SLICE_TILES) & ((marginal < cutoff) | (ambiguous != 0))
    tl.store(
        tile_flag + row * tile_stride + tile_offset + tiles,
        flagged.to(tl.int8),
        mask=tiles < SLICE_TILES,
    )
    key_tile = ((current & 0x00000000FFFFFFFF) // BLOCK).to(tl.int32) - tile_offset
    in_slice = (key_tile >= 0) & (key_tile < SLICE_TILES)
    superseded = in_slice & tl.gather(
        flagged, tl.minimum(tl.maximum(key_tile, 0), TILE_BLOCK - 1), 0
    )
    current = tl.where(superseded | (lanes >= top_k), _MAX_I64, current)

    temperature_index = tl.where(temperature_count == 1, 0, row)
    temperature = tl.load(temperatures + temperature_index)
    temperature = tl.where(temperature == 0.0, 1.0, temperature)
    # Only this row's flagged tiles are read again, while this slice's GEMM
    # result is still available. No host decision or projection rerun is needed.
    tl.debug_barrier()
    for step in range(0, SLICE_TILES):
        tile = tile_offset + step
        if tl.load(tile_flag + row * tile_stride + tile) != 0:
            positions = tl.arange(0, BLOCK)
            flat = tile * BLOCK + positions
            local = step * BLOCK + positions
            valid = (flat < vocab_size) & (local < slice_size)
            values = (
                tl.load(logits + row * logits_stride + local, mask=valid, other=-float("inf")).to(
                    tl.float32
                )
                / temperature
            )
            values = tl.where(
                (values > -float("inf")) & (values < float("inf")), values, -float("inf")
            )
            packed = _pack_score_index(values, flat, valid)
            if TOPK_BLOCK == 1:
                tile_top = tl.min(packed, axis=0, keep_dims=True)
            else:
                tile_top = ~tl.topk(~packed, TOPK_BLOCK)
            repaired = tl.cat(current, tile_top, can_reorder=True)
            if TOPK_BLOCK == 1:
                current = tl.min(repaired, axis=0, keep_dims=True)
            else:
                current = ~tl.topk(~repaired, TOPK_BLOCK)
    tl.store(running + row * running_stride + lanes, current, mask=lanes < top_k)
    # Every streaming row is complete, even when none of its tiles needed repair.
    tl.store(repair_flag + row, 1)


@triton.jit
def _repair_emit_kernel(
    running,
    repair_flag,
    partial_max,
    partial_sum,
    output_ids,
    output_logprobs,
    output_valid,
    output_counts,
    running_stride,
    partial_stride,
    tile_count,
    output_stride,
    top_k,
    SCAN_BLOCK: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr,
):
    """Emit the rows rebuilt by the vocabulary-sliced repair pass."""

    row = tl.program_id(0)
    if tl.load(repair_flag + row) != 0:
        ranks = tl.arange(0, OUTPUT_BLOCK)
        selected = ranks < top_k
        key = tl.load(running + row * running_stride + ranks, mask=selected, other=_MAX_I64)
        tile_offsets = tl.arange(0, SCAN_BLOCK)
        tile_valid = tile_offsets < tile_count
        tile_maxes = tl.load(
            partial_max + row * partial_stride + tile_offsets,
            mask=tile_valid,
            other=-float("inf"),
        )
        tile_sums = tl.load(
            partial_sum + row * partial_stride + tile_offsets,
            mask=tile_valid,
            other=0.0,
        )
        global_max = tl.max(tile_maxes, axis=0)
        has_mass = global_max != -float("inf")
        safe_max = tl.where(has_mass, global_max, 0.0)
        denominator = safe_max + tl.log(tl.sum(tile_sums * tl.exp(tile_maxes - safe_max), axis=0))
        tokens = (key & 0x00000000FFFFFFFF).to(tl.int32)
        scores = _unpack_score(key)
        valid = (
            selected & has_mass & (tokens >= 0) & (scores > -float("inf")) & (scores < float("inf"))
        )
        output_offset = row * output_stride + ranks
        tl.store(output_ids + output_offset, tokens, mask=selected)
        tl.store(
            output_logprobs + output_offset,
            tl.where(valid, scores - denominator, -float("inf")),
            mask=selected,
        )
        tl.store(output_valid + output_offset, valid, mask=selected)
        tl.store(output_counts + row, tl.sum(valid.to(tl.int32), axis=0))


def mcap_topk_tiling(vocab_size: int, top_k: int) -> tuple[int, int, int, int]:
    """Return ``(tiles, keep, candidates, chunk_count)``; one shared definition."""

    if vocab_size <= 0 or top_k <= 0:
        raise ValueError("vocab_size and top_k must be positive")
    tiles = (vocab_size + LOCAL_TILE_SIZE - 1) // LOCAL_TILE_SIZE
    # A row must be able to retain at least ``top_k`` candidates in total, so a
    # vocabulary with only a few tiles widens the per-tile cap instead of
    # silently dropping winners.
    keep = min(top_k, max(LOCAL_KEEP_MAX, -(-top_k // tiles)))
    candidates = tiles * keep
    chunk_count = (candidates + MERGE_CHUNK - 1) // MERGE_CHUNK
    return tiles, keep, candidates, chunk_count


def merge_chunk_width(top_k: int) -> int:
    """Keys one merge chunk keeps; also the stride between chunks."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    # A chunk holds at most ``MERGE_CHUNK`` candidates, so keeping more than that
    # per chunk is impossible and would leave uninitialised slots behind.
    return min(top_k, MERGE_CHUNK)


def allocate_mcap_topk_scratch(
    batch_size: int,
    vocab_size: int,
    top_k: int,
    device: torch.device,
) -> MCarryTopKScratch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tiles, keep, candidates, chunk_count = mcap_topk_tiling(vocab_size, top_k)
    return MCarryTopKScratch(
        local_keys=torch.empty((batch_size, candidates), dtype=torch.int64, device=device),
        local_marginal=torch.empty((batch_size, tiles), dtype=torch.int64, device=device),
        partial_max=torch.empty((batch_size, tiles), dtype=torch.float32, device=device),
        partial_sum=torch.empty((batch_size, tiles), dtype=torch.float32, device=device),
        merge_keys=torch.empty(
            (batch_size, chunk_count * merge_chunk_width(top_k)),
            dtype=torch.int64,
            device=device,
        ),
        final_keys=torch.empty((batch_size, top_k), dtype=torch.int64, device=device),
        repair_flag=torch.zeros((batch_size,), dtype=torch.int32, device=device),
        repair_running=torch.empty(
            (batch_size, 2 * triton.next_power_of_2(top_k)), dtype=torch.int64, device=device
        ),
        tile_flag=torch.zeros((batch_size, tiles), dtype=torch.int8, device=device),
        vocab_size=vocab_size,
        top_k=top_k,
        keep=keep,
    )


def unconstrained_topk_mcap_out(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    scratch: MCarryTopKScratch,
    out_token_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_valid_mask: torch.Tensor,
    out_valid_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write full-vocabulary Top-K/logprobs with one logits scan."""

    if (
        logits.ndim != 2
        or not logits.is_cuda
        or logits.stride(1) != 1
        or logits.stride(0) < logits.shape[1]
    ):
        raise ValueError("logits must be a CUDA matrix with contiguous vocabulary rows")
    rows, vocab_size = int(logits.shape[0]), int(logits.shape[1])
    if (
        temperatures.device != logits.device
        or temperatures.dtype != torch.float32
        or temperatures.ndim != 1
        or not temperatures.is_contiguous()
        or temperatures.numel() not in (1, rows)
    ):
        raise ValueError(
            "temperatures must be contiguous CUDA float32 with one value or one per row"
        )
    if vocab_size != scratch.vocab_size or top_k != scratch.top_k:
        raise ValueError("bounded Top-K scratch shape does not match logits")
    if rows > scratch.local_keys.shape[0]:
        raise ValueError("bounded Top-K scratch has insufficient rows")
    expected = (rows, top_k)
    for name, tensor, dtype in (
        ("out_token_ids", out_token_ids, torch.int32),
        ("out_logprobs", out_logprobs, torch.float32),
        ("out_valid_mask", out_valid_mask, torch.bool),
    ):
        if (
            tensor.device != logits.device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != expected
        ):
            raise ValueError(f"{name} must be {dtype} CUDA tensor with shape {expected}")
    if (
        out_valid_counts.device != logits.device
        or out_valid_counts.dtype != torch.int32
        or tuple(out_valid_counts.shape) != (rows,)
    ):
        raise ValueError("out_valid_counts must be a CUDA int32 tensor")

    _launch_local_slice(
        rows,
        logits,
        temperatures,
        top_k,
        scratch,
        tile_offset=0,
        slice_size=vocab_size,
    )
    return finish(
        rows,
        top_k,
        scratch,
        out_token_ids,
        out_logprobs,
        out_valid_mask,
        out_valid_counts,
        repair_logits=logits,
        temperatures=temperatures,
    )


def _local_slice_plan(
    vocab_size: int, top_k: int, scratch: MCarryTopKScratch
) -> tuple[int, int, int, int, bool, bool, int]:
    tiles, keep, candidates, chunk_count = mcap_topk_tiling(vocab_size, top_k)
    if keep != scratch.keep:
        raise ValueError("bounded Top-K scratch retention width does not match its tiling")
    need_marginal = keep < top_k
    i32_select = os.environ.get("VLLM_GR_TOPK_SELECT", "i64").strip().lower() == "i32"
    # The i32 surrogate ranks a truncated key, so it must also retain the value
    # just below the cutoff to notice a collision there.  That extra slot is
    # needed even when ``keep == top_k``, where exact i64 selection is already
    # complete and no marginal is stored.
    keep_block = triton.next_power_of_2(
        keep + 1 if need_marginal or (i32_select and keep > 1) else keep
    )
    # An ambiguous i32 cutoff can only be resolved by the exact repair pass, so
    # verification runs even when the marginal check is not needed.
    # NOTE: the i32 surrogate ranks a truncated key, so ``_finalize_kernel``
    # only detects a collision at the cutoff (``last_in == first_out``).  A pair
    # of same-prefix keys that both fall into the dropped set can make the
    # reported marginal the second-best dropped key.  Default i64 selection is a
    # strict total order and is unaffected; the monolithic path shares this edge.
    verify = need_marginal or (i32_select and keep_block > 1)
    return tiles, keep, candidates, chunk_count, need_marginal, verify, keep_block


def _launch_local_slice(
    rows: int,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    scratch: MCarryTopKScratch,
    *,
    tile_offset: int,
    slice_size: int,
) -> None:
    tiles, keep, candidates, _, need_marginal, _, keep_block = _local_slice_plan(
        scratch.vocab_size, top_k, scratch
    )
    mode = os.environ.get("VLLM_GR_TOPK_SELECT", "i64").strip().lower()
    slice_tiles = -(-slice_size // LOCAL_TILE_SIZE)
    _local_keys_kernel[(rows, slice_tiles)](
        logits,
        temperatures,
        scratch.local_keys,
        scratch.local_marginal,
        scratch.partial_max,
        scratch.partial_sum,
        scratch.vocab_size,
        logits.stride(0),
        temperatures.numel(),
        candidates,
        tiles,
        keep,
        scratch.tile_flag,
        tiles,
        tile_offset,
        slice_size,
        BLOCK=LOCAL_TILE_SIZE,
        KEEP_BLOCK=keep_block,
        NEED_MARGINAL=need_marginal,
        I32_SELECT=mode == "i32",
        LANE_BITS=(LOCAL_TILE_SIZE - 1).bit_length(),
        # 4 warps measured fastest: the in-block bitonic network is
        # register-bound, so more warps per CTA only lower occupancy.
        num_warps=4,
    )


def local_slice(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    scratch: MCarryTopKScratch,
    rows: int,
    *,
    tile_offset: int = 0,
) -> None:
    """Scan one contiguous vocabulary slice into the persistent scratch."""

    slice_size = int(logits.shape[1])
    if (
        logits.ndim != 2
        or not logits.is_cuda
        or logits.stride(1) != 1
        or logits.stride(0) < slice_size
        or slice_size <= 0
    ):
        raise ValueError("logits slice must be a CUDA matrix with contiguous vocabulary rows")
    if rows <= 0 or rows > scratch.local_keys.shape[0] or logits.shape[0] < rows:
        raise ValueError("logits slice has insufficient rows for the scratch")
    if not 0 <= tile_offset * LOCAL_TILE_SIZE < scratch.vocab_size:
        raise ValueError("tile_offset is outside the configured vocabulary")
    if top_k != scratch.top_k:
        raise ValueError("bounded Top-K scratch shape does not match the requested width")
    _launch_local_slice(
        rows,
        logits,
        temperatures,
        top_k,
        scratch,
        tile_offset=tile_offset,
        slice_size=slice_size,
    )


def finish(
    rows: int,
    top_k: int,
    scratch: MCarryTopKScratch,
    out_token_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_valid_mask: torch.Tensor,
    out_valid_counts: torch.Tensor,
    *,
    repair_logits: torch.Tensor | None = None,
    temperatures: torch.Tensor | None = None,
    allow_unrepaired: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge the scanned slices and emit the candidates.

    ``repair_logits`` is needed when ``keep < top_k``.  A sliced caller that has
    already dropped the logits must run ``repair_plan``/``repair_fold`` itself
    and pass ``allow_unrepaired=True``; otherwise the emitted Top-K can be
    inexact.
    """

    if top_k != scratch.top_k:
        raise ValueError("bounded Top-K scratch shape does not match the requested width")
    tiles, _, candidates, chunk_count, _, verify, _ = _local_slice_plan(
        scratch.vocab_size, top_k, scratch
    )
    # A chunk keeps ``chunk_width`` keys and that width is also its slot stride,
    # so ``merge_keys`` has exactly ``chunk_count * chunk_width`` columns.
    chunk_width = merge_chunk_width(top_k)
    chunk_stride = chunk_count * chunk_width
    _merge_chunk_kernel[(rows, chunk_count)](
        scratch.local_keys,
        scratch.merge_keys,
        candidates,
        candidates,
        chunk_stride,
        BLOCK=MERGE_CHUNK,
        SELECT_BLOCK=triton.next_power_of_2(chunk_width),
        STORE_BLOCK=chunk_width,
        num_warps=8,
    )
    final_block = triton.next_power_of_2(chunk_stride)
    _merge_chunk_kernel[(rows, 1)](
        scratch.merge_keys,
        scratch.final_keys,
        chunk_stride,
        chunk_stride,
        top_k,
        BLOCK=final_block,
        SELECT_BLOCK=triton.next_power_of_2(top_k),
        STORE_BLOCK=top_k,
        num_warps=8,
    )
    _finalize_kernel[(rows,)](
        scratch.final_keys,
        scratch.partial_max,
        scratch.partial_sum,
        scratch.local_marginal,
        out_token_ids,
        out_logprobs,
        out_valid_mask,
        out_valid_counts,
        scratch.repair_flag,
        scratch.tile_flag,
        tiles,
        tiles,
        tiles,
        top_k,
        out_token_ids.stride(0),
        top_k,
        NEED_VERIFY=verify,
        SCAN_BLOCK=triton.next_power_of_2(tiles),
        OUTPUT_BLOCK=triton.next_power_of_2(top_k),
        num_warps=8,
    )
    if verify and repair_logits is None and not allow_unrepaired:
        raise ValueError(
            "exact Top-K repair is required; pass repair_logits, or "
            "allow_unrepaired=True after repair_plan/repair_fold/repair_emit"
        )
    if verify and repair_logits is not None:
        if temperatures is None:
            raise ValueError("temperatures are required when repair is enabled")
        # Rare path: rows whose retained set missed global winners (or whose i32
        # cutoff collided) are rebuilt by folding the row tile by tile.
        topk_block = triton.next_power_of_2(top_k)
        _repair_row_kernel[(rows,)](
            repair_logits,
            temperatures,
            scratch.repair_flag,
            scratch.final_keys,
            scratch.tile_flag,
            scratch.partial_max,
            scratch.partial_sum,
            out_token_ids,
            out_logprobs,
            out_valid_mask,
            out_valid_counts,
            scratch.vocab_size,
            repair_logits.stride(0),
            temperatures.numel(),
            top_k,
            tiles,
            tiles,
            tiles,
            out_token_ids.stride(0),
            top_k,
            BLOCK=LOCAL_TILE_SIZE,
            TOPK_BLOCK=topk_block,
            SCAN_BLOCK=triton.next_power_of_2(tiles),
            OUTPUT_BLOCK=topk_block,
            num_warps=4,
        )
    return out_token_ids, out_logprobs, out_valid_mask, out_valid_counts


def stream_slice(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    scratch: MCarryTopKScratch,
    rows: int,
    *,
    tile_offset: int,
) -> None:
    """Scan, merge and exactly repair one slice without host synchronization.

    Call in vocabulary order, starting at tile zero, then use ``repair_emit``
    after the final slice. The logits can be overwritten as soon as this call
    returns; the running Top-K and normalization statistics live in scratch.
    """

    local_slice(logits, temperatures, top_k, scratch, rows, tile_offset=tile_offset)
    tiles, keep, candidates, _, _, _, _ = _local_slice_plan(scratch.vocab_size, top_k, scratch)
    slice_size = int(logits.shape[1])
    slice_tiles = -(-slice_size // LOCAL_TILE_SIZE)
    topk_block = triton.next_power_of_2(top_k)
    # Pad both inputs equally so tl.cat also has a power-of-two length.
    candidate_block = triton.next_power_of_2(max(topk_block, slice_tiles * keep))
    _stream_slice_kernel[(rows,)](
        logits,
        temperatures,
        scratch.local_keys,
        scratch.local_marginal,
        scratch.tile_flag,
        scratch.repair_running,
        scratch.repair_flag,
        logits.stride(0),
        temperatures.numel(),
        candidates,
        tiles,
        2 * topk_block,
        scratch.vocab_size,
        tile_offset,
        slice_size,
        top_k,
        KEEP=keep,
        SLICE_TILES=slice_tiles,
        CANDIDATE_BLOCK=candidate_block,
        TILE_BLOCK=triton.next_power_of_2(slice_tiles),
        TOPK_BLOCK=topk_block,
        FIRST=tile_offset == 0,
        BLOCK=LOCAL_TILE_SIZE,
        num_warps=4,
    )


def repair_plan(rows: int, scratch: MCarryTopKScratch, slice_tiles: int) -> list[int]:
    """Host-side: vocabulary slices that still hold a flagged tile."""

    if slice_tiles <= 0:
        raise ValueError("slice_tiles must be positive")
    flagged = scratch.tile_flag[:rows].any(dim=0)
    if not bool(flagged.any().item()):
        return []
    tiles = torch.nonzero(flagged, as_tuple=False).flatten()
    return torch.unique(tiles // slice_tiles).tolist()


def repair_fold(
    rows: int,
    top_k: int,
    scratch: MCarryTopKScratch,
    logits_slice: torch.Tensor,
    temperatures: torch.Tensor,
    *,
    tile_offset: int,
    first: bool,
) -> None:
    """Refold the flagged tiles of one re-scanned slice into the running set."""

    slice_size = int(logits_slice.shape[1])
    tiles, _, _, _, _, _, _ = _local_slice_plan(scratch.vocab_size, top_k, scratch)
    topk_block = triton.next_power_of_2(top_k)
    _repair_fold_kernel[(rows,)](
        logits_slice,
        temperatures,
        scratch.repair_flag,
        scratch.final_keys,
        scratch.repair_running,
        scratch.tile_flag,
        scratch.vocab_size,
        logits_slice.stride(0),
        temperatures.numel(),
        top_k,
        2 * topk_block,
        tiles,
        tile_offset,
        slice_size,
        -(-slice_size // LOCAL_TILE_SIZE),
        tiles,
        top_k,
        BLOCK=LOCAL_TILE_SIZE,
        TOPK_BLOCK=topk_block,
        FIRST=first,
        num_warps=4,
    )


def repair_emit(
    rows: int,
    top_k: int,
    scratch: MCarryTopKScratch,
    out_token_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_valid_mask: torch.Tensor,
    out_valid_counts: torch.Tensor,
) -> None:
    """Emit every row the sliced repair pass rebuilt."""

    tiles, _, _, _, _, _, _ = _local_slice_plan(scratch.vocab_size, top_k, scratch)
    topk_block = triton.next_power_of_2(top_k)
    _repair_emit_kernel[(rows,)](
        scratch.repair_running,
        scratch.repair_flag,
        scratch.partial_max,
        scratch.partial_sum,
        out_token_ids,
        out_logprobs,
        out_valid_mask,
        out_valid_counts,
        2 * topk_block,
        tiles,
        tiles,
        out_token_ids.stride(0),
        top_k,
        SCAN_BLOCK=triton.next_power_of_2(tiles),
        OUTPUT_BLOCK=triton.next_power_of_2(top_k),
        num_warps=4,
    )


__all__ = [
    "MCarryTopKScratch",
    "allocate_mcap_topk_scratch",
    "finish",
    "local_slice",
    "mcap_topk_tiling",
    "merge_chunk_width",
    "repair_emit",
    "repair_fold",
    "repair_plan",
    "stream_slice",
    "unconstrained_topk_mcap_out",
]
