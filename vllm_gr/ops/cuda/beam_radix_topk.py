# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exact request-level Beam Top-K using a multi-block radix threshold."""

from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl

from vllm_gr.ops.cuda.topk_sort_key import pack_score_index, unpack_score
from vllm_gr.v1.worker.cudagraph_capture import is_recoverable_capture_error

# mypy: disable-error-code="misc,no-untyped-def"

TILE = 2048
BINS = 256


@dataclass(frozen=True, slots=True)
class BeamRadixScratch:
    keys: torch.Tensor
    histograms: torch.Tensor
    cutoff: torch.Tensor
    rank: torch.Tensor
    better_offsets: torch.Tensor
    equal_offsets: torch.Tensor
    better_count: torch.Tensor
    selected_count: torch.Tensor
    winners: torch.Tensor
    scores: torch.Tensor
    indices: torch.Tensor

    def device_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))


@triton.jit
def _candidate_keys(
    Parents,
    Locals,
    Valid,
    flat,
    request,
    INPUT_WIDTH: tl.constexpr,
    CANDIDATE_WIDTH: tl.constexpr,
    COUNT: tl.constexpr,
    HAS_MASK: tl.constexpr,
    EXCLUDE_MASKED: tl.constexpr,
):
    in_range = flat < COUNT
    valid = in_range
    if HAS_MASK:
        valid = valid & tl.load(Valid + request * COUNT + flat, in_range, other=False)
    row = request * INPUT_WIDTH + flat // CANDIDATE_WIDTH
    parent = tl.load(Parents + row, valid, other=-float("inf"))
    local = tl.load(Locals + request * COUNT + flat, valid, other=-float("inf"))
    packed = pack_score_index(parent + local, flat, in_range)
    # Bias the signed score word to ascending unsigned order. This preserves
    # the existing score canonicalization and leaves UINT32_MAX for padding.
    key = (packed >> 32).to(tl.uint32) ^ 0x80000000
    live = valid if EXCLUDE_MASKED else in_range
    return tl.where(live, key, 0xFFFFFFFF)


@triton.jit
def _radix_histogram(
    Parents,
    Locals,
    Valid,
    Keys,
    Histograms,
    Cutoff,
    INPUT_WIDTH: tl.constexpr,
    CANDIDATE_WIDTH: tl.constexpr,
    COUNT: tl.constexpr,
    TILES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BINS: tl.constexpr,
    SHIFT: tl.constexpr,
    HAS_MASK: tl.constexpr,
    EXCLUDE_MASKED: tl.constexpr,
):
    request, tile = tl.program_id(0), tl.program_id(1)
    flat = tile * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offsets = request * COUNT + flat
    if SHIFT == 24:
        keys = _candidate_keys(
            Parents,
            Locals,
            Valid,
            flat,
            request,
            INPUT_WIDTH,
            CANDIDATE_WIDTH,
            COUNT,
            HAS_MASK,
            EXCLUDE_MASKED,
        )
        tl.store(Keys + offsets, keys, flat < COUNT)
        eligible = (flat < COUNT) & (keys != 0xFFFFFFFF)
    else:
        keys = tl.load(Keys + offsets, flat < COUNT, other=0xFFFFFFFF)
        cutoff = tl.load(Cutoff + request)
        eligible = (keys != 0xFFFFFFFF) & ((keys >> (SHIFT + 8)) == (cutoff >> (SHIFT + 8)))
    digit = ((keys >> SHIFT) & (BINS - 1)).to(tl.int32)
    hist = tl.histogram(digit, BINS, mask=eligible)
    index = (request * TILES + tile) * BINS + tl.arange(0, BINS)
    tl.store(Histograms + index, hist)


@triton.jit
def _radix_choose(
    Histograms,
    Cutoff,
    Rank,
    TILES: tl.constexpr,
    TILE_COUNT: tl.constexpr,
    BINS: tl.constexpr,
    SHIFT: tl.constexpr,
    SELECTION: tl.constexpr,
):
    request = tl.program_id(0)
    tile = tl.arange(0, TILE_COUNT)
    digit = tl.arange(0, BINS)
    offsets = (request * TILES + tile[:, None]) * BINS + digit[None, :]
    histogram = tl.sum(tl.load(Histograms + offsets, tile[:, None] < TILES, other=0), axis=0)
    previous = tl.cumsum(histogram) - histogram
    total = tl.sum(histogram, axis=0)
    if SHIFT == 24:
        rank = SELECTION - 1
        cutoff = tl.full((), 0, tl.uint32)
    else:
        rank = tl.load(Rank + request)
        cutoff = tl.load(Cutoff + request)
    chosen = tl.min(tl.where(previous + histogram > rank, digit, BINS), axis=0)
    dropped = tl.sum(tl.where(digit == chosen, previous, 0), axis=0)
    next_cutoff = cutoff | (chosen.to(tl.uint32) << SHIFT)
    # If fewer than SELECTION candidates are valid, retain every valid key.
    tl.store(Cutoff + request, tl.where(rank < total, next_cutoff, 0xFFFFFFFF))
    tl.store(Rank + request, rank - dropped)


@triton.jit
def _count_winners(
    Keys,
    Cutoff,
    Better,
    Equal,
    COUNT: tl.constexpr,
    TILES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    request, tile = tl.program_id(0), tl.program_id(1)
    flat = tile * TILE_SIZE + tl.arange(0, TILE_SIZE)
    key = tl.load(Keys + request * COUNT + flat, flat < COUNT, other=0xFFFFFFFF)
    cutoff = tl.load(Cutoff + request)
    slot = request * TILES + tile
    tl.store(Better + slot, tl.sum((key < cutoff).to(tl.int32), axis=0))
    tl.store(Equal + slot, tl.sum(((key == cutoff) & (key != 0xFFFFFFFF)).to(tl.int32), axis=0))


@triton.jit
def _scan_winners(
    Better,
    Equal,
    BetterCount,
    SelectedCount,
    TILES: tl.constexpr,
    TILE_COUNT: tl.constexpr,
    SELECTION: tl.constexpr,
):
    request = tl.program_id(0)
    tile = tl.arange(0, TILE_COUNT)
    offsets = request * TILES + tile
    better = tl.load(Better + offsets, tile < TILES, other=0)
    equal = tl.load(Equal + offsets, tile < TILES, other=0)
    tl.store(Better + offsets, tl.cumsum(better) - better, tile < TILES)
    tl.store(Equal + offsets, tl.cumsum(equal) - equal, tile < TILES)
    better_count = tl.sum(better, axis=0)
    tl.store(BetterCount + request, better_count)
    tl.store(SelectedCount + request, tl.minimum(better_count + tl.sum(equal, axis=0), SELECTION))


@triton.jit
def _gather_winners(
    Keys,
    Cutoff,
    Better,
    Equal,
    BetterCount,
    Winners,
    COUNT: tl.constexpr,
    TILES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    SELECTION: tl.constexpr,
):
    request, tile = tl.program_id(0), tl.program_id(1)
    flat = tile * TILE_SIZE + tl.arange(0, TILE_SIZE)
    key = tl.load(Keys + request * COUNT + flat, flat < COUNT, other=0xFFFFFFFF)
    cutoff = tl.load(Cutoff + request)
    better = key < cutoff
    equal = (key == cutoff) & (key != 0xFFFFFFFF)
    slot = request * TILES + tile
    better_slot = tl.load(Better + slot) + tl.cumsum(better.to(tl.int32)) - 1
    # Visit cutoff ties in flat index order, across tiles and within each tile.
    equal_slot = (
        tl.load(BetterCount + request) + tl.load(Equal + slot) + tl.cumsum(equal.to(tl.int32)) - 1
    )
    output = tl.where(better, better_slot, equal_slot)
    packed = ((key ^ 0x80000000).to(tl.int64) << 32) | flat.to(tl.int64)
    tl.store(
        Winners + request * SELECTION + output, packed, (better | equal) & (output < SELECTION)
    )


@triton.jit
def _sort_winners(
    Winners,
    SelectedCount,
    Scores,
    Indices,
    SELECTION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    offset = request * SELECTION + rank
    count = tl.load(SelectedCount + request)
    padding = tl.full((), 0x7F800000FFFFFFFF, tl.int64)
    packed = tl.load(Winners + offset, rank < count, other=padding)
    ordered = tl.sort(packed, descending=False)
    tl.store(Scores + offset, unpack_score(ordered), rank < SELECTION)
    tl.store(Indices + offset, ordered.to(tl.int32), rank < SELECTION)


def allocate_beam_radix_scratch(
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    selection_width: int,
    device: torch.device,
) -> BeamRadixScratch:
    """Allocate stable workspace for this request geometry."""
    count = beam_width * candidate_width
    tiles = triton.cdiv(count, TILE)

    def int_buffer(*shape: int) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=torch.int32)

    return BeamRadixScratch(
        keys=torch.empty((batch_size, count), device=device, dtype=torch.uint32),
        histograms=int_buffer(batch_size, tiles, BINS),
        cutoff=torch.empty((batch_size,), device=device, dtype=torch.uint32),
        rank=int_buffer(batch_size),
        better_offsets=int_buffer(batch_size, tiles),
        equal_offsets=int_buffer(batch_size, tiles),
        better_count=int_buffer(batch_size),
        selected_count=int_buffer(batch_size),
        winners=torch.empty((batch_size, selection_width), device=device, dtype=torch.int64),
        scores=torch.empty((batch_size, selection_width), device=device, dtype=torch.float32),
        indices=int_buffer(batch_size, selection_width),
    )


def beam_radix_topk_out(
    cumulative_logprobs: torch.Tensor,
    top_logprobs: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None,
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    selection_width: int,
    scratch: BeamRadixScratch,
    *,
    mask_invalid_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Select from contiguous FP32 parent and local scores, in request-major order.

    Write sorted FP32 scores and request-local int32 indices into matching
    caller-owned scratch, padded with ``(-inf, -1)`` if selection is short.
    """
    count = beam_width * candidate_width
    tiles = triton.cdiv(count, TILE)
    tile_count = triton.next_power_of_2(tiles)
    grid = (batch_size, tiles)
    rows = (batch_size,)
    for shift in (24, 16, 8, 0):
        _radix_histogram[grid](
            cumulative_logprobs,
            top_logprobs,
            candidate_valid_mask,
            scratch.keys,
            scratch.histograms,
            scratch.cutoff,
            beam_width,
            candidate_width,
            count,
            tiles,
            TILE,
            BINS,
            shift,
            candidate_valid_mask is not None,
            mask_invalid_indices,
            num_warps=8,
        )
        _radix_choose[rows](
            scratch.histograms,
            scratch.cutoff,
            scratch.rank,
            tiles,
            tile_count,
            BINS,
            shift,
            selection_width,
            num_warps=8,
        )
    _count_winners[grid](
        scratch.keys,
        scratch.cutoff,
        scratch.better_offsets,
        scratch.equal_offsets,
        count,
        tiles,
        TILE,
        num_warps=8,
    )
    _scan_winners[rows](
        scratch.better_offsets,
        scratch.equal_offsets,
        scratch.better_count,
        scratch.selected_count,
        tiles,
        tile_count,
        selection_width,
    )
    _gather_winners[grid](
        scratch.keys,
        scratch.cutoff,
        scratch.better_offsets,
        scratch.equal_offsets,
        scratch.better_count,
        scratch.winners,
        count,
        tiles,
        TILE,
        selection_width,
        num_warps=8,
    )
    _sort_winners[rows](
        scratch.winners,
        scratch.selected_count,
        scratch.scores,
        scratch.indices,
        selection_width,
        triton.next_power_of_2(selection_width),
        num_warps=8,
    )
    return scratch.scores, scratch.indices, selection_width


@dataclass(frozen=True, slots=True)
class BeamRadixSelectionGraph:
    """A startup capture bound to one persistent candidate workspace row."""

    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    scratch: BeamRadixScratch

    @classmethod
    def capture(
        cls,
        parent_scores: torch.Tensor,
        local_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        scratch: BeamRadixScratch,
    ) -> "BeamRadixSelectionGraph | None":
        width, candidate_width = local_scores.shape
        device = local_scores.device
        inputs = (parent_scores, local_scores, candidate_mask)
        args = (*inputs, 1, width, candidate_width, width, scratch)
        stream = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        side.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            # The startup workspace has no live candidates. An empty mask
            # warms compilation without reading the uninitialized scores.
            candidate_mask.zero_()
            beam_radix_topk_out(*args, mask_invalid_indices=True)
            try:
                with torch.cuda.graph(graph, stream=side):
                    beam_radix_topk_out(*args, mask_invalid_indices=True)
            except RuntimeError as error:
                if not is_recoverable_capture_error(error):
                    raise
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Radix selection capture did not leave capture mode"
                    ) from error
                torch.accelerator.synchronize(device)
                return None
        stream.wait_stream(side)
        return cls(graph, inputs, scratch)

    def replay(
        self,
        parent_scores: torch.Tensor,
        local_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        if torch.cuda.is_current_stream_capturing() or any(
            live.data_ptr() != bound.data_ptr() or live.shape != bound.shape
            for live, bound in zip((parent_scores, local_scores, candidate_mask), self.inputs)
        ):
            return None
        self.graph.replay()
        return self.scratch.scores, self.scratch.indices, self.scratch.scores.shape[1]
