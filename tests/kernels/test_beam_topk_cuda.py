# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reference guards for the bounded-retention Top-K and its candidate merge.

Both pipelines must reproduce the stable ``(-score, index)`` order of the CPU
oracle bit for bit, and a vocabulary-sliced scan must match the single-shot one.
"""

import pytest
import torch

from vllm_gr.ops.gpu.beam_selection import (
    _hierarchical_topk_out,
    allocate_beam_selection_scratch,
)
from vllm_gr.ops.gpu.unconstrained_topk import (
    allocate_unconstrained_topk_scratch,
    unconstrained_topk_out,
)
from vllm_gr.ops.gpu.unconstrained_topk_mcap import (
    LOCAL_KEEP_MAX,
    LOCAL_TILE_SIZE,
    MCarryTopKScratch,
    allocate_flash_logits,
    allocate_mcap_topk_scratch,
    finish,
    flash_logits_rows,
    local_slice,
    mcap_topk_tiling,
    merge_chunk_width,
    repair_emit,
    repair_fold,
    repair_plan,
    stream_slice,
    unconstrained_topk_mcap_out,
)
from vllm_gr.v1.worker.gpu.beam_stage_runner import head_slice_plan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _run_both(
    logits: torch.Tensor, temperatures: torch.Tensor, top_k: int, *, select: str = "i64"
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    rows = logits.shape[0]
    legacy_scratch = allocate_unconstrained_topk_scratch(
        rows, logits.shape[1], top_k, logits.device
    )
    new_scratch = allocate_mcap_topk_scratch(
        rows, logits.shape[1], top_k, logits.device, select=select
    )
    outputs = []
    for launch, scratch in (
        (unconstrained_topk_out, legacy_scratch),
        (unconstrained_topk_mcap_out, new_scratch),
    ):
        token_ids = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
        logprobs = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
        valid = torch.empty((rows, top_k), dtype=torch.bool, device=logits.device)
        counts = torch.empty((rows,), dtype=torch.int32, device=logits.device)
        launch(logits, temperatures, top_k, scratch, token_ids, logprobs, valid, counts)
        outputs.append((token_ids, logprobs, valid, counts))
    return outputs


def _oracle(
    logits: torch.Tensor, temperatures: torch.Tensor, top_k: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    ids, values, masks = [], [], []
    for row in range(logits.shape[0]):
        temperature = float(temperatures[min(row, temperatures.numel() - 1)])
        if temperature == 0.0:
            temperature = 1.0
        scores = logits[row].double() / temperature
        ids_expected = torch.argsort(scores, descending=True, stable=True)[:top_k].to(torch.int32)
        logprobs = torch.log_softmax(scores, dim=0)
        count = int(torch.isfinite(scores).sum().item())
        count = min(count, top_k)
        ids.append(ids_expected[:count])
        values.append(logprobs[ids_expected[:count]].float())
        masks.append(count)
    return ids, values, masks


@pytest.mark.parametrize(  # type: ignore[misc]
    "vocab_size,top_k",
    [
        (17, 1),
        (17, 32),
        (1031, 3),
        (2048, 256),
        (5123, 128),
        (8192, 64),
        # Non-power-of-two widths whose padded selection block is wider than the
        # number of keys kept, at the final stage (3 -> 4) and in a multi-chunk
        # first stage (6 keys per chunk out of 173 chunks).
        (8192, 100),
        (176256, 6),
    ],
)
def test_mcap_matches_legacy_and_oracle(vocab_size: int, top_k: int) -> None:
    generator = torch.Generator().manual_seed(913)
    logits_cpu = torch.randn((3, vocab_size), generator=generator)
    logits_cpu[0].zero_()  # Every score ties; the lower token ID must win.
    logits = logits_cpu.cuda()
    temperatures = torch.tensor([0.0, 0.75, 1.0], device="cuda")

    (ref_ids, ref_lp, ref_valid, ref_counts), (new_ids, new_lp, new_valid, new_counts) = _run_both(
        logits, temperatures, top_k
    )

    assert torch.equal(new_ids, ref_ids), "token ids diverged from the legacy path"
    assert torch.equal(new_lp, ref_lp), "logprobs diverged from the legacy path"
    assert torch.equal(new_valid, ref_valid)
    assert torch.equal(new_counts, ref_counts)

    ids_expected, lp_expected, masks = _oracle(logits_cpu, temperatures.cpu(), top_k)
    assert new_counts.tolist() == masks
    for row, expected in enumerate(ids_expected):
        assert new_ids[row, : masks[row]].tolist() == expected.tolist()
        torch.testing.assert_close(
            new_lp[row, : masks[row]].cpu(), lp_expected[row], rtol=1e-4, atol=1e-4
        )
        assert new_valid[row, : masks[row]].all().item()
        assert not new_valid[row, masks[row] :].any().item()


def test_mcap_repair_path_is_exact_and_actually_taken() -> None:
    """One tile supplies most of the global Top-K, so the marginal check fails."""

    vocab_size = 4096
    top_k = 256
    logits = torch.full((1, vocab_size), -100.0, device="cuda")
    # 200 strictly decreasing scores inside the first tile: keeping 32 of them
    # cannot produce the exact Top-K, so the repair pass must run.
    logits[0, :200] = torch.linspace(50.0, 0.0, 200, device="cuda")
    temperatures = torch.ones(1, dtype=torch.float32, device="cuda")
    scratch = allocate_mcap_topk_scratch(1, vocab_size, top_k, logits.device)
    token_ids = torch.empty((1, top_k), dtype=torch.int32, device="cuda")
    logprobs = torch.empty((1, top_k), dtype=torch.float32, device="cuda")
    valid = torch.empty((1, top_k), dtype=torch.bool, device="cuda")
    counts = torch.empty((1,), dtype=torch.int32, device="cuda")

    unconstrained_topk_mcap_out(
        logits, temperatures, top_k, scratch, token_ids, logprobs, valid, counts
    )

    assert int(scratch.repair_flag[0].item()) == 1
    expected_ids = torch.argsort(logits[0].double(), descending=True, stable=True)[:top_k]
    assert token_ids[0].tolist() == expected_ids.to(torch.int32).tolist()
    torch.testing.assert_close(
        logprobs[0].cpu(),
        torch.log_softmax(logits[0].double(), dim=0)[expected_ids].float().cpu(),
        rtol=1e-4,
        atol=1e-4,
    )
    assert valid[0].all().item()
    assert int(counts[0].item()) == top_k


@pytest.mark.l2  # type: ignore[misc]
@pytest.mark.parametrize("top_k", [769, 1024])  # type: ignore[misc]
def test_mcap_wide_repair_keeps_exact_order(top_k: int) -> None:
    """Exercise full-tile repair with ties, a partial tile, and short rows."""
    vocab_size = 3 * 1024 + 513
    logits = torch.full((3, vocab_size), -float("inf"), device="cuda")
    logits[0, :2048] = 5
    logits[1, :1024] = torch.linspace(0, 2, 1024, device="cuda")
    logits[1, -513:] = 4
    logits[2, :65] = -0.5
    logits[2, 65] = float("nan")
    logits[2, 66] = float("inf")
    temperatures = torch.ones(3, device="cuda")
    scratch = allocate_mcap_topk_scratch(3, vocab_size, top_k, logits.device, select="i64")
    token_ids = torch.empty((3, top_k), dtype=torch.int32, device=logits.device)
    logprobs = torch.empty((3, top_k), dtype=torch.float32, device=logits.device)
    valid = torch.empty((3, top_k), dtype=torch.bool, device=logits.device)
    counts = torch.empty(3, dtype=torch.int32, device=logits.device)
    unconstrained_topk_mcap_out(
        logits, temperatures, top_k, scratch, token_ids, logprobs, valid, counts
    )
    assert scratch.repair_flag[:2].tolist() == [1, 1]
    assert scratch.tile_flag[1, -1].item() == 1

    finite = torch.where(torch.isfinite(logits), logits, -float("inf"))
    ids, scores, lengths = _oracle(finite.cpu(), temperatures.cpu(), top_k)
    assert counts.tolist() == lengths
    for row, length in enumerate(lengths):
        torch.testing.assert_close(token_ids[row, :length].cpu(), ids[row], rtol=0, atol=0)
        torch.testing.assert_close(logprobs[row, :length].cpu(), scores[row], rtol=1e-5, atol=1e-5)
        assert valid[row, :length].all().item()
        assert not valid[row, length:].any().item()


def test_mcap_accepts_padded_rows_empty_tiles_and_nan() -> None:
    storage = torch.empty((2, 2112), dtype=torch.float32, device="cuda")
    logits = storage[:, :2048]
    logits[0, :1024].zero_()
    logits[0, 1024:].fill_(-float("inf"))
    logits[1].fill_(-float("inf"))
    logits[1, 0] = float("nan")
    temperatures = torch.ones(2, dtype=torch.float32, device="cuda")

    (ref_ids, ref_lp, ref_valid, ref_counts), (new_ids, new_lp, new_valid, new_counts) = _run_both(
        logits, temperatures, 3
    )

    assert torch.equal(new_ids, ref_ids)
    assert torch.equal(new_lp, ref_lp)
    assert torch.equal(new_valid, ref_valid)
    assert torch.equal(new_counts, ref_counts)
    assert logits.stride() == (2112, 1)
    assert new_ids[0].tolist() == [0, 1, 2]
    torch.testing.assert_close(
        new_lp[0], torch.log_softmax(logits[0], dim=0)[:3], rtol=1e-4, atol=1e-4
    )
    assert new_valid[0].all().item()
    assert not new_valid[1].any().item()
    assert new_counts.tolist() == [3, 0]


def test_mcap_counts_only_finite_candidates() -> None:
    logits = torch.full((3, 2048), -float("inf"), device="cuda")
    logits[0, 10] = 0.0
    logits[1, 1] = 1.0
    logits[1, 2] = 0.0
    logits[1, 3] = float("nan")
    logits[2, 4] = float("nan")
    top_k = 4

    (ref_ids, ref_lp, ref_valid, ref_counts), (new_ids, new_lp, new_valid, new_counts) = _run_both(
        logits, torch.ones(3, device="cuda"), top_k
    )

    assert torch.equal(new_ids, ref_ids)
    assert torch.equal(new_lp, ref_lp)
    assert torch.equal(new_valid, ref_valid)
    assert torch.equal(new_counts, ref_counts)
    assert new_counts.tolist() == [1, 2, 0]
    assert new_valid.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, False, False],
    ]
    assert new_ids[0, 0].item() == 10
    assert new_ids[1, :2].tolist() == [1, 2]
    torch.testing.assert_close(new_lp[0, 0].cpu(), torch.tensor(0.0))
    torch.testing.assert_close(
        new_lp[1, :2].cpu(), torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)
    )
    assert torch.isneginf(new_lp[~new_valid]).all().item()


def test_mcap_tiling_is_shared_by_allocation_and_launch() -> None:
    vocab_size = 176256
    top_k = 256
    tiles, keep, candidates, chunk_count = mcap_topk_tiling(vocab_size, top_k)
    assert tiles == (vocab_size + 1023) // 1024
    assert keep == min(top_k, LOCAL_KEEP_MAX)
    assert candidates == tiles * keep
    assert chunk_count == -(-candidates // 1024)
    scratch = allocate_mcap_topk_scratch(2, vocab_size, top_k, torch.device("cuda"))
    assert scratch.local_keys.shape == (2, candidates)
    assert scratch.merge_keys.shape == (2, chunk_count * merge_chunk_width(top_k))
    assert scratch.tile_flag.shape == (2, tiles)
    assert scratch.keep == keep
    # A chunk never stores more keys than it can hold, so the padded selection
    # width cannot make a row of ``merge_keys`` wider than the allocation.
    assert merge_chunk_width(3) == 3
    assert merge_chunk_width(4096) == 1024


def test_mcap_matches_legacy_on_onerec_shape() -> None:
    """Real OneRec shape: 173 vocabulary tiles and a 32-wide local retention."""

    vocab_size, top_k, rows = 176256, 256, 2
    generator = torch.Generator().manual_seed(20261031)
    logits = torch.randn((rows, vocab_size), generator=generator).cuda()
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")

    (ref_ids, ref_lp, ref_valid, ref_counts), (new_ids, new_lp, new_valid, new_counts) = _run_both(
        logits, temperatures, top_k
    )

    assert mcap_topk_tiling(vocab_size, top_k) == (
        173,
        LOCAL_KEEP_MAX,
        173 * LOCAL_KEEP_MAX,
        -(-(173 * LOCAL_KEEP_MAX) // 1024),
    )
    assert torch.equal(new_ids, ref_ids)
    assert torch.equal(new_lp, ref_lp)
    assert torch.equal(new_valid, ref_valid)
    assert torch.equal(new_counts, ref_counts)


def test_mcap_i32_surrogate_stays_bit_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional 32-bit surrogate must not change any emitted value."""

    generator = torch.Generator().manual_seed(4242)
    vocab_size, top_k = 8192, 128
    base = torch.randn((2, vocab_size), generator=generator).cuda()
    tied = (base * 4).round() / 4
    duplicated = base.clone()
    duplicated[:, 1000:3000] = 3.0

    for logits in (base, tied, duplicated):
        (
            (ref_ids, ref_lp, ref_valid, ref_counts),
            (
                new_ids,
                new_lp,
                new_valid,
                new_counts,
            ),
        ) = _run_both(logits, torch.ones(2, device="cuda"), top_k, select="i32")
        assert torch.equal(new_ids, ref_ids)
        assert torch.equal(new_lp, ref_lp)
        assert torch.equal(new_valid, ref_valid)
        assert torch.equal(new_counts, ref_counts)


@pytest.mark.parametrize("top_k", [1, 2])  # type: ignore[misc]
def test_mcap_i32_small_k_repairs_collision_at_cutoff(
    monkeypatch: pytest.MonkeyPatch, top_k: int
) -> None:
    """An i32 prefix collision must still return the exact Top-K when keep == top_k."""

    # 1.0 and the next float share one truncated surrogate prefix, and the third
    # value is the larger of the three, so the surrogate alone would rank the tie
    # by lane and drop the true winner.
    rows = 3
    logits = torch.full((rows, 1024), -float("inf"), device="cuda")
    logits[:, 0] = 1.0
    logits[:, 1] = 1.0
    logits[:, 2] = 1.0000001192092896
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")

    (ref_ids, ref_lp, ref_valid, ref_counts), (new_ids, new_lp, new_valid, new_counts) = _run_both(
        logits, temperatures, top_k, select="i32"
    )

    assert new_ids.cpu().tolist() == [[2, 0][:top_k]] * rows
    assert torch.equal(new_ids, ref_ids)
    assert torch.equal(new_lp, ref_lp)
    assert torch.equal(new_valid, ref_valid)
    assert torch.equal(new_counts, ref_counts)


def test_mcap_vocabulary_slices_match_full_scan() -> None:
    """A vocabulary-sliced scan must reproduce the single-shot scan."""

    vocab_size, top_k, rows, slice_cols = 8192, 128, 2, 2048
    generator = torch.Generator().manual_seed(77)
    logits = torch.randn((rows, vocab_size), generator=generator).cuda()
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")

    def buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
            torch.empty((rows,), dtype=torch.int32, device="cuda"),
        )

    full_scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device)
    full_out = buffers()
    unconstrained_topk_mcap_out(logits, temperatures, top_k, full_scratch, *full_out)

    sliced_scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device)
    for start in range(0, vocab_size, slice_cols):
        local_slice(
            logits[:, start : start + slice_cols],
            temperatures,
            top_k,
            sliced_scratch,
            rows,
            tile_offset=start // LOCAL_TILE_SIZE,
        )
    sliced_out = buffers()
    finish(
        rows,
        top_k,
        sliced_scratch,
        *sliced_out,
        repair_logits=logits,
        temperatures=temperatures,
    )

    for got, want in zip(sliced_out, full_out):
        assert torch.equal(got, want), "sliced scan diverged from the single-shot scan"


def test_mcap_sliced_repair_matches_full_scan() -> None:
    """A flagged tile must be repaired by re-scanning only its own slice."""

    vocab_size, top_k, rows, slice_cols = 8192, 128, 2, 2048
    generator = torch.Generator().manual_seed(99)
    logits = torch.full((rows, vocab_size), -100.0, device="cuda")
    logits[:] = torch.randn((rows, vocab_size), generator=generator).cuda()
    # The whole Top-K comes from the first tile, so keeping a bounded number of
    # its candidates cannot be exact and only that tile is flagged.  The tail is
    # strictly ordered (no ties) so no other tile can look unsafe.
    logits[:, :200] = torch.linspace(50.0, 0.0, 200, device="cuda")
    logits[:, 200:] = -100.0 - torch.arange(vocab_size - 200, device="cuda") / 1.0e6
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")

    def buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
            torch.empty((rows,), dtype=torch.int32, device="cuda"),
        )

    reference = buffers()
    reference_scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device)
    unconstrained_topk_mcap_out(logits, temperatures, top_k, reference_scratch, *reference)

    scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device)
    sliced = buffers()
    for start in range(0, vocab_size, slice_cols):
        local_slice(
            logits[:, start : start + slice_cols],
            temperatures,
            top_k,
            scratch,
            rows,
            tile_offset=start // LOCAL_TILE_SIZE,
        )
    finish(rows, top_k, scratch, *sliced, allow_unrepaired=True)

    slices = repair_plan(rows, scratch, slice_cols // LOCAL_TILE_SIZE)
    # The hot tile must be re-scanned; neighbouring tiles can also look unsafe
    # because the retained set is deliberately incomplete.
    assert 0 in slices, f"the hot slice must be re-scanned, got {slices}"
    for index, slice_index in enumerate(slices):
        start = slice_index * slice_cols
        cols = min(slice_cols, vocab_size - start)
        repair_fold(
            rows,
            top_k,
            scratch,
            logits[:, start : start + cols],
            temperatures,
            tile_offset=start // LOCAL_TILE_SIZE,
            first=index == 0,
        )
    repair_emit(rows, top_k, scratch, *sliced)

    for got, want in zip(sliced, reference):
        assert torch.equal(got, want), "sliced repair diverged from the single-shot scan"


def test_budget_reduced_slice_width_matches_full_scan() -> None:
    """A budget-reduced width must stay tile-aligned, or tokens map to wrong tiles."""

    rows, vocab_size, top_k = 4096, 33000, 8
    use_slicing, cols = head_slice_plan(rows, vocab_size, 2, 29696, 1024, 128 * 2**20)
    assert use_slicing
    assert cols == 16384 and cols % LOCAL_TILE_SIZE == 0

    logits = torch.full((rows, vocab_size), -100.0, dtype=torch.bfloat16, device="cuda")
    # The unique maximum starts exactly at the second block boundary.
    logits[:, cols] = 100.0
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")

    def buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
            torch.empty((rows,), dtype=torch.int32, device="cuda"),
        )

    reference = buffers()
    reference_scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device)
    unconstrained_topk_mcap_out(logits, temperatures, top_k, reference_scratch, *reference)

    sliced = buffers()
    scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device)
    for start in range(0, vocab_size, cols):
        width = min(cols, vocab_size - start)
        local_slice(
            logits[:, start : start + width],
            temperatures,
            top_k,
            scratch,
            rows,
            tile_offset=start // LOCAL_TILE_SIZE,
        )
    finish(rows, top_k, scratch, *sliced, repair_logits=logits, temperatures=temperatures)

    assert (reference[0][:, 0] == cols).all(), "the full scan must select the unique maximum"
    for got, want in zip(sliced, reference):
        assert torch.equal(got, want), "budget-reduced slices diverged from the full scan"


@pytest.mark.parametrize(
    "top_k,mode",
    [(1, "i64"), (3, "i64"), (32, "i64"), (128, "i64"), (2, "i32"), (32, "i32")],
)  # type: ignore[misc]
def test_mcap_streaming_slices_repair_before_buffer_reuse(
    monkeypatch: pytest.MonkeyPatch, top_k: int, mode: str
) -> None:
    """Repair hot/tied tiles without a host read, even in a captured graph."""

    rows, vocab_size, slice_cols = 4, 8205, 2048
    logits = torch.randn((rows, vocab_size), device="cuda")
    logits[0].fill_(-10)
    logits[0, :3] = torch.tensor([1.0, 1.0, 1.0000001192092896], device="cuda")
    logits[1].zero_()  # Ties force exact repair beyond the retained prefix.
    logits[2, -200:] = torch.linspace(10, 100, 200, device="cuda")
    logits[3].fill_(-float("inf"))
    logits[3, :2] = torch.tensor([float("nan"), float("inf")], device="cuda")
    temperatures = torch.tensor([0.0, 0.75, 1.5, 1.0], device="cuda")

    def buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
            torch.empty(rows, dtype=torch.int32, device="cuda"),
        )

    scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device, select=mode)
    reference_scratch = allocate_mcap_topk_scratch(
        rows, vocab_size, top_k, logits.device, select=mode
    )
    result, expected = buffers(), buffers()
    buffer = torch.empty((rows, slice_cols), device="cuda")

    def run() -> None:
        for start in range(0, vocab_size, slice_cols):
            width = min(slice_cols, vocab_size - start)
            sliced = buffer[:, :width]
            sliced.copy_(logits[:, start : start + width])
            stream_slice(
                sliced, temperatures, top_k, scratch, rows, tile_offset=start // LOCAL_TILE_SIZE
            )
        repair_emit(rows, top_k, scratch, *result)

    run()  # Compile before capture.
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration in range(2):
        if iteration:
            # Replaying must reset the running set and pick up changed inputs.
            logits[2, :200].fill_(200)
        graph.replay()
        unconstrained_topk_mcap_out(logits, temperatures, top_k, reference_scratch, *expected)
        for got, want in zip(result, expected):
            assert torch.equal(got, want)


def test_mcap_single_candidate_width() -> None:
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn((2, 4096), generator=generator).cuda()
    (ref_ids, ref_lp, _, ref_counts), (new_ids, new_lp, _, new_counts) = _run_both(
        logits, torch.ones(2, device="cuda"), 1
    )
    assert torch.equal(new_ids, ref_ids)
    assert torch.equal(new_lp, ref_lp)
    assert torch.equal(new_counts, ref_counts)


def _merge_reference(
    cumulative: torch.Tensor,
    top_logprobs: torch.Tensor,
    valid: torch.Tensor,
    beam_width: int,
    candidate_width: int,
    selection_width: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batch = cumulative.numel() // beam_width
    expected = []
    for request in range(batch):
        rows = slice(request * beam_width, (request + 1) * beam_width)
        scores = cumulative[rows][:, None] + top_logprobs[rows]
        flat_valid = valid[rows].reshape(-1)
        flat_scores = scores.reshape(-1).masked_fill(~flat_valid, -float("inf"))
        order = torch.argsort(flat_scores, descending=True, stable=True)[:selection_width]
        expected.append((flat_scores[order], order.to(torch.int32)))
    return expected


@pytest.mark.parametrize(  # type: ignore[misc]
    "beam_width,candidate_width,selection_width",
    [
        (2, 4, 1),
        (128, 3, 1),
        (64, 64, 64),
        (256, 256, 256),
        (128, 32, 32),
        (1024, 256, 1024),
        (1024, 256, 64),
        (256, 256, 32),
    ],
)
def test_hierarchical_topk_matches_reference(
    beam_width: int, candidate_width: int, selection_width: int
) -> None:
    """The multi-way merge must reproduce the stable reference order."""

    batch = 2
    generator = torch.Generator().manual_seed(20261031)
    cumulative = torch.randn((batch * beam_width,), generator=generator).cuda()
    logprobs = torch.randn((batch * beam_width, candidate_width), generator=generator).cuda()
    valid = (torch.rand((batch * beam_width, candidate_width), generator=generator) > 0.1).cuda()

    scratch = allocate_beam_selection_scratch(
        batch, beam_width, candidate_width, selection_width, cumulative.device
    )
    scores, flat, stride = _hierarchical_topk_out(
        cumulative,
        logprobs,
        valid,
        batch,
        beam_width,
        candidate_width,
        selection_width,
        scratch,
        mask_invalid_indices=True,
    )

    expected = _merge_reference(
        cumulative, logprobs, valid, beam_width, candidate_width, selection_width
    )
    for request, (want_scores, want_flat) in enumerate(expected):
        got_scores = scores[request, :selection_width]
        got_flat = flat[request, :selection_width]
        assert torch.equal(got_flat, want_flat), f"request {request}: flat indices diverged"
        assert torch.equal(got_scores, want_scores), f"request {request}: scores diverged"
        assert stride >= selection_width


@pytest.mark.parametrize("mask_invalid_indices", [False, True])  # type: ignore[misc]
def test_hierarchical_top1_ties_and_invalid_candidates(mask_invalid_indices: bool) -> None:
    batch, beam_width, candidate_width = 4, 128, 3
    cumulative = torch.zeros(batch * beam_width, device="cuda")
    logprobs = torch.zeros((batch, beam_width * candidate_width), device="cuda")
    valid = torch.zeros_like(logprobs, dtype=torch.bool)
    # Ties across tiles, an empty request, a winner in the padded last tile,
    # and valid -inf candidates must retain the established index semantics.
    valid[0, 1] = valid[0, 257] = True
    logprobs[2].fill_(10.0)
    logprobs[2, -1] = 0.0
    valid[2, -1] = True
    logprobs[3].fill_(-float("inf"))
    valid[3].fill_(True)
    scratch = allocate_beam_selection_scratch(
        batch, beam_width, candidate_width, 1, cumulative.device
    )

    scores, flat, _ = _hierarchical_topk_out(
        cumulative,
        logprobs.reshape(-1, candidate_width),
        valid.reshape(-1, candidate_width),
        batch,
        beam_width,
        candidate_width,
        1,
        scratch,
        mask_invalid_indices=mask_invalid_indices,
    )

    assert flat[:, 0].tolist() == [1, -1 if mask_invalid_indices else 0, 383, 0]
    assert scores[:, 0].tolist() == [0.0, -float("inf"), 0.0, -float("inf")]


def _flash_supported() -> bool:
    from vllm_gr.ops.gpu.unconstrained_topk_mcap import _flash_topk_kernel

    return _flash_topk_kernel(torch.accelerator.current_device_index()) is not None


_SelectOut = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _select_out(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    scratch: MCarryTopKScratch,
    *,
    unit_temperature: bool,
) -> _SelectOut:
    rows = logits.shape[0]
    out = (
        torch.empty((rows, top_k), dtype=torch.int32, device=logits.device),
        torch.empty((rows, top_k), dtype=torch.float32, device=logits.device),
        torch.empty((rows, top_k), dtype=torch.bool, device=logits.device),
        torch.empty((rows,), dtype=torch.int32, device=logits.device),
    )
    unconstrained_topk_mcap_out(
        logits, temperatures, top_k, scratch, *out, unit_temperature=unit_temperature
    )
    return out


def _assert_same_selection(got: _SelectOut, want: _SelectOut) -> None:
    assert torch.equal(got[0], want[0])
    torch.testing.assert_close(got[1], want[1], rtol=1e-5, atol=1e-5)
    assert torch.equal(got[2], want[2])
    assert torch.equal(got[3], want[3])


def _flash_parity_logits() -> torch.Tensor:
    rows, vocab_size = 6, 2048
    generator = torch.Generator().manual_seed(20261122)
    logits = torch.randn((rows, vocab_size), generator=generator).to(torch.bfloat16)
    logits[0, :1024].zero_()
    logits[1, 0] = float("inf")
    logits[1, 1] = float("nan")
    logits[2].fill_(-float("inf"))
    # Signed zeros are equal scores: the smaller token ID must win.
    logits[3].fill_(-1.0)
    logits[3, 0] = -0.0
    logits[3, 1] = 0.0
    logits[4].fill_(-1.0)
    logits[4, 5] = 0.0
    logits[4, 6] = -0.0
    logits[4, 7] = -0.0
    return logits.cuda()


def _flash_backend(backend: str, monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """Select the Top-K behind ``select="flash"`` and return its recorded call shapes.

    ``flashinfer`` uses the installed library, so devices without FilteredTopK
    (for example L20) take the i64 fallback. ``reference_topk`` substitutes a
    stable ``torch.sort`` with the same smaller-index tie-break, so the whole
    Flash pipeline runs on every device. FlashInfer's own kernel is not
    exercised by that variant.
    """

    import vllm_gr.ops.gpu.unconstrained_topk_mcap as mcap

    calls: list[tuple[int, ...]] = []
    if backend == "reference_topk":

        def top_k(values: torch.Tensor, k: int, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(tuple(values.shape))
            ordered, order = torch.sort(values.float(), dim=1, descending=True, stable=True)
            return ordered[:, :k].to(values.dtype), order[:, :k]

        monkeypatch.setattr(mcap, "_flash_topk_kernel", lambda device_index: (top_k, 1))
    return calls


def _expect_flash(backend: str) -> bool:
    return backend == "reference_topk" or _flash_supported()


_FLASH_BACKENDS = pytest.mark.parametrize("backend", ["flashinfer", "reference_topk"])


@_FLASH_BACKENDS  # type: ignore[misc]
@pytest.mark.parametrize("top_k", [1, 7])  # type: ignore[misc]
def test_flash_topk_matches_i64_with_ties_signed_zero_and_nonfinite(
    backend: str, top_k: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _flash_backend(backend, monkeypatch)
    logits = _flash_parity_logits()
    rows, vocab_size = logits.shape
    temperatures = torch.zeros(rows, dtype=torch.float32, device="cuda")
    shared = allocate_flash_logits(rows, vocab_size, logits.dtype, logits.device)
    assert (shared is not None) == _expect_flash(backend)
    if shared is not None:
        shared.fill_(float("nan"))
    reference = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device, select="i64")
    flash = allocate_mcap_topk_scratch(
        rows, vocab_size, top_k, logits.device, select="flash", flash_logits=shared
    )
    want = _select_out(logits, temperatures, top_k, reference, unit_temperature=True)
    got = _select_out(logits, temperatures, top_k, flash, unit_temperature=True)

    _assert_same_selection(got, want)
    assert want[0][3, 0].item() == 0
    assert want[0][4, 0].item() == 5
    if shared is not None:
        # The sanitized copy replaces the NaN fill only when Flash executed.
        assert not torch.isnan(shared[:rows]).any()
    if backend == "reference_topk":
        assert calls == [(rows, vocab_size)]


@_FLASH_BACKENDS  # type: ignore[misc]
def test_flash_topk_checks_temperature_on_every_call_with_reused_scratch(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BF16 cannot hold 1.40625 / 0.7 and 1.4140625 / 0.7 apart; a stale
    # unit-temperature decision would store both as 2.015625 and pick token 0.
    calls = _flash_backend(backend, monkeypatch)
    logits = torch.full((2, 2048), -10.0, dtype=torch.bfloat16, device="cuda")
    logits[:, 0] = 1.40625
    logits[:, 1] = 1.4140625
    rows, vocab_size = logits.shape
    shared = allocate_flash_logits(rows, vocab_size, logits.dtype, logits.device)
    flash = allocate_mcap_topk_scratch(
        rows, vocab_size, 1, logits.device, select="flash", flash_logits=shared
    )
    reference = allocate_mcap_topk_scratch(rows, vocab_size, 1, logits.device, select="i64")
    for temperature, unit in ((1.0, True), (0.7, False), (1.0, True)):
        temperatures = torch.full((rows,), temperature, dtype=torch.float32, device="cuda")
        got = _select_out(logits, temperatures, 1, flash, unit_temperature=unit)
        want = _select_out(logits, temperatures, 1, reference, unit_temperature=unit)
        _assert_same_selection(got, want)
        assert got[0][:, 0].tolist() == [1, 1]
    if backend == "reference_topk":
        assert calls == [(rows, vocab_size)] * 2


@_FLASH_BACKENDS  # type: ignore[misc]
def test_flash_topk_writes_padded_output_rows(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FlashInfer returns compact rows, but the ``out`` API accepts rows with
    # trailing padding. K=7 with a row stride of 8 must keep every row aligned
    # and leave the padding column untouched.
    _flash_backend(backend, monkeypatch)
    logits = _flash_parity_logits()
    rows, vocab_size = logits.shape
    top_k, stride = 7, 8
    temperatures = torch.zeros(rows, dtype=torch.float32, device="cuda")
    shared = allocate_flash_logits(rows, vocab_size, logits.dtype, logits.device)
    flash = allocate_mcap_topk_scratch(
        rows, vocab_size, top_k, logits.device, select="flash", flash_logits=shared
    )
    reference = allocate_mcap_topk_scratch(rows, vocab_size, top_k, logits.device, select="i64")
    want = _select_out(logits, temperatures, top_k, reference, unit_temperature=True)

    ids = torch.full((rows, stride), -77, dtype=torch.int32, device="cuda")
    logprobs = torch.full((rows, stride), -123.0, dtype=torch.float32, device="cuda")
    valid = torch.ones((rows, stride), dtype=torch.bool, device="cuda")
    counts = torch.empty((rows,), dtype=torch.int32, device="cuda")
    unconstrained_topk_mcap_out(
        logits,
        temperatures,
        top_k,
        flash,
        ids[:, :top_k],
        logprobs[:, :top_k],
        valid[:, :top_k],
        counts,
        unit_temperature=True,
    )

    expected_ids = torch.full_like(ids, -77)
    expected_ids[:, :top_k] = want[0]
    expected_valid = torch.ones_like(valid)
    expected_valid[:, :top_k] = want[2]
    assert torch.equal(ids, expected_ids)
    assert torch.equal(valid, expected_valid)
    assert torch.equal(counts, want[3])
    torch.testing.assert_close(logprobs[:, :top_k], want[1], rtol=1e-5, atol=1e-5)
    assert (logprobs[:, top_k] == -123.0).all()


@_FLASH_BACKENDS  # type: ignore[misc]
def test_flash_topk_serves_wide_beam_multi_request_prefill(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Width and candidate count 512 with two requests: Prefill selects 2 rows,
    # which Flash serves, while Decode (1024 rows) cannot use it.
    calls = _flash_backend(backend, monkeypatch)
    top_k, prefill_rows, vocab_size = 512, 2, 4096
    generator = torch.Generator().manual_seed(512)
    logits = torch.randn((prefill_rows, vocab_size), generator=generator)
    logits = logits.to(torch.bfloat16).cuda()
    temperatures = torch.ones(prefill_rows, dtype=torch.float32, device="cuda")
    reserved = flash_logits_rows(512, 2)
    assert reserved == prefill_rows
    shared = allocate_flash_logits(reserved, vocab_size, logits.dtype, logits.device)
    assert (shared is not None) == _expect_flash(backend)
    flash = allocate_mcap_topk_scratch(
        prefill_rows, vocab_size, top_k, logits.device, select="flash", flash_logits=shared
    )
    reference = allocate_mcap_topk_scratch(
        prefill_rows, vocab_size, top_k, logits.device, select="i64"
    )
    got = _select_out(logits, temperatures, top_k, flash, unit_temperature=True)
    want = _select_out(logits, temperatures, top_k, reference, unit_temperature=True)

    _assert_same_selection(got, want)
    if backend == "reference_topk":
        assert calls == [(prefill_rows, vocab_size)]
    # A Decode-sized call does not fit the Prefill reservation and uses i64.
    calls.clear()
    wide = torch.randn((1024, vocab_size), generator=generator).to(torch.bfloat16).cuda()
    flash_wide = allocate_mcap_topk_scratch(
        1024, vocab_size, top_k, logits.device, select="flash", flash_logits=shared
    )
    _select_out(wide, torch.ones(1024, device="cuda"), top_k, flash_wide, unit_temperature=True)
    assert calls == []


def test_flash_logits_are_shared_and_sized_for_eligible_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # W=64: batches of 1, 2 and 4 use 64, 128 and 256 rows and share one
    # 256-row buffer; B=8 would need 512 rows, beyond FLASH_MAX_ROWS.
    assert flash_logits_rows(64, 8) == 256
    assert flash_logits_rows(64, 2) == 128
    assert flash_logits_rows(32, 8) == 256
    assert flash_logits_rows(4, 8) == 32
    assert flash_logits_rows(0, 8) == 0
    assert flash_logits_rows(64, 0) == 0
    # From width 257 no Decode batch fits in FLASH_MAX_ROWS, but Prefill selects
    # one row per request and still qualifies: B=2 at width 512 is 2 rows.
    assert flash_logits_rows(257, 8) == 8
    assert flash_logits_rows(512, 2) == 2
    assert flash_logits_rows(1024, 8) == 8
    assert flash_logits_rows(1024, 1) == 1

    calls = _flash_backend("reference_topk", monkeypatch)
    vocab_size = 2048
    logits = _flash_parity_logits()
    rows = logits.shape[0]
    shared = allocate_flash_logits(2 * rows, vocab_size, logits.dtype, logits.device)
    assert shared is not None and tuple(shared.shape) == (2 * rows, vocab_size)
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")
    # Two adapters of different row counts reuse the same buffer.
    for batch_rows in (rows, rows // 2):
        scratch = allocate_mcap_topk_scratch(
            batch_rows, vocab_size, 7, logits.device, select="flash", flash_logits=shared
        )
        _select_out(
            logits[:batch_rows], temperatures[:batch_rows], 7, scratch, unit_temperature=True
        )
    assert calls == [(rows, vocab_size), (rows // 2, vocab_size)]

    # Rows beyond the reserved capacity, or a mismatched vocabulary or dtype,
    # use i64 instead of allocating in the request path.
    calls.clear()
    small = allocate_flash_logits(rows - 1, vocab_size, logits.dtype, logits.device)
    mismatched = allocate_flash_logits(rows, vocab_size // 2, logits.dtype, logits.device)
    for buffer in (small, mismatched):
        scratch = allocate_mcap_topk_scratch(
            rows, vocab_size, 7, logits.device, select="flash", flash_logits=buffer
        )
        reference = allocate_mcap_topk_scratch(rows, vocab_size, 7, logits.device, select="i64")
        got = _select_out(logits, temperatures, 7, scratch, unit_temperature=True)
        want = _select_out(logits, temperatures, 7, reference, unit_temperature=True)
        _assert_same_selection(got, want)
    assert calls == []

    # Nothing is reserved for a scratch that is not ``flash``.
    plain = allocate_mcap_topk_scratch(
        rows, vocab_size, 7, logits.device, select="i64", flash_logits=shared
    )
    assert plain.flash_logits is None


def _forbid_flashinfer(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make any FlashInfer capability query or Top-K call fail the test.

    The probe stands in for FlashInfer's JIT module build, which can take
    minutes on a cold cache and is unavailable with FLASHINFER_DISABLE_JIT=1.
    """

    import flashinfer
    import flashinfer.topk as flashinfer_topk

    import vllm_gr.ops.gpu.unconstrained_topk_mcap as mcap

    calls: list[str] = []

    def forbidden(name: str):  # type: ignore[no-untyped-def]
        def probe(*args: object, **kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"FlashInfer {name} must not be reached")

        return probe

    monkeypatch.setattr(flashinfer_topk, "can_implement_filtered_topk", forbidden("capability"))
    monkeypatch.setattr(flashinfer, "top_k", forbidden("top_k"))
    mcap._flash_topk_kernel.cache_clear()
    monkeypatch.setattr(mcap.torch.cuda, "get_device_name", lambda index=None: "Test GPU")
    return calls


def test_flash_topk_excludes_devices_below_filtered_topk_shared_memory_before_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.ops.gpu.unconstrained_topk_mcap as mcap

    calls = _forbid_flashinfer(monkeypatch)
    monkeypatch.setattr(mcap, "_shared_memory_per_sm", lambda index: 100 * 1024)
    try:
        assert mcap._flash_topk_kernel(0) is None
        logits = _flash_parity_logits()
        rows, vocab_size = logits.shape
        assert allocate_flash_logits(rows, vocab_size, logits.dtype, logits.device) is None
    finally:
        mcap._flash_topk_kernel.cache_clear()
    assert calls == []


def test_flash_topk_queries_flashinfer_only_on_devices_with_enough_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.ops.gpu.unconstrained_topk_mcap as mcap

    calls = _forbid_flashinfer(monkeypatch)
    monkeypatch.setattr(mcap, "_shared_memory_per_sm", lambda index: 228 * 1024)
    try:
        # A device that passes the property check reaches FlashInfer's own
        # query; its failure still ends in the i64 fallback, not an exception.
        assert mcap._flash_topk_kernel(0) is None
    finally:
        mcap._flash_topk_kernel.cache_clear()
    assert calls == ["capability"]


def test_flash_topk_falls_back_to_i64_without_touching_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.ops.gpu.unconstrained_topk_mcap as mcap

    calls = _forbid_flashinfer(monkeypatch)
    monkeypatch.setattr(mcap, "_shared_memory_per_sm", lambda index: 100 * 1024)
    logits = _flash_parity_logits()
    rows, vocab_size = logits.shape
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")
    # A buffer that was reserved anyway is left untouched.
    shared = torch.full((rows, vocab_size), float("nan"), dtype=logits.dtype, device="cuda")
    flash = allocate_mcap_topk_scratch(
        rows, vocab_size, 7, logits.device, select="flash", flash_logits=shared
    )
    reference = allocate_mcap_topk_scratch(rows, vocab_size, 7, logits.device, select="i64")
    try:
        got = _select_out(logits, temperatures, 7, flash, unit_temperature=True)
    finally:
        mcap._flash_topk_kernel.cache_clear()
    want = _select_out(logits, temperatures, 7, reference, unit_temperature=True)
    _assert_same_selection(got, want)
    assert torch.isnan(shared).all()
    assert calls == []


def test_flash_stats_kernel_stores_one_zero_and_sanitizes_nonfinite() -> None:
    # Runs on every CUDA device, including those without FilteredTopK.
    from vllm_gr.ops.gpu.unconstrained_topk_mcap import _flash_stats_kernel

    logits = _flash_parity_logits()
    rows, vocab_size = logits.shape
    temperatures = torch.zeros(rows, dtype=torch.float32, device="cuda")
    scratch = allocate_mcap_topk_scratch(rows, vocab_size, 7, logits.device, select="flash")
    sanitized = torch.empty_like(logits)
    tiles = -(-vocab_size // LOCAL_TILE_SIZE)
    _flash_stats_kernel[(rows, tiles)](
        logits,
        temperatures,
        sanitized,
        scratch.partial_max,
        scratch.partial_sum,
        vocab_size,
        logits.stride(0),
        sanitized.stride(0),
        temperatures.numel(),
        tiles,
        BLOCK=LOCAL_TILE_SIZE,
        num_warps=4,
    )
    zeros = sanitized == 0
    assert zeros[3, :2].all() and zeros[4, 5:8].all()
    assert not torch.signbit(sanitized[zeros]).any()
    assert sanitized[1, 0].item() == -float("inf") and sanitized[1, 1].item() == -float("inf")
    finite = torch.isfinite(logits)
    assert torch.equal(sanitized[finite], logits[finite])
