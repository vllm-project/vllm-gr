# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference guards for the bounded-retention Top-K and its candidate merge.

Both pipelines must reproduce the stable ``(-score, index)`` order of the CPU
oracle bit for bit, and a vocabulary-sliced scan must match the single-shot one.
"""

import pytest
import torch

from vllm_gr.ops.cuda.beam_selection import (
    _hierarchical_topk_out,
    allocate_beam_selection_scratch,
)
from vllm_gr.ops.cuda.unconstrained_topk import (
    allocate_unconstrained_topk_scratch,
    unconstrained_topk_out,
)
from vllm_gr.ops.cuda.unconstrained_topk_mcap import (
    LOCAL_KEEP_MAX,
    allocate_mcap_topk_scratch,
    mcap_topk_tiling,
    merge_chunk_width,
    unconstrained_topk_mcap_out,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _run_both(
    logits: torch.Tensor, temperatures: torch.Tensor, top_k: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    rows = logits.shape[0]
    legacy_scratch = allocate_unconstrained_topk_scratch(
        rows, logits.shape[1], top_k, logits.device
    )
    new_scratch = allocate_mcap_topk_scratch(rows, logits.shape[1], top_k, logits.device)
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
    for row, expected in enumerate(ids_expected):
        assert new_ids[row, : masks[row]].tolist() == expected.tolist()
        torch.testing.assert_close(
            new_lp[row, : masks[row]].cpu(), lp_expected[row], rtol=1e-4, atol=1e-4
        )


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


def test_mcap_accepts_padded_rows_empty_tiles_and_nan() -> None:
    storage = torch.empty((2, 2112), dtype=torch.float32, device="cuda")
    logits = storage[:, :2048]
    logits[0, :1024].zero_()
    logits[0, 1024:].fill_(-float("inf"))
    logits[1].fill_(-float("inf"))
    logits[1, 0] = float("nan")
    temperatures = torch.ones(2, dtype=torch.float32, device="cuda")

    (ref_ids, ref_lp, _, ref_counts), (new_ids, new_lp, new_valid, new_counts) = _run_both(
        logits, temperatures, 3
    )

    assert torch.equal(new_ids, ref_ids)
    assert torch.equal(new_lp, ref_lp)
    assert torch.equal(new_counts, ref_counts)
    assert logits.stride() == (2112, 1)
    assert new_ids[0].tolist() == [0, 1, 2]
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
    assert new_counts.tolist() == [1, 2, 0]
    assert new_ids[0, 0].item() == 10
    assert new_ids[1, :2].tolist() == [1, 2]
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

    monkeypatch.setenv("VLLM_GR_TOPK_SELECT", "i32")
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
        ) = _run_both(logits, torch.ones(2, device="cuda"), top_k)
        assert torch.equal(new_ids, ref_ids)
        assert torch.equal(new_lp, ref_lp)
        assert torch.equal(new_valid, ref_valid)
        assert torch.equal(new_counts, ref_counts)


@pytest.mark.parametrize("top_k", [1, 2])  # type: ignore[misc]
def test_mcap_i32_small_k_repairs_collision_at_cutoff(
    monkeypatch: pytest.MonkeyPatch, top_k: int
) -> None:
    """An i32 prefix collision must still return the exact Top-K when keep == top_k."""

    monkeypatch.setenv("VLLM_GR_TOPK_SELECT", "i32")
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
        logits, temperatures, top_k
    )

    assert new_ids.cpu().tolist() == [[2, 0][:top_k]] * rows
    assert torch.equal(new_ids, ref_ids)
    assert torch.equal(new_lp, ref_lp)
    assert torch.equal(new_valid, ref_valid)
    assert torch.equal(new_counts, ref_counts)


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
