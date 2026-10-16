# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression cases for full-vocabulary candidate selection."""

import pytest
import torch

from vllm_gr.ops.cuda.unconstrained_topk import (
    allocate_unconstrained_topk_scratch,
    unconstrained_topk_out,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("vocab_size,top_k", [(17, 1), (17, 32), (1031, 3), (5123, 128)])  # type: ignore[misc]
def test_unconstrained_topk_matches_full_vocab_reference(vocab_size: int, top_k: int) -> None:
    generator = torch.Generator().manual_seed(408)
    logits_cpu = torch.randn((2, vocab_size), generator=generator)
    logits_cpu[0].zero_()  # Every score ties; the lower token ID must win.
    logits = logits_cpu.cuda()
    temperatures = torch.tensor([0.0, 0.75], device="cuda")
    scratch = allocate_unconstrained_topk_scratch(2, vocab_size, top_k, logits.device)
    token_ids = torch.empty((2, top_k), dtype=torch.int32, device="cuda")
    logprobs = torch.empty((2, top_k), dtype=torch.float32, device="cuda")
    valid = torch.empty((2, top_k), dtype=torch.bool, device="cuda")
    counts = torch.empty((2,), dtype=torch.int32, device="cuda")

    unconstrained_topk_out(logits, temperatures, top_k, scratch, token_ids, logprobs, valid, counts)

    for row, temperature in enumerate((1.0, 0.75)):
        scores = logits_cpu[row] / temperature
        expected_ids = sorted(range(vocab_size), key=lambda index: (-scores[index].item(), index))[
            :top_k
        ]
        expected_logprobs = torch.log_softmax(scores, dim=0)[expected_ids]
        count = min(vocab_size, top_k)
        assert token_ids[row, :count].tolist() == expected_ids
        torch.testing.assert_close(
            logprobs[row, :count].cpu(), expected_logprobs, rtol=1e-4, atol=1e-4
        )
        assert valid[row, :count].all().item()
        assert not valid[row, count:].any().item()
    assert counts.tolist() == [min(vocab_size, top_k)] * 2


def test_unconstrained_topk_accepts_padded_rows_and_empty_tiles() -> None:
    storage = torch.empty((2, 2112), dtype=torch.float32, device="cuda")
    logits = storage[:, :2048]
    logits[0, :1024].zero_()
    logits[0, 1024:].fill_(-float("inf"))
    logits[1].fill_(-float("inf"))
    logits[1, 0] = float("nan")
    temperatures = torch.ones(2, dtype=torch.float32, device="cuda")
    scratch = allocate_unconstrained_topk_scratch(2, 2048, 3, logits.device)
    token_ids = torch.empty((2, 3), dtype=torch.int32, device="cuda")
    logprobs = torch.empty((2, 3), dtype=torch.float32, device="cuda")
    valid = torch.empty((2, 3), dtype=torch.bool, device="cuda")
    counts = torch.empty((2,), dtype=torch.int32, device="cuda")

    unconstrained_topk_out(logits, temperatures, 3, scratch, token_ids, logprobs, valid, counts)

    assert logits.stride() == (2112, 1)
    assert token_ids[0].tolist() == [0, 1, 2]
    torch.testing.assert_close(
        logprobs[0], torch.log_softmax(logits[0], dim=0)[:3], rtol=1e-4, atol=1e-4
    )
    assert valid[0].all().item()
    assert counts.tolist() == [3, 0]
    assert not valid[1].any().item()


def test_unconstrained_topk_counts_only_finite_candidates() -> None:
    logits = torch.full((3, 2048), -float("inf"), device="cuda")
    logits[0, 10] = 0.0
    logits[1, 1] = 1.0
    logits[1, 2] = 0.0
    logits[1, 3] = float("nan")
    logits[2, 4] = float("nan")
    top_k = 4
    scratch = allocate_unconstrained_topk_scratch(3, 2048, top_k, logits.device)
    token_ids = torch.empty((3, top_k), dtype=torch.int32, device="cuda")
    logprobs = torch.empty((3, top_k), dtype=torch.float32, device="cuda")
    valid = torch.empty((3, top_k), dtype=torch.bool, device="cuda")
    counts = torch.empty((3,), dtype=torch.int32, device="cuda")

    unconstrained_topk_out(
        logits,
        torch.ones(3, device="cuda"),
        top_k,
        scratch,
        token_ids,
        logprobs,
        valid,
        counts,
    )

    assert counts.tolist() == [1, 2, 0]
    assert valid.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, False, False],
    ]
    assert token_ids[0, 0].item() == 10
    assert token_ids[1, :2].tolist() == [1, 2]
    torch.testing.assert_close(logprobs[0, 0].cpu(), torch.tensor(0.0))
    torch.testing.assert_close(
        logprobs[1, :2].cpu(), torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)
    )
    assert torch.isneginf(logprobs[~valid]).all().item()
