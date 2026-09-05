# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk


def test_rec_constrained_topk_step_zero_filters_and_normalizes() -> None:
    logits = torch.tensor([[0.0, 3.0, 2.0, 1.0]])
    tokens, logprobs = rec_constrained_topk(
        logits,
        torch.empty((1, 0), dtype=torch.int32),
        torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.zeros(5, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.ones(1),
        current_step=0,
        top_k=2,
    )
    assert tokens.tolist() == [[2, 3]]
    expected = torch.log_softmax(logits, dim=1)[0, torch.tensor([2, 3])]
    torch.testing.assert_close(logprobs[0], expected)


def test_rec_constrained_topk_step_one_uses_prefix_csr() -> None:
    logits = torch.tensor([[0.0, 4.0, 2.0, 3.0]])
    tokens, _ = rec_constrained_topk(
        logits,
        torch.tensor([[1, 0]], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.tensor([0, 0, 2, 3, 3], dtype=torch.int32),
        torch.tensor([1, 3, 0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.ones(1),
        current_step=1,
        top_k=2,
    )
    assert tokens.tolist() == [[1, 3]]


def test_rec_constrained_topk_step_two_uses_pair_table() -> None:
    logits = torch.tensor([[0.0, 4.0, 2.0, 3.0]])
    tokens, _ = rec_constrained_topk(
        logits,
        torch.tensor([[1, 2, 0]], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.tensor([1 * 4 + 2], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([1, 3], dtype=torch.int32),
        torch.ones(1),
        current_step=2,
        top_k=2,
    )
    assert tokens.tolist() == [[1, 3]]


def test_rec_constrained_topk_bucketed_path_accepts_scalar_temperature() -> None:
    logits = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    tokens, probs = rec_constrained_topk(
        logits,
        torch.tensor([[1, 0]], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.tensor([0, 0, 4, 8], dtype=torch.int32),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.tensor(1.0),
        current_step=1,
        top_k=1,
        max_prefix1_degree=4,
    )
    assert tokens.shape == (1, 1)
    assert probs.shape == (1, 1)
