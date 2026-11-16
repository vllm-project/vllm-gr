# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import pytest
import torch

from vllm_gr.constraints.scoring import INVALID_LOGIT
from vllm_gr.ops.rec_constrained_topk import (
    _candidate_set_logprobs,
    rec_constrained_topk,
)


def test_rec_constrained_topk_npu_fallback_scales_by_temperature() -> None:
    """The NPU Torch fallback must apply the effective temperature before
    normalizing over the legal candidate set; otherwise temperature != {0, 1}
    returns different logprobs from the fused NPU operator and the pre-PR path.
    """
    candidate_logits = torch.tensor(
        [
            [10.0, 9.0, 5.0, 1.0, float("-inf"), float("-inf")],
            [2.0, 8.0, 4.0, 6.0, float("-inf"), float("-inf")],
        ]
    )
    top_indices = torch.tensor([[0, 1, 2, 3], [1, 3, 2, 0]], dtype=torch.long)
    top_valid = torch.ones(2, 4, dtype=torch.bool)

    for temperatures, temps_view in (
        (torch.tensor(2.0), torch.tensor(2.0)),
        (torch.tensor([2.0]), torch.tensor(2.0)),
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])),
    ):
        probs = _candidate_set_logprobs(candidate_logits, top_indices, top_valid, temperatures)
        expected = torch.log_softmax(candidate_logits / temps_view.reshape(-1, 1), dim=1).gather(
            1, top_indices
        )
        torch.testing.assert_close(probs, expected)

    # Zero temperature behaves like unity, and invalid slots keep the sentinel.
    probs_unit = _candidate_set_logprobs(
        candidate_logits, top_indices, top_valid, torch.tensor(0.0)
    )
    expected_unit = torch.log_softmax(candidate_logits, dim=1).gather(1, top_indices)
    torch.testing.assert_close(probs_unit, expected_unit)

    invalid_indices = torch.tensor([[0, 1, 2, 4], [1, 3, 2, 5]], dtype=torch.long)
    invalid_mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    probs_invalid = _candidate_set_logprobs(
        candidate_logits, invalid_indices, invalid_mask, torch.tensor(2.0)
    )
    torch.testing.assert_close(probs_invalid[:, 3], torch.full((2,), INVALID_LOGIT))


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


def test_rec_constrained_topk_step_zero_drops_nan_candidates() -> None:
    # A NaN candidate must sort last (mirroring the fused CUDA kernel), not
    # be treated as the largest value and win selection.  The NaN candidate
    # stays in the shared window, so the row's logprobs surface the engine's
    # protocol error (NaN -> beam_device_error).
    logits = torch.tensor([[0.0, 3.0, float("nan"), 1.0]])
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
    # Candidate token 2 is NaN; {3, 0} win selection with finite candidates.
    assert tokens.tolist() == [[3, 0]]
    assert torch.isnan(logprobs).all()


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


pytestmark = pytest.mark.cpu_test
