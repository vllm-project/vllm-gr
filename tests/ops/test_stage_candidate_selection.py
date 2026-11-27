# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compare stage candidate scoring with a masked full-vocabulary oracle."""

import pytest
import torch

from vllm_gr.ops.unconstrained_topk_torch import select_stage_candidates

pytestmark = [pytest.mark.cpu_test, pytest.mark.l2]


def test_stage_logprobs_do_not_depend_on_illegal_logits_or_candidate_width() -> None:
    logits = torch.tensor([[90, 2, 1, 100, -3, -80], [-20, 0, 1, 80, 4, -60]])
    ids = torch.tensor([1, 2, 4])
    temperatures = torch.tensor([1.0, 0.5])
    logits = logits.to(torch.bfloat16)
    masked = torch.full_like(logits, -torch.inf)
    masked[:, ids] = logits[:, ids]
    expected = (masked.float() / temperatures[:, None]).log_softmax(-1)

    for width in (1, 2):
        result = select_stage_candidates(logits, temperatures, width, ids)
        expected_scores, expected_token_ids = expected.topk(width)
        assert result.valid_mask.all().item()
        torch.testing.assert_close(result.token_ids, expected_token_ids.to(torch.int32))
        torch.testing.assert_close(result.logprobs, expected_scores)


def test_stage_ties_and_invalid_slots() -> None:
    logits = torch.tensor([[20.0, 0.0, 30.0, 0.0, torch.nan]])
    result = select_stage_candidates(logits, torch.tensor([0.0]), 4, torch.tensor([1, 3, 4]))
    assert result.valid_mask.tolist() == [[True, True, False, False]]
    assert result.token_ids[result.valid_mask].tolist() == [1, 3]
    torch.testing.assert_close(result.logprobs[result.valid_mask], torch.zeros(2).log_softmax(0))
    assert torch.isneginf(result.logprobs[~result.valid_mask]).all().item()
