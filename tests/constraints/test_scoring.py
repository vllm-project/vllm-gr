# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm_gr.constraints.scoring import INVALID_LOGIT, score_selected_tokens


def test_selected_scores_use_a_shared_domain_across_parent_rows() -> None:
    # Each row has one legal token.  Legal-set normalization would incorrectly
    # return zero for both and erase the model's strong preference for row 1.
    logits = torch.tensor([[0.0, 10.0], [1.0, -10.0]])
    token_ids = torch.tensor([[0], [0]], dtype=torch.int32)
    valid_mask = torch.ones_like(token_ids, dtype=torch.bool)

    actual = score_selected_tokens(logits, token_ids, valid_mask, torch.ones(1))
    expected = torch.log_softmax(logits, dim=1).gather(1, token_ids.long())

    torch.testing.assert_close(actual, expected)
    assert actual[1, 0] > actual[0, 0]


def test_selected_scores_support_vocab_offset_temperature_and_padding() -> None:
    logits = torch.tensor([[1.0, 2.0, 4.0], [4.0, 2.0, 1.0]])
    original = logits.clone()
    token_ids = torch.tensor([[11, 12], [10, 99]], dtype=torch.int32)
    valid_mask = torch.tensor([[True, True], [True, False]])
    temperatures = torch.tensor([0.0, 2.0])

    actual = score_selected_tokens(
        logits,
        token_ids,
        valid_mask,
        temperatures,
        vocab_offset=10,
    )
    scaled = logits / torch.tensor([[1.0], [2.0]])
    expected = torch.log_softmax(scaled, dim=1).gather(
        1,
        torch.tensor([[1, 2], [0, 0]]),
    )
    expected[1, 1] = INVALID_LOGIT

    torch.testing.assert_close(actual, expected)
    assert torch.equal(logits, original)
