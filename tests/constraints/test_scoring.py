# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import pytest
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


def test_inf_window_surfaces_error_signal_in_shared_reference() -> None:
    # A +inf window element gives a +inf logsumexp, which the shared reference
    # turns into NaN logprobs (matching the fused CUDA kernel): the row
    # surfaces the engine's protocol error (beam_device_error) instead of
    # yielding garbage finite values.
    logits = torch.tensor([[0.0, 1.0, float("inf")]])
    finite_selected = score_selected_tokens(
        logits,
        torch.tensor([[1]], dtype=torch.int32),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1),
    )
    assert torch.isnan(finite_selected).all()

    inf_selected = score_selected_tokens(
        logits,
        torch.tensor([[2]], dtype=torch.int32),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1),
    )
    assert torch.isnan(inf_selected).all()


def test_nan_window_propagates_error_signal() -> None:
    # NaN logits intentionally stay non-finite: a NaN anywhere in the window
    # makes the row's logsumexp NaN, surfacing the engine's protocol error
    # (beam_device_error) exactly as +inf logits do.  This is the fused CUDA
    # kernel behavior and the torch reference must match it.
    all_nan = torch.full((2, 5), float("nan"))
    token_ids = torch.tensor([[1], [3]], dtype=torch.int32)
    valid_mask = torch.tensor([[True], [True]], dtype=torch.bool)
    actual = score_selected_tokens(all_nan, token_ids, valid_mask, torch.zeros(2))
    assert torch.isnan(actual).all()

    mixed = torch.tensor([[0.0, float("nan"), 1.0, 2.0, 3.0]])
    actual = score_selected_tokens(
        mixed,
        torch.tensor([[2]], dtype=torch.int32),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1),
    )
    assert torch.isnan(actual).all()


pytestmark = pytest.mark.cpu_test
