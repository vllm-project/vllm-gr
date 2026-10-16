# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm_gr.v1.worker.full_vocab import full_vocab_topk


def test_normalizes_over_entire_vocabulary_and_returns_global_ids() -> None:
    # The winner is outside the hypothetical SID interval [2, 4). The four
    # omitted tokens also carry probability mass and must affect its score.
    logits = torch.tensor([[4.0, 1.0, 3.0, 2.0, 0.0, -1.0]])
    candidates = full_vocab_topk(logits, 2)
    denominator = math.log(sum(math.exp(value) for value in logits[0].tolist()))
    assert candidates.token_ids.tolist() == [[0, 2]]
    torch.testing.assert_close(
        candidates.logprobs, torch.tensor([[4.0 - denominator, 3.0 - denominator]])
    )
    assert candidates.valid_mask.tolist() == [[True, True]]
    assert candidates.valid_counts.tolist() == [2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_scores_remain_fp32_for_each_model_dtype(dtype: torch.dtype) -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [-2.0, 0.0, 1.0]], dtype=dtype)
    candidates = full_vocab_topk(logits, 2)
    expected = logits.float().log_softmax(-1)
    assert candidates.token_ids.dtype == torch.int32
    assert candidates.logprobs.dtype == torch.float32
    assert candidates.valid_mask.dtype == torch.bool
    assert candidates.valid_counts.dtype == torch.int32
    torch.testing.assert_close(
        candidates.logprobs, expected.gather(-1, candidates.token_ids.long())
    )


def test_small_vocabulary_is_padded_without_fabricating_candidates() -> None:
    candidates = full_vocab_topk(torch.tensor([[1.0, 2.0]]), 4)
    assert candidates.token_ids.shape == (1, 4)
    assert candidates.token_ids[0, :2].tolist() == [1, 0]
    assert candidates.valid_mask.tolist() == [[True, True, False, False]]
    assert candidates.valid_counts.tolist() == [2]
    assert torch.isneginf(candidates.logprobs[0, 2:]).all()


def test_invalid_rows_and_negative_infinity_tokens_never_become_valid() -> None:
    candidates = full_vocab_topk(
        torch.tensor([[0.0, -math.inf, -math.inf], [-math.inf, -math.inf, -math.inf]]), 3
    )
    assert candidates.valid_counts.tolist() == [1, 0]
    assert candidates.token_ids[0, 0].item() == 0
    assert candidates.logprobs[0, 0].item() == 0
    assert not torch.isnan(candidates.logprobs).any()
    assert torch.isneginf(candidates.logprobs[~candidates.valid_mask]).all()


def test_maximum_beam_width_can_request_512_candidates() -> None:
    candidates = full_vocab_topk(torch.arange(1024, dtype=torch.float32).reshape(1, -1) / 100, 512)
    assert candidates.token_ids.shape == (1, 512)
    assert candidates.token_ids[0].tolist() == list(range(1023, 511, -1))
    assert candidates.valid_counts.tolist() == [512]


@pytest.mark.parametrize("top_k", [0, -1, 513, True, 1.5])
def test_rejects_invalid_candidate_width(top_k: object) -> None:
    with pytest.raises(ValueError, match="candidate width"):
        full_vocab_topk(torch.ones((1, 3)), top_k)


def test_rejects_invalid_logits_shape_and_dtype() -> None:
    for shape in ((3,), (0, 3), (2, 0), (1, 2, 3)):
        with pytest.raises(ValueError, match="shape"):
            full_vocab_topk(torch.empty(shape), 2)
    with pytest.raises(TypeError, match="FP16"):
        full_vocab_topk(torch.ones((1, 3), dtype=torch.int32), 2)
