# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from vllm_gr.ops.sliced_logits import (
    compute_sliced_logits,
    set_vocab_slice_config,
    get_vocab_slice_config,
    is_vocab_slice_enabled,
    init_vocab_slice_config_from_env,
)
from vllm_gr.logprobs import extract_and_dedup_flat_logprobs
from vllm_gr.entrypoints.openai.serving_engine import _extract_and_dedup_segment


class MockLMHead(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(vocab_size))
        else:
            self.bias = None


def test_sliced_logits_equivalence():
    batch_size = 4
    hidden_dim = 128
    full_vocab_size = 170000
    offset = 20000
    slice_size = 24000

    torch.manual_seed(42)
    lm_head = MockLMHead(full_vocab_size, hidden_dim, bias=True)
    hidden_states = torch.randn(batch_size, hidden_dim)

    # 1. Compute full logits
    full_logits = torch.matmul(hidden_states, lm_head.weight.T) + lm_head.bias
    expected_slice_logits = full_logits[:, offset : offset + slice_size]

    # 2. Compute sliced logits
    sliced_logits = compute_sliced_logits(
        hidden_states=hidden_states,
        lm_head=lm_head,
        embedding_bias=lm_head.bias,
        offset=offset,
        slice_size=slice_size,
        quant_group_size=128,
    )

    assert sliced_logits.shape == (batch_size, slice_size)
    assert torch.allclose(sliced_logits, expected_slice_logits, atol=1e-5, rtol=1e-5)


def test_quantization_group_alignment():
    batch_size = 2
    hidden_dim = 64
    full_vocab_size = 50000
    unaligned_offset = 20013
    slice_size = 24000
    group_size = 128

    lm_head = MockLMHead(full_vocab_size, hidden_dim, bias=False)
    hidden_states = torch.randn(batch_size, hidden_dim)

    full_logits = torch.matmul(hidden_states, lm_head.weight.T)
    expected_slice_logits = full_logits[:, unaligned_offset : unaligned_offset + slice_size]

    sliced_logits = compute_sliced_logits(
        hidden_states=hidden_states,
        lm_head=lm_head,
        embedding_bias=None,
        offset=unaligned_offset,
        slice_size=slice_size,
        quant_group_size=group_size,
    )

    assert sliced_logits.shape == (batch_size, slice_size)
    assert torch.allclose(sliced_logits, expected_slice_logits, atol=1e-5, rtol=1e-5)


def test_token_id_remapping():
    set_vocab_slice_config(offset=20000, slice_size=24000)
    assert is_vocab_slice_enabled()

    class MockFlatLogprobs:
        def __init__(self, token_ids, logprobs):
            self.start_indices = [0]
            self.end_indices = [len(token_ids)]
            self.token_ids = token_ids
            self.logprobs = logprobs
            self.ranks = [0] * len(token_ids)
            self.decoded_tokens = [""] * len(token_ids)

    flat = MockFlatLogprobs(token_ids=[0, 1, 2], logprobs=[-0.1, -0.5, -1.0])
    remapped_ids, _, _, _ = extract_and_dedup_flat_logprobs(flat, logprobs_num=3)
    assert remapped_ids == [20000, 20001, 20002]

    # Test _extract_and_dedup_segment
    seg_ids, _, _, _ = _extract_and_dedup_segment(
        raw_token_ids=[5, 10],
        raw_logprobs=[-0.2, -0.8],
        raw_ranks=[0, 1],
        raw_decoded=["a", "b"],
        start_offset=0,
        end_offset=2,
        logprobs_num=2,
    )
    assert seg_ids == [20005, 20010]

    # Reset config
    set_vocab_slice_config(None, None)
    assert not is_vocab_slice_enabled()


def test_env_vars_config(monkeypatch):
    set_vocab_slice_config(None, None)
    monkeypatch.setenv("VLLM_GR_VOCAB_OFFSET", "20000")
    monkeypatch.setenv("VLLM_GR_VOCAB_SLICE_SIZE", "24576")
    init_vocab_slice_config_from_env()

    assert is_vocab_slice_enabled()
    offset, slice_size, quant_g = get_vocab_slice_config()
    assert offset == 20000
    assert slice_size == 24576

    monkeypatch.delenv("VLLM_GR_VOCAB_OFFSET")
    monkeypatch.delenv("VLLM_GR_VOCAB_SLICE_SIZE")
    init_vocab_slice_config_from_env()
    assert not is_vocab_slice_enabled()
