# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import torch
import torch.nn as nn

from vllm_gr.entrypoints.openai.serving_engine import _extract_and_dedup_segment
from vllm_gr.entrypoints.openai.serving_models import Catalog
from vllm_gr.logprobs import extract_and_dedup_flat_logprobs
from vllm_gr.ops.sliced_logits import (
    compute_sliced_logits,
    get_vocab_slice_config,
    is_vocab_slice_enabled,
    set_vocab_slice_config,
)


class MockLMHead(nn.Module):  # type: ignore[misc]
    def __init__(self, vocab_size: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(vocab_size))
        else:
            self.bias = None


def test_sliced_logits_equivalence() -> None:
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


def test_quantization_group_alignment() -> None:
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


def test_token_id_remapping() -> None:
    set_vocab_slice_config(offset=20000, slice_size=24000)
    assert is_vocab_slice_enabled()

    class MockFlatLogprobs:
        def __init__(self, token_ids: list[int], logprobs: list[float]) -> None:
            self.start_indices = [0]
            self.end_indices = [len(token_ids)]
            self.token_ids = token_ids
            self.logprobs = logprobs
            self.ranks = [0] * len(token_ids)
            self.decoded_tokens = [""] * len(token_ids)

    flat = MockFlatLogprobs(token_ids=[0, 1, 2], logprobs=[-0.1, -0.5, -1.0])
    remapped_ids, _, _, _ = extract_and_dedup_flat_logprobs(flat, logprobs_num=3)
    assert list(remapped_ids) == [20000, 20001, 20002]

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
    assert list(seg_ids) == [20005, 20010]

    # Reset config
    set_vocab_slice_config(None, None)
    assert not is_vocab_slice_enabled()


def test_catalog_slice_config(tmp_path: Any) -> None:
    set_vocab_slice_config(None, None)
    assert not is_vocab_slice_enabled()

    class DummyTokenizer:
        def convert_tokens_to_ids(self, tokens: Any) -> Any:
            mapping = {
                "<|sid_begin|>": 1,
                "<s_a_100>": 1000,
                "<s_b_200>": 1200,
                "<s_c_500>": 2499,
                "<|sid_end|>": 2,
            }
            if isinstance(tokens, list):
                return [mapping.get(t, None) for t in tokens]
            return mapping.get(tokens, None)

    catalog_data = [["<|sid_begin|>", "<s_a_100>", "<s_b_200>", "<s_c_500>", "<|sid_end|>"]]
    catalog_file = tmp_path / "test_catalog.json"
    catalog_file.write_text(json.dumps(catalog_data))

    cat = Catalog(DummyTokenizer())
    cat.load(str(catalog_file))

    assert is_vocab_slice_enabled()
    assert cat.vocab_offset == 1000
    assert cat.vocab_slice_size == 1500  # 2499 - 1000 + 1
    offset, slice_size, _ = get_vocab_slice_config()
    assert offset == 1000
    assert slice_size == 1500

    set_vocab_slice_config(None, None)
    assert not is_vocab_slice_enabled()
