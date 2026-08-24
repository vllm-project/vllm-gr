# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_gr.beam_search_decision_utils import (
    extract_and_dedup_flat_logprobs,
    extract_and_dedup_segment,
)
from vllm_gr.ops.sliced_logits import (
    compute_sliced_logits,
    get_rank_vocab_bounds,
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

    # Test upper-range slice tokens where tid >= offset (e.g. offset=20000, slice_size=24000, tid=21000 -> 41000)
    seg_ids_upper, _, _, _ = extract_and_dedup_segment(
        raw_token_ids=[21000, 41000],
        raw_logprobs=[-0.2, -0.8],
        raw_ranks=[0, 1],
        raw_decoded=["a", "b"],
        start_offset=0,
        end_offset=2,
        logprobs_num=2,
    )
    assert list(seg_ids_upper) == [41000, 41000]

    # Test slice_size > offset scenario (e.g. offset=1000, slice_size=1500, tid=1200 -> 2200)
    set_vocab_slice_config(offset=1000, slice_size=1500)
    seg_ids_small_offset, _, _, _ = extract_and_dedup_segment(
        raw_token_ids=[1200, 2200],
        raw_logprobs=[-0.2, -0.8],
        raw_ranks=[0, 1],
        raw_decoded=["a", "b"],
        start_offset=0,
        end_offset=2,
        logprobs_num=2,
    )
    assert list(seg_ids_small_offset) == [2200, 2200]

    # Reset config
    set_vocab_slice_config(None, None)
    assert not is_vocab_slice_enabled()


class MockTPHead(nn.Module):  # type: ignore[misc]
    """Simulates a shard of lm_head.weight for a specific TP rank."""

    def __init__(self, full_weight: torch.Tensor, rank: int, tp_size: int):
        super().__init__()
        vocab_size = full_weight.shape[0]
        shard_size = vocab_size // tp_size
        self.rank = rank
        self.tp_size = tp_size
        self.vocab_start_index = rank * shard_size
        self.vocab_end_index = (rank + 1) * shard_size
        self.weight = nn.Parameter(
            full_weight[self.vocab_start_index : self.vocab_end_index].clone(),
            requires_grad=False,
        )


def test_get_rank_vocab_bounds() -> None:
    full_w = torch.randn(1000, 64)
    head_rank0 = MockTPHead(full_w, rank=0, tp_size=2)
    head_rank1 = MockTPHead(full_w, rank=1, tp_size=2)

    assert get_rank_vocab_bounds(head_rank0) == (0, 500)
    assert get_rank_vocab_bounds(head_rank1) == (500, 1000)


def test_tp_sliced_logits_equivalence_tp2() -> None:
    """Verify that sliced logits under TP=2 match un-sliced logits across ranks."""
    torch.manual_seed(42)
    vocab_size = 10000
    embed_dim = 32
    batch_size = 4
    hidden_states = torch.randn(batch_size, embed_dim)
    full_weight = torch.randn(vocab_size, embed_dim)

    offset = 2000
    slice_size = 3000

    # Un-sliced reference logits for the slice [2000:5000]
    full_logits = F.linear(hidden_states, full_weight)
    ref_sliced_logits = full_logits[:, offset : offset + slice_size]

    # Rank 0 covers [0:5000], requested slice [2000:5000] -> rank 0 local [2000:5000]
    rank0_head = MockTPHead(full_weight, rank=0, tp_size=2)
    rank0_logits = compute_sliced_logits(
        hidden_states, rank0_head, offset=offset, slice_size=slice_size
    )

    # Rank 1 covers [5000:10000], requested slice [2000:5000] -> rank 1 has no overlap
    rank1_head = MockTPHead(full_weight, rank=1, tp_size=2)
    rank1_logits = compute_sliced_logits(
        hidden_states, rank1_head, offset=offset, slice_size=slice_size
    )

    # Rank 0 should compute exact logits for the slice [2000:5000]
    assert torch.allclose(rank0_logits, ref_sliced_logits, atol=1e-5, rtol=1e-5)
    # Rank 1 overlap is empty
    assert rank1_logits.shape[-1] == 0


def test_tp_sliced_logits_spanning_multiple_ranks() -> None:
    """Verify slice spanning across Rank 0 and Rank 1 under TP=2."""
    torch.manual_seed(123)
    vocab_size = 10000
    embed_dim = 32
    batch_size = 2
    hidden_states = torch.randn(batch_size, embed_dim)
    full_weight = torch.randn(vocab_size, embed_dim)

    # Slice [4000:6000] spans Rank 0 ([0:5000]) and Rank 1 ([5000:10000])
    offset = 4000
    slice_size = 2000
    full_logits = F.linear(hidden_states, full_weight)
    ref_sliced_logits = full_logits[:, offset : offset + slice_size]

    rank0_head = MockTPHead(full_weight, rank=0, tp_size=2)
    rank0_logits = compute_sliced_logits(
        hidden_states, rank0_head, offset=offset, slice_size=slice_size
    )

    rank1_head = MockTPHead(full_weight, rank=1, tp_size=2)
    rank1_logits = compute_sliced_logits(
        hidden_states, rank1_head, offset=offset, slice_size=slice_size
    )

    # Rank 0 computes [4000:5000] (1000 logits), Rank 1 computes [5000:6000] (1000 logits)
    assert rank0_logits.shape[-1] == 1000
    assert rank1_logits.shape[-1] == 1000

    gathered_logits = torch.cat([rank0_logits, rank1_logits], dim=-1)
    assert torch.allclose(gathered_logits, ref_sliced_logits, atol=1e-5, rtol=1e-5)


def test_quantized_lm_head_fallback() -> None:
    """Verify that quantized lm_head uses safe logits post-slicing fallback."""
    torch.manual_seed(99)
    vocab_size = 5000
    embed_dim = 16
    batch_size = 2
    hidden_states = torch.randn(batch_size, embed_dim)
    full_weight = torch.randn(vocab_size, embed_dim)

    class DummyQuantMethod:
        def apply(self, layer: Any, x: torch.Tensor, bias: Any = None) -> torch.Tensor:
            return F.linear(x, layer.weight, bias)

    class MockQuantizedHead(nn.Module):  # type: ignore[misc]
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.quant_method = DummyQuantMethod()

    quant_head = MockQuantizedHead(full_weight)
    offset = 1000
    slice_size = 500

    full_logits = F.linear(hidden_states, full_weight)
    ref_sliced = full_logits[:, offset : offset + slice_size]

    sliced_logits = compute_sliced_logits(
        hidden_states, quant_head, offset=offset, slice_size=slice_size
    )

    assert sliced_logits.shape == (batch_size, slice_size)
    assert torch.allclose(sliced_logits, ref_sliced, atol=1e-5, rtol=1e-5)
    # Ensure pre-sliced weight proxy was NOT created for quantized head
    assert not hasattr(quant_head, "_sliced_weight")
