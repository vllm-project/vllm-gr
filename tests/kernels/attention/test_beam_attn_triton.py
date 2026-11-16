# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Parity guards for the CUDA Triton suffix-KV gather.

``extract_suffix_kv`` serves the standard (non short-suffix) Beam decode path.
It is compared element-wise with the pure-PyTorch reference used by the NPU
recompute path, on both a contiguous paged cache and the K/V-interleaved
strided views used by the vllm-ascend 0.30 pool layout.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    extract_suffix_kv_pytorch,
)

pytestmark = [
    pytest.mark.gpu_test,
    pytest.mark.l2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _assert_gather_matches_reference(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
) -> None:
    from vllm_gr.v1.attention.backends.beam_attn_triton import extract_suffix_kv

    out_k, out_v = extract_suffix_kv(k_cache, v_cache, slots, slots.numel(), block_size)
    # Production metadata supplies int32 slots; indexing only needs the same indices.
    ref_k, ref_v = extract_suffix_kv_pytorch(k_cache, v_cache, slots.long())

    assert out_k.shape == (slots.numel(), k_cache.shape[2], k_cache.shape[3])
    assert out_k.dtype == k_cache.dtype
    assert out_v.dtype == v_cache.dtype
    assert out_k.is_contiguous()
    assert out_v.is_contiguous()
    assert torch.equal(out_k, ref_k)
    assert torch.equal(out_v, ref_v)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("heads", "head_dim", "block_size", "dtype"),
    [
        (8, 64, 16, torch.float32),
        # Non-power-of-two head_dim exercises the kernel's BLOCK_SIZE_D mask.
        (4, 80, 16, torch.bfloat16),
    ],
)
def test_contiguous_paged_gather_matches_reference(
    heads: int, head_dim: int, block_size: int, dtype: torch.dtype
) -> None:
    num_blocks = 5
    torch.manual_seed(1234)
    k_cache = torch.randn(num_blocks, block_size, heads, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    total_slots = num_blocks * block_size
    slots = torch.tensor(
        [
            0,
            total_slots - 1,
            block_size + 1,
            2 * block_size - 1,
            total_slots - 2,
            2,
            2,  # Duplicate gathers must return independent copies, not aliases.
            1,
        ],
        dtype=torch.int32,
        device="cuda",
    )

    _assert_gather_matches_reference(k_cache, v_cache, slots, block_size)


def test_interleaved_layout_gather_matches_reference() -> None:
    num_blocks, block_size, heads, head_dim = 4, 8, 2, 64
    torch.manual_seed(1235)
    pool = torch.randn(
        num_blocks, 2, block_size, heads, head_dim, device="cuda", dtype=torch.float32
    )
    k_cache = pool[:, 0]
    v_cache = pool[:, 1]
    assert not k_cache.is_contiguous()
    slots = torch.tensor([0, 9, 17, 31, 5, 27], dtype=torch.int32, device="cuda")

    _assert_gather_matches_reference(k_cache, v_cache, slots, block_size)


# First and last physical slot.
@pytest.mark.parametrize("slot", [0, 47])  # type: ignore[misc]
def test_single_token_boundaries(slot: int) -> None:
    from vllm_gr.v1.attention.backends.beam_attn_triton import extract_suffix_kv

    num_blocks, block_size, heads, head_dim = 3, 16, 4, 64
    torch.manual_seed(1236)
    k_cache = torch.randn(num_blocks, block_size, heads, head_dim, device="cuda")
    v_cache = torch.randn_like(k_cache)
    slots = torch.tensor([slot], dtype=torch.int32, device="cuda")

    out_k, out_v = extract_suffix_kv(k_cache, v_cache, slots, 1, block_size)

    flat_k = k_cache.view(-1, heads, head_dim)
    flat_v = v_cache.view(-1, heads, head_dim)
    assert torch.equal(out_k[0], flat_k[slot])
    assert torch.equal(out_v[0], flat_v[slot])
