# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""ROCm Beam cascade: paged prefix + single-query GQA suffix + LSE merge."""

import math

import pytest
import torch

pytestmark = [pytest.mark.gpu_test, pytest.mark.l2]


def _require_rocm_flash_attn() -> None:
    from vllm_gr.utils import accelerator

    if not torch.cuda.is_available() or not accelerator.is_rocm():
        pytest.skip("requires a ROCm GPU")
    pytest.importorskip("flash_attn")


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, 1)
    v = v.float().repeat_interleave(group, 1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


@pytest.mark.parametrize("prefix_lens", [(200, 37), (128, 1)])  # type: ignore[misc]
def test_rocm_cascade_matches_full_attention(prefix_lens: tuple[int, int]) -> None:
    _require_rocm_flash_attn()
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

    from vllm_gr.v1.attention.backends.beam_attention_state import (
        normalize_flash_attn_state,
    )
    from vllm_gr.v1.attention.backends.beam_attn_gpu import flash_attn_varlen_func
    from vllm_gr.v1.attention.backends.rocm_flash_attn import ROCM_PAGED_KV_BLOCK_SIZE

    torch.manual_seed(0)
    device, dtype = torch.device("cuda"), torch.bfloat16
    heads, kv_heads, dim, beams, steps = 16, 8, 128, 4, 3
    block = ROCM_PAGED_KV_BLOCK_SIZE
    scale = dim**-0.5
    batch = len(prefix_lens)
    blocks_per_seq = max(math.ceil(n / block) for n in prefix_lens)
    num_blocks = batch * blocks_per_seq + 3
    key_cache = torch.randn(num_blocks, block, kv_heads, dim, device=device, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    order = torch.randperm(num_blocks, device=device)[: batch * blocks_per_seq]
    block_table = order.view(batch, blocks_per_seq).to(torch.int32)
    query = torch.randn(batch * beams, heads, dim, device=device, dtype=dtype)
    suffix_k = torch.randn(batch * beams * steps, kv_heads, dim, device=device, dtype=dtype)
    suffix_v = torch.randn_like(suffix_k)

    def cu(lengths: list[int]) -> torch.Tensor:
        return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device).int()

    prefix_out, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu([beams] * batch),
        seqused_k=torch.tensor(prefix_lens, device=device, dtype=torch.int32),
        max_seqlen_q=beams,
        max_seqlen_k=max(prefix_lens),
        softmax_scale=scale,
        causal=False,
        block_table=block_table,
        return_softmax_lse=True,
        fa_version=2,
        num_splits=0,
    )
    prefix_out, prefix_lse = normalize_flash_attn_state(
        prefix_out,
        prefix_lse,
        fa_version=2,
        num_kv_heads=kv_heads,
        max_query_len=beams,
        paged_kv=True,
    )
    suffix_out, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=suffix_k,
        v=suffix_v,
        cu_seqlens_q=cu([1] * batch * beams),
        cu_seqlens_k=cu([steps] * batch * beams),
        max_seqlen_q=1,
        max_seqlen_k=steps,
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=True,
        fa_version=2,
        num_splits=0,
    )
    # This single-query GQA shape is where CUDA FA2 needs an LSE reorder.
    suffix_out, suffix_lse = normalize_flash_attn_state(
        suffix_out, suffix_lse, fa_version=2, num_kv_heads=kv_heads, max_query_len=1
    )
    out = torch.empty_like(query)
    merge_attn_states(out, prefix_out, prefix_lse, suffix_out, suffix_lse)

    for request, length in enumerate(prefix_lens):
        table = block_table[request, : math.ceil(length / block)].long()
        prefix_k = key_cache[table].reshape(-1, kv_heads, dim)[:length]
        prefix_v = value_cache[table].reshape(-1, kv_heads, dim)[:length]
        for beam in range(beams):
            row = request * beams + beam
            own = slice(row * steps, (row + 1) * steps)
            expected = _reference(
                query[row : row + 1],
                torch.cat([prefix_k, suffix_k[own]]),
                torch.cat([prefix_v, suffix_v[own]]),
                scale,
            )
            torch.testing.assert_close(out[row : row + 1].float(), expected, atol=2e-2, rtol=2e-2)
