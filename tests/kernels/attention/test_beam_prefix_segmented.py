# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.attention.backends.beam_attn_gpu import prefix_attention
from vllm_gr.v1.attention.backends.beam_prefix_policy import PrefixPlan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("version,splits,segments", [(2, 1, 4), (3, 1, 4), (3, 1, 1), (3, 4, 1)])
def test_ragged_prefix_empty_tail_and_graph_replay(version, splits, segments):
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    if not is_fa_version_supported(version):
        pytest.skip("backend unavailable")
    torch.manual_seed(541)
    heads, kv_heads, dim, page = 4, 2, 64, 16
    q = torch.randn(6, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3 * 16, page, kv_heads, dim, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    cu = torch.tensor([0, 1, 4, 6], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 19, 129], device="cuda", dtype=torch.int32)
    # Non-contiguous table rows and reordered pages catch accidental KV copies/indexing.
    table = torch.arange(3 * 32, device="cuda", dtype=torch.int32).reshape(3, 32)[:, :16] // 2
    meta = SimpleNamespace(
        prefix_cu_seqlens_q_device=cu,
        prefix_seqlens_kv_device=lengths,
        prefix_max_q_len=3,
        prefix_max_seq_len=256,
        prefix_block_table=table,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
    )

    def run():
        return prefix_attention(
            q,
            k,
            v,
            meta,
            scale=dim**-0.5,
            fa_version=version,
            sliding_window_size=(-1, -1),
            logits_soft_cap=0,
            plan=PrefixPlan(version, splits, segments),
        )

    def check(state, host_cu, host_lengths):
        for b, length in enumerate(host_lengths):
            begin, end = host_cu[b : b + 2]
            if not length:
                assert torch.count_nonzero(state.out[begin:end]) == 0
                assert torch.isneginf(state.lse[:, begin:end]).all()
                continue
            keys = k[table[b].long()].flatten(0, 1)[:length].float().repeat_interleave(2, 1)
            values = v[table[b].long()].flatten(0, 1)[:length].float().repeat_interleave(2, 1)
            score = torch.einsum("qhd,khd->qhk", q[begin:end].float(), keys) * dim**-0.5
            expected = torch.einsum("qhk,khd->qhd", score.softmax(-1), values)
            torch.testing.assert_close(
                state.out[begin:end].float(), expected, atol=0.015, rtol=0.02
            )
            torch.testing.assert_close(
                state.lse[:, begin:end], score.logsumexp(-1).T, atol=0.003, rtol=0.002
            )

    check(run(), [0, 1, 4, 6], [0, 19, 129])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state = run()
    # Same addresses, different lengths/query ownership/empty segment locations.
    cu.copy_(torch.tensor([0, 2, 3, 6], device="cuda", dtype=torch.int32))
    lengths.copy_(torch.tensor([127, 0, 37], device="cuda", dtype=torch.int32))
    q.mul_(0.5)
    graph.replay()
    check(state, [0, 2, 3, 6], [127, 0, 37])


@torch.inference_mode()
def test_segmented_prefix_and_fused_suffix_graph():
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    torch.manual_seed(542)
    width, heads, kv_heads, dim = 7, 4, 2, 64
    q = torch.randn(width, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(16, 16, kv_heads, dim, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    suffix_k = torch.randn(1, width, kv_heads, 3, dim, device="cuda", dtype=q.dtype)
    suffix_v = torch.randn_like(suffix_k)
    length = torch.tensor([129], device="cuda", dtype=torch.int32)
    meta = SimpleNamespace(
        prefix_cu_seqlens_q_device=torch.tensor([0, width], device="cuda", dtype=torch.int32),
        prefix_seqlens_kv_device=length,
        prefix_max_q_len=width,
        prefix_max_seq_len=256,
        prefix_block_table=torch.arange(16, device="cuda", dtype=torch.int32)[None, :],
        prefix_scheduler_metadata=None,
        max_num_splits=0,
    )
    step = torch.tensor([1], device="cuda", dtype=torch.int32)
    active = torch.ones(width, device="cuda", dtype=torch.bool)
    output = torch.empty_like(q)

    def run():
        state = prefix_attention(
            q,
            k,
            v,
            meta,
            scale=dim**-0.5,
            fa_version=2,
            sliding_window_size=(-1, -1),
            logits_soft_cap=0,
            plan=PrefixPlan(2, 1, 4),
        )
        merge_short_suffix_attention(
            output, q, state.out, state.lse, suffix_k, suffix_v, 0, step, dim**-0.5, active
        )

    def check(prefix_len, suffix_len):
        pk = k.flatten(0, 1)[:prefix_len].float().repeat_interleave(2, 1)
        pv = v.flatten(0, 1)[:prefix_len].float().repeat_interleave(2, 1)
        sk = suffix_k[0, :, :, :suffix_len].float().repeat_interleave(2, 1)
        sv = suffix_v[0, :, :, :suffix_len].float().repeat_interleave(2, 1)
        ps = torch.einsum("qhd,khd->qhk", q.float(), pk) * dim**-0.5
        ss = torch.einsum("qhd,qhkd->qhk", q.float(), sk) * dim**-0.5
        weights = torch.cat((ps, ss), -1).softmax(-1)
        expected = torch.einsum("qhk,khd->qhd", weights[:, :, :prefix_len], pv)
        expected += torch.einsum("qhk,qhkd->qhd", weights[:, :, prefix_len:], sv)
        torch.testing.assert_close(output.float(), expected, atol=0.015, rtol=0.02)

    run()
    check(129, 2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    length.zero_()
    step.fill_(2)
    graph.replay()
    check(0, 3)
