# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""The opt-in dense CUDA Prefill must equal attention over each request's context.

The dense path attends over the layer's fresh K/V, so it is only valid when a
request has no earlier context in the paged cache. Tests make the taken path
observable: a zeroed paged cache can only match the reference via the dense
path, and a request with cached context can only match it via the paged path.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.attention.backends import beam_attn_gpu
from vllm_gr.v1.attention.backends.beam_attn_gpu import prefill_attention

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.l2,
]

HEADS, KV_HEADS, DIM, PAGE = 16, 8, 128, 16
SCALE = DIM**-0.5


def _case(query_lengths, context_lengths=None, padding=0):
    """Scheduled tokens per request plus optional earlier context per request."""
    torch.manual_seed(611)
    context_lengths = context_lengths or [0] * len(query_lengths)
    kv_lengths = [q + c for q, c in zip(query_lengths, context_lengths)]
    total = sum(query_lengths)
    q = torch.randn(total + padding, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total + padding, KV_HEADS, DIM, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    # Full per-request K/V (context then scheduled tokens), for cache and reference.
    full_k, full_v, start = [], [], 0
    for q_len, c_len in zip(query_lengths, context_lengths):
        ctx_k = torch.randn(c_len, KV_HEADS, DIM, device="cuda", dtype=q.dtype)
        ctx_v = torch.randn_like(ctx_k)
        full_k.append(torch.cat([ctx_k, k[start : start + q_len]]))
        full_v.append(torch.cat([ctx_v, v[start : start + q_len]]))
        start += q_len
    request_pages = [-(-length // PAGE) for length in kv_lengths]
    num_pages = sum(request_pages) + 3
    key_cache = torch.zeros(num_pages, PAGE, KV_HEADS, DIM, device="cuda", dtype=q.dtype)
    value_cache = torch.zeros_like(key_cache)
    # Shuffled pages catch accidental reliance on a contiguous cache layout.
    order = torch.randperm(num_pages, device="cuda").to(torch.int32)
    table = torch.zeros(len(kv_lengths), max(request_pages), device="cuda", dtype=torch.int32)
    used = 0
    for b, length in enumerate(kv_lengths):
        table[b, : request_pages[b]] = order[used : used + request_pages[b]]
        used += request_pages[b]
        rows = torch.arange(length, device="cuda")
        blocks = table[b, rows // PAGE].long()
        key_cache[blocks, rows % PAGE] = full_k[b]
        value_cache[blocks, rows % PAGE] = full_v[b]
    starts = [0, *torch.tensor(query_lengths).cumsum(0).tolist()]
    meta = SimpleNamespace(
        query_start_loc=torch.tensor(starts, device="cuda", dtype=torch.int32),
        max_query_len=max(query_lengths),
        seq_lens=torch.tensor(kv_lengths, device="cuda", dtype=torch.int32),
        max_seq_len=max(kv_lengths),
        block_table_tensor=table,
        causal=True,
        num_actual_tokens=total + padding,
        num_reqs=len(query_lengths),
        query_start_loc_cpu=starts,
        seq_lens_cpu=kv_lengths,
    )
    expected = torch.empty(total, HEADS, DIM, device="cuda", dtype=torch.float32)
    for b, (q_len, c_len) in enumerate(zip(query_lengths, context_lengths)):
        query = q[starts[b] : starts[b + 1]].float()
        keys = full_k[b].float().repeat_interleave(HEADS // KV_HEADS, 1)
        values = full_v[b].float().repeat_interleave(HEADS // KV_HEADS, 1)
        score = torch.einsum("qhd,khd->hqk", query, keys) * SCALE
        # Query i sits at absolute position c_len + i.
        visible = torch.arange(c_len + q_len, device="cuda")[None, :] <= (
            c_len + torch.arange(q_len, device="cuda")[:, None]
        )
        score = score.masked_fill(~visible, float("-inf"))
        expected[starts[b] : starts[b + 1]] = torch.einsum(
            "hqk,khd->qhd", score.softmax(-1), values
        )
    return q, k, v, key_cache, value_cache, meta, expected


def _run(q, key_cache, value_cache, meta, **kwargs):
    out = torch.zeros_like(q)
    prefill_attention(
        output=out,
        query=q,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=meta,
        sliding_window_size=(-1, -1),
        scale=SCALE,
        logits_soft_cap=kwargs.pop("logits_soft_cap", 0.0),
        vllm_flash_attn_version=2,
        **kwargs,
    )
    return out


def _check(out, expected):
    torch.testing.assert_close(out[: len(expected)].float(), expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
@pytest.mark.parametrize(
    "mode,lengths,padding",
    [
        ("fa2", (1000,), 0),
        ("fa2", (700, 1300, 64), 0),
        ("fa2", (900,), 124),
        ("cudnn", (1500,), 0),
        # Captured buckets pad past cu_seqlens_q[-1]; real rows must not change.
        ("cudnn", (1500,), 36),
        # Multi-request and short Prefill fall back to dense FA2.
        ("cudnn", (700, 1300), 0),
        ("cudnn", (1,), 0),
    ],
)
def test_dense_prefill_attends_over_fresh_kv(mode, lengths, padding):
    q, k, v, key_cache, value_cache, meta, expected = _case(lengths, padding=padding)
    _check(_run(q, key_cache, value_cache, meta), expected)
    # A zeroed paged cache can only reproduce the reference via the dense path.
    key_cache.zero_()
    value_cache.zero_()
    _check(
        _run(q, key_cache, value_cache, meta, dense_key=k, dense_value=v, dense_mode=mode), expected
    )


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["fa2", "cudnn"])
@pytest.mark.parametrize(
    "query_lengths,context_lengths",
    [
        # A non-Beam decode row (q_len=1) beside a fresh prompt, as in an eager batch.
        ((1, 300), (256, 0)),
        # A single request whose earlier tokens are already cached.
        ((200,), (512,)),
    ],
)
def test_dense_prefill_declines_requests_with_cached_context(mode, query_lengths, context_lengths):
    q, k, v, key_cache, value_cache, meta, expected = _case(query_lengths, context_lengths)
    out = _run(q, key_cache, value_cache, meta, dense_key=k, dense_value=v, dense_mode=mode)
    _check(out, expected)


@torch.inference_mode()
def test_cudnn_rejection_falls_back_to_dense_fa2(monkeypatch):
    import torch.nn.functional as F

    def reject(*args, **kwargs):
        raise RuntimeError("No available kernel. Aborting execution.")

    monkeypatch.setattr(F, "scaled_dot_product_attention", reject)
    beam_attn_gpu._warn_cudnn_unavailable.cache_clear()
    q, k, v, key_cache, value_cache, meta, expected = _case((800,))
    key_cache.zero_()
    value_cache.zero_()
    _check(
        _run(q, key_cache, value_cache, meta, dense_key=k, dense_value=v, dense_mode="cudnn"),
        expected,
    )


@torch.inference_mode()
def test_dense_prefill_declines_unsupported_layer():
    """Softcap is not handled by the dense kernels, so the paged path runs."""
    q, k, v, key_cache, value_cache, meta, _ = _case((300,))
    paged = _run(q, key_cache, value_cache, meta, logits_soft_cap=30.0)
    dense = _run(
        q,
        key_cache,
        value_cache,
        meta,
        logits_soft_cap=30.0,
        dense_key=k * 0,
        dense_value=v * 0,
        dense_mode="fa2",
    )
    torch.testing.assert_close(dense, paged, atol=0, rtol=0)


def _capture(q, key_cache, value_cache, meta, out, **kwargs):
    run = lambda: prefill_attention(  # noqa: E731
        output=out,
        query=q,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=meta,
        sliding_window_size=(-1, -1),
        scale=SCALE,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=2,
        **kwargs,
    )
    run()  # warm up outside capture, as the graph runners do
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


@torch.inference_mode()
@pytest.mark.parametrize("mode", [None, "fa2", "cudnn"])
def test_decode_graph_capture_keeps_paged_attention(mode):
    """A native decode capture must not freeze dense attention.

    Native decode capture uses seq_len == query_len == 1 dummies, which pass the
    CPU length check; the graph then replays with the request's cached context.
    """
    q, k, v, key_cache, value_cache, meta, expected = _case((1,), (256,))
    meta.seq_lens.fill_(1)
    meta.seq_lens_cpu = [1]
    kwargs = {} if mode is None else {"dense_key": k, "dense_value": v, "dense_mode": mode}
    out = torch.zeros_like(q)
    graph = _capture(q, key_cache, value_cache, meta, out, **kwargs)
    # Replay as a real decode step: same buffers, 256 cached context tokens.
    meta.seq_lens.fill_(257)
    meta.seq_lens_cpu = [257]
    out.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    _check(out, expected)


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["fa2", "cudnn"])
def test_fresh_prefill_capture_scope_captures_dense_attention(mode):
    from vllm_gr.v1.attention.backends.beam_dense_prefill import fresh_prefill_capture

    q, k, v, key_cache, value_cache, meta, expected = _case((600,))
    # A zeroed paged cache can only reproduce the reference via the dense path.
    key_cache.zero_()
    value_cache.zero_()
    out = torch.zeros_like(q)
    with fresh_prefill_capture():
        graph = _capture(
            q, key_cache, value_cache, meta, out, dense_key=k, dense_value=v, dense_mode=mode
        )
    out.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    _check(out, expected)
