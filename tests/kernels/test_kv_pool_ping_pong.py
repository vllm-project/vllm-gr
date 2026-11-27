# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""KV history, attention, slot isolation and replay across fixed-address banks."""

from types import SimpleNamespace

import pytest
import torch
import triton

from vllm_gr.ops.cuda.beam_kv_cache import cache_unshared_kv_graph, copy_suffix_kv_graph
from vllm_gr.ops.cuda.beam_kv_reorder import (
    _select_kv_ping_pong_kernel,
    select_unshared_kv_ping_pong,
)
from vllm_gr.v1.attention.backends.beam_short_suffix import merge_short_suffix_attention
from vllm_gr.v1.beam.attention_context import BeamAttentionContext
from vllm_gr.v1.beam.attention_pool import BeamAttentionPool

pytestmark = [pytest.mark.l2, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")]


def _context(batch: int, dim: int, dtype: torch.dtype) -> BeamAttentionContext:
    return BeamAttentionContext(
        max_batch=batch,
        max_beams=8,
        num_kv_heads=2,
        max_decode_steps=3,
        head_dim=dim,
        num_layers=2,
        device=torch.device("cuda"),
        dtype=dtype,
        enable_kv_pool_ping_pong=True,
    )


def _bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(
        actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16)
    )


def _banks(pool: BeamAttentionPool) -> tuple[torch.Tensor, ...]:
    assert pool.standby_key is not None and pool.standby_value is not None
    return pool.unshared_key, pool.unshared_value, pool.standby_key, pool.standby_value


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize("batch", [1, 3])  # type: ignore[misc]
@pytest.mark.parametrize("dim", [16, 64])  # type: ignore[misc]
@pytest.mark.parametrize("target_width", [4, 8])  # type: ignore[misc]
def test_ping_pong_history_attention_and_slot_isolation(
    dtype: torch.dtype, batch: int, dim: int, target_width: int
) -> None:
    context = _context(batch, dim, dtype)
    assert context.pool_key_ptrs is not None and context.pool_value_ptrs is not None
    assert context.standby_key_ptrs is not None and context.standby_value_ptrs is not None
    slot = batch - 1
    reference = [(pool.unshared_key.clone(), pool.unshared_value.clone()) for pool in context.pools]
    addresses = [tensor.data_ptr() for pool in context.pools for tensor in _banks(pool)]
    # Other slots contain observable data; reordering the active slot must
    # never disturb them, in either bank or across different layers.
    if slot:
        for layer, pool in enumerate(context.pools):
            for tensor in _banks(pool):
                tensor[:slot].fill_(layer + 17)
    width = context.max_beams
    for step in range(3):
        context.decode_step_buf.fill_(step)
        for layer, pool in enumerate(context.pools):
            curr_k = torch.randn(width, 2, dim, dtype=dtype, device="cuda")
            curr_v = torch.randn_like(curr_k)
            cache_unshared_kv_graph(
                pool.unshared_key,
                pool.unshared_value,
                curr_k,
                curr_v,
                slot,
                context.decode_step_buf,
                standby_key=pool.standby_key,
                standby_value=pool.standby_value,
            )
            rk, rv = reference[layer]
            rk[slot, :width, :, step].copy_(curr_k)
            rv[slot, :width, :, step].copy_(curr_v)
            packed = [torch.empty(width * 3, 2, dim, dtype=dtype, device="cuda") for _ in range(4)]
            copy_suffix_kv_graph(
                pool.unshared_key,
                pool.unshared_value,
                *packed[:2],
                slot,
                context.decode_step_buf,
                beam_width=width,
                standby_key=pool.standby_key,
                standby_value=pool.standby_value,
            )
            copy_suffix_kv_graph(
                rk, rv, *packed[2:], slot, context.decode_step_buf, beam_width=width
            )
            for actual, expected in zip(packed[:2], packed[2:]):
                _bits_equal(actual[: width * (step + 1)], expected[: width * (step + 1)])

            query = torch.randn(width, 4, dim, dtype=dtype, device="cuda")
            prefix = torch.randn_like(query)
            lse = torch.randn(4, width, dtype=torch.float32, device="cuda")
            actual, expected = torch.empty_like(query), torch.empty_like(query)
            merge_short_suffix_attention(
                actual,
                query,
                prefix,
                lse,
                pool.unshared_key,
                pool.unshared_value,
                slot,
                context.decode_step_buf,
                dim**-0.5,
                standby_key=pool.standby_key,
                standby_value=pool.standby_value,
            )
            merge_short_suffix_attention(
                expected,
                query,
                prefix,
                lse,
                rk,
                rv,
                slot,
                context.decode_step_buf,
                dim**-0.5,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Fan-out, arbitrary ordering, and rescuing a high parent while the
        # live width shrinks are independent of the physical pool width.
        mapping = [width - 1, 0, width - 1, 1, 2, 0, 3, 3][:target_width]
        parents = torch.tensor(mapping, dtype=torch.int32, device="cuda")
        if step < 2:
            select_unshared_kv_ping_pong(
                parents,
                slot,
                [pool.unshared_key for pool in context.pools],
                [pool.unshared_value for pool in context.pools],
                [pool.standby_key for pool in context.pools],
                [pool.standby_value for pool in context.pools],
                step,
                context.pool_key_ptrs,
                context.pool_value_ptrs,
                context.standby_key_ptrs,
                context.standby_value_ptrs,
            )
            for rk, rv in reference:
                for tensor in (rk, rv):
                    history = tensor[slot, :, :, : step + 1].index_select(0, parents.long())
                    tensor[slot, :target_width, :, : step + 1].copy_(history)
            width = target_width
        if slot:
            for layer, pool in enumerate(context.pools):
                for tensor in _banks(pool):
                    assert bool((tensor[:slot] == layer + 17).all())
    assert addresses == [tensor.data_ptr() for pool in context.pools for tensor in _banks(pool)]
    # At B=1 the existing scratch is the second bank, not a third allocation.
    assert context.standby_key_buf is not None and context.standby_value_buf is not None
    assert context.aux_key_buf.data_ptr() == context.standby_key_buf.data_ptr()
    assert context.aux_value_buf.data_ptr() == context.standby_value_buf.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_one_graph_replays_banks_masks_and_reused_sessions(dtype: torch.dtype) -> None:
    context = _context(2, 16, dtype)
    pool = context.pools[0]
    slot, width, heads, dim = 1, 8, 2, 16
    current_k = torch.randn(width, heads, dim, dtype=dtype, device="cuda")
    current_v = torch.randn_like(current_k)
    active = torch.ones(width, dtype=torch.bool, device="cuda")
    execute = torch.ones(1, dtype=torch.bool, device="cuda")
    parents = torch.zeros(width, dtype=torch.int32, device="cuda")
    packed_k = torch.empty(width * 3, heads, dim, dtype=dtype, device="cuda")
    packed_v = torch.empty_like(packed_k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache_unshared_kv_graph(
            pool.unshared_key,
            pool.unshared_value,
            current_k,
            current_v,
            slot,
            context.decode_step_buf,
            active,
            standby_key=pool.standby_key,
            standby_value=pool.standby_value,
        )
        copy_suffix_kv_graph(
            pool.unshared_key,
            pool.unshared_value,
            packed_k,
            packed_v,
            slot,
            context.decode_step_buf,
            width,
            active,
            standby_key=pool.standby_key,
            standby_value=pool.standby_value,
        )
        _select_kv_ping_pong_kernel[(2 * width * heads,)](
            context.pool_key_ptrs,
            context.pool_value_ptrs,
            context.standby_key_ptrs,
            context.standby_value_ptrs,
            slot * pool.unshared_key.stride(0) * pool.unshared_key.element_size(),
            width,
            context.decode_step_buf,
            parents,
            H=heads,
            BLOCK_D=triton.next_power_of_2(dim),
            D=dim,
            stride_pw=pool.unshared_key.stride(1),
            stride_ph=pool.unshared_key.stride(2),
            stride_ps=pool.unshared_key.stride(3),
            active_mask=active,
            execute=execute,
            DEVICE_STEPS=True,
        )
    # Capture/warmup may write banks. Session reuse intentionally performs no
    # KV clearing; step zero overwrites the valid history before it is read.
    rk, rv = torch.zeros_like(pool.unshared_key), torch.zeros_like(pool.unshared_value)
    for length in (1, 3, 2, 3):
        for step in range(length):
            context.decode_step_buf.fill_(step)
            current_k.normal_()
            current_v.normal_()
            active.fill_(True)
            active[-1] = False
            execute.fill_(step + 1 < length)
            parents.fill_(0 if step + 1 < length else -1)
            rk[slot, :-1, :, step].copy_(current_k[:-1])
            rv[slot, :-1, :, step].copy_(current_v[:-1])
            graph.replay()
            for actual, tensor in ((packed_k, rk), (packed_v, rv)):
                expected = tensor[slot, :, :, : step + 1].clone()
                expected[-1].zero_()
                expected = expected.permute(0, 2, 1, 3).reshape(-1, heads, dim)
                _bits_equal(actual[: width * (step + 1)], expected)
            if step + 1 < length:
                for tensor in (rk, rv):
                    selected = tensor[slot, 0:1, :, : step + 1].clone()
                    tensor[slot, :-1, :, : step + 1].copy_(selected)


@pytest.mark.parametrize("batch_size", [1, 3])  # type: ignore[misc]
def test_batch_attention_views_keep_both_banks_and_dynamic_opt_out(batch_size: int) -> None:
    from vllm_gr.v1.attention.backends.beam_attn_gpu import GPUBeamAttentionBatch

    context = _context(batch_size, 16, torch.bfloat16)
    sessions = tuple(
        SimpleNamespace(
            context=context,
            slot=slot,
            beam_width=8,
            decode_step=0,
            session_id=str(slot),
            is_active=True,
        )
        for slot in range(batch_size)
    )
    batch = GPUBeamAttentionBatch(sessions)
    dynamic = GPUBeamAttentionBatch(sessions, kv_pool_ping_pong=False)
    for source, view, opted_out in zip(context.pools, batch.pools, dynamic.pools):
        for tensor, flattened in zip(_banks(source), _banks(view)):
            assert tensor.data_ptr() == flattened.data_ptr()
            assert flattened.shape == (1, batch_size * 8, 2, 3, 16)
        assert opted_out.standby_key is None and opted_out.standby_value is None
        for step in range(3):
            context.decode_step_buf.fill_(step)
            current_k = torch.randn(batch_size * 8, 2, 16, dtype=torch.bfloat16, device="cuda")
            current_v = torch.randn_like(current_k)
            cache_unshared_kv_graph(
                view.unshared_key,
                view.unshared_value,
                current_k,
                current_v,
                0,
                context.decode_step_buf,
                standby_key=view.standby_key,
                standby_value=view.standby_value,
            )
            key_bank, value_bank = _banks(source)[2:] if step & 1 else _banks(source)[:2]
            _bits_equal(key_bank[:, :, :, step], current_k.view(batch_size, 8, 2, 16))
            _bits_equal(value_bank[:, :, :, step], current_v.view(batch_size, 8, 2, 16))
