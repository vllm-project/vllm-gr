# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Experimental eager CPU backend: provider selection and reference attention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gr.platforms.cpu import CPU_BACKEND
from vllm_gr.platforms.identity import PlatformIdentity
from vllm_gr.platforms.loader import load_backend_plugins
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests, resolve_backend
from vllm_gr.v1.attention.backends import beam_attn_cpu

pytestmark = pytest.mark.cpu_test


def _identity(enum: str, device: str) -> PlatformIdentity:
    return PlatformIdentity(enum, device, device, "cls", ("cls",))


@pytest.mark.l1  # type: ignore[misc]
def test_cpu_provider_registers_and_matches_only_cpu() -> None:
    _reset_backend_registry_for_tests()
    try:
        load_backend_plugins()
        assert resolve_backend("cpu") is CPU_BACKEND
    finally:
        _reset_backend_registry_for_tests()
    assert CPU_BACKEND.matches(_identity("CPU", "cpu"))
    assert not CPU_BACKEND.matches(_identity("CUDA", "cuda"))
    assert CPU_BACKEND.get_attention_execution() is beam_attn_cpu


def _dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    group = q.shape[1] // k.shape[1]
    k, v = (t.repeat_interleave(group, dim=1).transpose(0, 1) for t in (k, v))
    out = F.scaled_dot_product_attention(q.transpose(0, 1), k, v, is_causal=causal)
    return out.transpose(0, 1)


@pytest.mark.l1  # type: ignore[misc]
def test_cpu_prefill_and_beam_decode_match_dense_attention() -> None:
    torch.manual_seed(0)
    heads, kv_heads, dim, block, prompt, width, step = 4, 2, 8, 4, 6, 3, 1
    scale = dim**-0.5
    key_cache = torch.zeros(3, block, kv_heads, dim)
    value_cache = torch.zeros_like(key_cache)
    blocks = torch.tensor([[2, 0]])
    key, value = torch.randn(2, prompt, kv_heads, dim).unbind(0)
    query = torch.randn(prompt, heads, dim)
    slots = torch.tensor([2 * block + i if i < block else i - block for i in range(prompt)])

    beam_attn_cpu.reshape_and_cache(key, value, key_cache, value_cache, slots, "auto", None, None)
    prefill = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=prompt,
        block_table=blocks,
        query_start_loc_cpu=[0, prompt],
        seq_lens_cpu=[prompt],
        causal=True,
    )
    out = torch.empty_like(query)
    beam_attn_cpu.prefill_attention(out, query, key_cache, value_cache, prefill, None, scale, 0, 0)
    torch.testing.assert_close(out, _dense(query, key, value, causal=True))

    # Each beam attends to the shared prompt plus its own decode_step + 1 suffix tokens.
    pool = SimpleNamespace(
        unshared_key=torch.randn(1, width, kv_heads, step + 1, dim),
        unshared_value=torch.randn(1, width, kv_heads, step + 1, dim),
    )
    session = SimpleNamespace(is_active=True, decode_step=step, beam_width=width, slot=0)
    decode = SimpleNamespace(
        is_beam_decode=True,
        beam_session=session,
        prefix_indices=torch.arange(width),
        std_indices=None,
        num_prefix_indices=width,
        suffix_max_q_len=1,
        prefix_block_table=blocks,
        prefix_cu_seqlens_q_cpu=[width],
        prefix_seqlens_kv_cpu=[prompt],
    )
    beam_query = torch.randn(width, heads, dim)
    out = torch.empty_like(beam_query)
    beam_attn_cpu.cascade_attention(
        out, beam_query, key_cache, value_cache, decode, None, scale, 0, 0, beam_pool=pool
    )
    for beam in range(width):
        beam_key = torch.cat([key, pool.unshared_key[0, beam].transpose(0, 1)])
        beam_value = torch.cat([value, pool.unshared_value[0, beam].transpose(0, 1)])
        expected = _dense(beam_query[beam : beam + 1], beam_key, beam_value, causal=False)
        torch.testing.assert_close(out[beam : beam + 1], expected)
