# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disallow-untyped-decorators=False
"""Preserve the disabled path's low-precision arithmetic for tiny suffixes."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def packed_fa2_reference(
    query: torch.Tensor,
    prefix: torch.Tensor,
    prefix_lse: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot: int,
    step: torch.Tensor,
    active: torch.Tensor | None,
) -> torch.Tensor:
    """Run the real disabled path, including FA2's GQA LSE conversion."""
    from vllm_gr.v1.attention.backends import beam_attn_gpu as gpu
    from vllm_gr.v1.attention.backends.beam_graph_kernels import copy_suffix_kv_graph

    width, heads, dim = query.shape
    kv_heads, capacity = key.shape[2:4]
    extent = max(int(step.item()) + 1, 1)
    packed_key = torch.empty((width * capacity, kv_heads, dim), dtype=query.dtype, device="cuda")
    packed_value = torch.empty_like(packed_key)
    copy_suffix_kv_graph(key, value, packed_key, packed_value, slot, step, width, active)
    cu_q = torch.arange(width + 1, dtype=torch.int32, device="cuda")
    suffix, lse = gpu.flash_attn_varlen_func(
        q=query.contiguous(),
        k=packed_key,
        v=packed_value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_q * extent,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        softmax_scale=dim**-0.5,
        causal=True,
        return_softmax_lse=True,
        fa_version=2,
        num_splits=0,
    )
    group = heads // kv_heads
    if group > 1:
        lse = lse.view(kv_heads, group, -1).transpose(1, 2).contiguous().view(heads, -1)
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    gpu.merge_attn_states(output, prefix.contiguous(), prefix_lse.contiguous(), suffix, lse)
    return output


@pytest.mark.parametrize(
    "geometry",
    [
        (1, 4, 4, 64),
        (7, 8, 4, 128),
        (128, 16, 8, 128),
        (256, 16, 4, 256),
        (3, 8, 1, 64),
        (3, 3, 1, 64),
        (7, 16, 1, 128),
        (512, 16, 8, 128),
    ],
)
@pytest.mark.parametrize("capacity,decode_step", [(1, 0), (2, 1), (3, -1), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_matches_packed_fa2_bitwise(
    geometry: tuple[int, int, int, int], capacity: int, decode_step: int, dtype: torch.dtype
) -> None:
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    torch.manual_seed(7819)
    width, heads, kv_heads, dim = geometry
    query_storage = torch.randn((width, heads, dim * 2), dtype=dtype, device="cuda")
    query = query_storage[..., ::2]
    key = torch.randn((4, width + 3, kv_heads, capacity, dim), dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    prefix = torch.randn_like(query)
    lse = torch.randn((width, heads), device="cuda").T
    active = torch.arange(width, device="cuda") % 3 != 1
    step = torch.tensor([decode_step], dtype=torch.int32, device="cuda")
    expected = packed_fa2_reference(query, prefix, lse, key, value, 3, step, active)
    output_storage = torch.full_like(query_storage, 123)
    output = output_storage[..., ::2]
    merge_short_suffix_attention(output, query, prefix, lse, key, value, 3, step, dim**-0.5, active)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.all(output_storage[..., 1::2] == 123)


@pytest.mark.parametrize("slot", [0, 3])
@pytest.mark.parametrize("masked", [False, True])
@torch.inference_mode()
def test_fa2_graph_replay_reads_step_mask_and_pool(slot: int, masked: bool) -> None:
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    torch.manual_seed(7820)
    width, heads, kv_heads, dim = 256, 16, 8, 128
    query = torch.randn((width, heads, dim), dtype=torch.bfloat16, device="cuda")
    key = torch.randn((4, width, kv_heads, 3, dim), dtype=query.dtype, device="cuda")
    value = torch.randn_like(key)
    prefix = torch.randn_like(query)
    lse = torch.randn((heads, width), device="cuda")
    active = torch.arange(width, device="cuda") % 3 != 1 if masked else None
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)

    def launch() -> None:
        merge_short_suffix_attention(
            output, query, prefix, lse, key, value, slot, step, dim**-0.5, active
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    # Negative bootstrap steps are defined only for the masked packed-copy
    # path; an unmasked copy requires a positive suffix extent.
    for decode_step in [0, 2, 1, -1, 0, 2] if masked else [0, 2, 1, 0, 2]:
        step.fill_(decode_step)
        if active is not None:
            active.logical_not_()
        key.copy_(key.roll(1, 1))
        value.copy_(value.roll(1, 1))
        expected = packed_fa2_reference(query, prefix, lse, key, value, slot, step, active)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_fa2_runtime_slot_reuses_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.attention.backends import beam_short_suffix_fa2 as short
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    production_kernel = short._short_suffix_fa2_merge_kernel
    fresh_kernel = type(production_kernel)(
        production_kernel.fn,
        do_not_specialize=[
            parameter.name for parameter in production_kernel.params if parameter.do_not_specialize
        ],
    )
    monkeypatch.setattr(short, "_short_suffix_fa2_merge_kernel", fresh_kernel)
    cache = fresh_kernel.device_caches[torch.accelerator.current_device_index()][0]
    assert not cache
    torch.manual_seed(7821)
    query = torch.randn((7, 8, 128), dtype=torch.bfloat16, device="cuda")
    key = torch.randn((4, 10, 4, 3, 128), dtype=query.dtype, device="cuda")
    value = torch.randn_like(key)
    prefix = torch.randn_like(query)
    lse = torch.randn((8, 7), device="cuda")
    step = torch.ones(1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    for slot in range(4):
        expected = packed_fa2_reference(query, prefix, lse, key, value, slot, step, None)
        merge_short_suffix_attention(output, query, prefix, lse, key, value, slot, step, 128**-0.5)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert len(cache) == 1, f"slot={slot} created another compiled specialization"
