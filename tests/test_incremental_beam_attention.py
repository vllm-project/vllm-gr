# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit contracts for the incremental Beam Attention computation layer."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vllm_gr.v1.attention.backends import beam_attn_gpu
from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    BeamAttentionMetadataBuilder,
)
from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.context import BEAM_DATA_VAR


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_copy_pool_suffix_to_fixed_buffer_is_beam_major(dtype: torch.dtype) -> None:
    max_beams, active_beams = 4, 3
    num_heads, max_steps, active_steps, head_dim = 2, 4, 2, 3
    pool = BeamAttentionPool(
        max_beams=max_beams,
        local_num_kv_heads=num_heads,
        max_decode_steps=max_steps,
        head_dim=head_dim,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    values = torch.arange(pool.unshared_key.numel(), dtype=torch.float32).reshape(
        pool.unshared_key.shape
    )
    pool.unshared_key.copy_(values.to(dtype))
    pool.unshared_value.copy_((values + 1000).to(dtype))

    capacity = max_beams * max_steps
    suffix_k_buf = torch.full((capacity, num_heads, head_dim), -1, dtype=dtype)
    suffix_v_buf = torch.full_like(suffix_k_buf, -2)
    key_ptr = suffix_k_buf.data_ptr()
    value_ptr = suffix_v_buf.data_ptr()

    contig_k, contig_v = beam_attn_gpu._copy_pool_suffix_to_buffer(
        pool.unshared_key,
        pool.unshared_value,
        suffix_k_buf,
        suffix_v_buf,
        active_beams,
        active_steps,
    )

    expected_k = (
        pool.unshared_key[0, :active_beams, :, :active_steps, :]
        .permute(0, 2, 1, 3)
        .reshape(-1, num_heads, head_dim)
    )
    expected_v = (
        pool.unshared_value[0, :active_beams, :, :active_steps, :]
        .permute(0, 2, 1, 3)
        .reshape(-1, num_heads, head_dim)
    )
    torch.testing.assert_close(contig_k, expected_k)
    torch.testing.assert_close(contig_v, expected_v)
    assert torch.all(suffix_k_buf[active_beams * active_steps :] == -1)
    assert torch.all(suffix_v_buf[active_beams * active_steps :] == -2)
    assert suffix_k_buf.data_ptr() == key_ptr
    assert suffix_v_buf.data_ptr() == value_ptr


def test_copy_pool_suffix_validates_capacity_before_writing() -> None:
    pool = BeamAttentionPool(4, 2, 3, 8, torch.device("cpu"), torch.float16)
    suffix_k_buf = torch.full((3, 2, 8), 7, dtype=torch.float16)
    suffix_v_buf = torch.full_like(suffix_k_buf, 9)

    with pytest.raises(ValueError, match="buffer capacity"):
        beam_attn_gpu._copy_pool_suffix_to_buffer(
            pool.unshared_key,
            pool.unshared_value,
            suffix_k_buf,
            suffix_v_buf,
            beam_width=2,
            suffix_kv_len=2,
        )

    assert torch.all(suffix_k_buf == 7)
    assert torch.all(suffix_v_buf == 9)


def _incremental_metadata(beam_width: int, suffix_len: int) -> BeamAttentionMetadata:
    return BeamAttentionMetadata(
        num_actual_tokens=beam_width,
        max_query_len=beam_width,
        query_start_loc=torch.tensor([0, beam_width], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int64),
        prefix_indices=torch.arange(beam_width),
        prefix_cu_seqlens_q_device=torch.tensor([0, beam_width], dtype=torch.int32),
        prefix_seqlens_kv_device=torch.tensor([4], dtype=torch.int32),
        prefix_max_q_len=beam_width,
        prefix_max_seq_len=4,
        prefix_block_table=torch.tensor([[0]], dtype=torch.int32),
        suffix_cu_seqlens_q_device=torch.arange(beam_width + 1, dtype=torch.int32),
        suffix_cu_seqlens_k_device=torch.arange(
            0, (beam_width + 1) * suffix_len, suffix_len, dtype=torch.int32
        ),
        suffix_max_q_len=1,
        suffix_max_seq_len=suffix_len,
        suffix_total_tokens=beam_width * suffix_len,
        num_prefix_indices=beam_width,
        num_heads=4,
        num_kv_heads=2,
        is_incremental_decode=True,
        incremental_beam_width=beam_width,
        incremental_suffix_kv_len=suffix_len,
    )


def test_incremental_metadata_builder_uses_active_session_contract() -> None:
    beam_width, prefix_len, block_size = 3, 5, 4
    session = BeamSearchSession(
        layer_names=("layer",),
        max_beams=4,
        local_num_kv_heads=2,
        max_decode_steps=3,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    session.begin_session(beam_width, prefix_len)
    builder = object.__new__(BeamAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.block_size = block_size
    builder.aot_schedule = False
    builder.aot_sliding_window = None
    builder.use_full_cuda_graph = False
    builder.max_cudagraph_size = None
    builder.max_num_splits = 0
    builder.num_heads_q = 4
    builder.num_kv_heads = 2
    builder._is_cuda = True
    builder._is_npu = False
    builder.npu_causal_atten_mask = None
    common = type(
        "CommonMetadata",
        (),
        {
            "num_reqs": 1,
            "query_start_loc_cpu": torch.tensor([0, beam_width], dtype=torch.int32),
            "query_start_loc": torch.tensor([0, beam_width], dtype=torch.int32),
            "seq_lens": torch.tensor([prefix_len + 1], dtype=torch.int32),
            "block_table_tensor": torch.tensor([[7, 8]], dtype=torch.int32),
            "num_actual_tokens": beam_width,
            "max_query_len": beam_width,
            "max_seq_len": prefix_len + 1,
            "causal": True,
            "slot_mapping": torch.arange(beam_width, dtype=torch.int64),
        },
    )()
    beam_info = {
        "beam_data": {
            "request": {
                "is_beam_decode": True,
                "is_incremental_decode": True,
                "beam_width": beam_width,
                "decode_step": 0,
                "prefix_len": prefix_len,
                "cache_len": prefix_len,
            }
        },
        "req_ids": ["request"],
        "beam_session": session,
    }

    token = BEAM_DATA_VAR.set(beam_info)
    try:
        metadata = builder.build(0, common)
    finally:
        BEAM_DATA_VAR.reset(token)

    assert metadata.is_incremental_decode
    assert metadata.beam_session is session
    assert metadata.incremental_beam_width == beam_width
    assert metadata.incremental_suffix_kv_len == 1
    assert metadata.prefix_indices.tolist() == [0, 1, 2]
    assert metadata.prefix_cu_seqlens_q_device.tolist() == [0, 3]
    assert metadata.prefix_seqlens_kv_device is session.actual_shared_kvlen_buf
    assert metadata.prefix_block_table.tolist() == [[7, 8]]
    assert metadata.suffix_cu_seqlens_q_device.tolist() == [0, 1, 2, 3]
    assert metadata.suffix_cu_seqlens_k_device.tolist() == [0, 1, 2, 3]
    assert metadata.suffix_total_tokens == beam_width
    assert metadata.suffix_slot_mapping_out is None


def test_incremental_cascade_uses_pool_buffer_and_caller_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beam_width, suffix_len = 2, 2
    num_query_heads, num_kv_heads, head_dim = 4, 2, 8
    metadata = _incremental_metadata(beam_width, suffix_len)
    pool = BeamAttentionPool(
        beam_width,
        num_kv_heads,
        suffix_len,
        head_dim,
        torch.device("cpu"),
        torch.float16,
    )
    pool.unshared_key.copy_(
        torch.arange(pool.unshared_key.numel(), dtype=torch.float16).reshape(
            pool.unshared_key.shape
        )
    )
    pool.unshared_value.copy_(pool.unshared_key + 100)
    suffix_k_buf = torch.empty(beam_width * suffix_len, num_kv_heads, head_dim, dtype=torch.float16)
    suffix_v_buf = torch.empty_like(suffix_k_buf)
    query = torch.zeros(beam_width, num_query_heads, head_dim, dtype=torch.float16)
    output = torch.full_like(query, -1)
    calls: list[dict[str, Any]] = []

    def fake_flash_attn_varlen_func(
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        value = 1 if len(calls) == 1 else 2
        result = torch.full_like(kwargs["q"], value)
        lse = torch.zeros(num_query_heads, beam_width, dtype=torch.float32)
        return result, lse

    def fake_merge(
        out: torch.Tensor,
        prefix_out: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_out: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> None:
        del prefix_lse, suffix_lse
        out.copy_(prefix_out + suffix_out)

    monkeypatch.setattr(beam_attn_gpu, "flash_attn_varlen_func", fake_flash_attn_varlen_func)
    monkeypatch.setattr(beam_attn_gpu, "merge_attn_states", fake_merge)

    beam_attn_gpu.cascade_attention_incremental(
        output=output,
        query=query,
        key_cache=torch.empty(1, 1, num_kv_heads, head_dim, dtype=torch.float16),
        value_cache=torch.empty(1, 1, num_kv_heads, head_dim, dtype=torch.float16),
        attn_metadata=metadata,
        unshared_key=pool.unshared_key,
        unshared_value=pool.unshared_value,
        suffix_k_buf=suffix_k_buf,
        suffix_v_buf=suffix_v_buf,
        sliding_window_size=(-1, -1),
        scale=0.5,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=3,
    )

    assert len(calls) == 2
    assert calls[0]["block_table"] is metadata.prefix_block_table
    assert calls[0]["causal"] is False
    assert calls[1]["block_table"] is None
    assert calls[1]["causal"] is True
    torch.testing.assert_close(calls[1]["k"], suffix_k_buf)
    torch.testing.assert_close(calls[1]["v"], suffix_v_buf)
    assert torch.all(output == 3)
