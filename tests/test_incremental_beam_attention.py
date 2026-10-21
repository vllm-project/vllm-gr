# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit contracts for the incremental Beam Attention computation layer."""

from __future__ import annotations

import torch

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    BeamAttentionMetadataBuilder,
)
from vllm_gr.v1.beam import BeamAttentionContext, BeamSearchSession
from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool
from vllm_gr.v1.beam.context import BEAM_DATA_VAR


def _make_session(
    *,
    beam_width: int = 3,
    max_beams: int = 4,
    num_kv_heads: int = 2,
    max_decode_steps: int = 3,
    head_dim: int = 8,
    num_layers: int = 1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
    session_id: str = "test_session",
) -> BeamSearchSession:
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=max_beams,
        num_kv_heads=num_kv_heads,
        max_decode_steps=max_decode_steps,
        head_dim=head_dim,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )
    slot = context.registry.allocate(session_id, beam_width=0, prefix_len=0)
    return BeamSearchSession(session_id=session_id, slot=slot, context=context)


def test_pool_to_suffix_copy_is_beam_major() -> None:
    """Test that the pool-to-suffix copy logic (done inline in cascade_attention)
    produces the correct beam-major layout.

    In our 8ffc23d implementation, the copy is:
        pool_k = pool.unshared_key[slot, :W, :, :step+1, :]  # [W, H, step+1, D]
        suffix_k_buf[:total_kv] = pool_k.permute(0, 2, 1, 3).reshape(-1, H, D)
    """
    max_beams, active_beams = 4, 3
    num_heads, max_steps, active_steps, head_dim = 2, 4, 2, 3
    pool = BeamAttentionPool(
        max_batch=1,
        max_beams=max_beams,
        num_kv_heads=num_heads,
        max_decode_steps=max_steps,
        head_dim=head_dim,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    values = torch.arange(pool.unshared_key.numel(), dtype=torch.float32).reshape(
        pool.unshared_key.shape
    )
    pool.unshared_key.copy_(values.to(torch.float16))
    pool.unshared_value.copy_((values + 1000).to(torch.float16))

    slot = 0
    W = active_beams
    step = active_steps - 1  # active_steps = step + 1

    # Simulate the pool-to-suffix copy from cascade_attention
    pool_k = pool.unshared_key[slot, :W, :, : step + 1, :]  # [W, H, step+1, D]
    pool_v = pool.unshared_value[slot, :W, :, : step + 1, :]

    total_kv = W * (step + 1)
    suffix_k_buf = torch.zeros(W * max_steps, num_heads, head_dim, dtype=torch.float16)
    suffix_v_buf = torch.zeros_like(suffix_k_buf)
    suffix_k_buf[:total_kv] = pool_k.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim)
    suffix_v_buf[:total_kv] = pool_v.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim)

    expected_k = (
        pool.unshared_key[slot, :active_beams, :, :active_steps, :]
        .permute(0, 2, 1, 3)
        .reshape(-1, num_heads, head_dim)
    )
    expected_v = (
        pool.unshared_value[slot, :active_beams, :, :active_steps, :]
        .permute(0, 2, 1, 3)
        .reshape(-1, num_heads, head_dim)
    )
    torch.testing.assert_close(suffix_k_buf[:total_kv], expected_k)
    torch.testing.assert_close(suffix_v_buf[:total_kv], expected_v)
    assert torch.all(suffix_k_buf[total_kv:] == 0)
    assert torch.all(suffix_v_buf[total_kv:] == 0)


def test_suffix_buf_from_session_returns_correct_slice() -> None:
    """Test that session.suffix_k_buf / suffix_v_buf return the correct
    slice of the context's shared suffix buffers for the session's slot."""
    session = _make_session(
        beam_width=3, max_beams=4, num_kv_heads=2, max_decode_steps=3, head_dim=8
    )
    session.begin_session(beam_width=3, prefix_len=5)

    k_buf = session.suffix_k_buf
    v_buf = session.suffix_v_buf

    # Expected shape: [beam_width * max_decode_steps, num_kv_heads, head_dim]
    assert k_buf.shape == (3 * 3, 2, 8)
    assert v_buf.shape == (3 * 3, 2, 8)

    # Verify that the suffix buffers are views into the context's shared buffers
    offset = session.slot * 4 * 3  # slot * max_beams * max_decode_steps
    size = 3 * 3  # beam_width * max_decode_steps
    assert k_buf.data_ptr() == session.context.suffix_k_buf[offset : offset + size].data_ptr()
    assert v_buf.data_ptr() == session.context.suffix_v_buf[offset : offset + size].data_ptr()


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
        num_prefix_indices=beam_width,
        num_heads=4,
        num_kv_heads=2,
        is_beam_decode=True,
        beam_session=None,
        incremental_suffix_kv_len=suffix_len,
    )


def test_incremental_metadata_builder_uses_active_session_contract() -> None:
    beam_width, prefix_len, block_size = 3, 5, 4
    session = _make_session(
        beam_width=beam_width, max_beams=4, num_kv_heads=2, max_decode_steps=3, head_dim=8
    )
    session.begin_session(beam_width, prefix_len)

    # Build a minimal metadata builder using our 8ffc23d API

    # Create a minimal builder instance
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
    builder._fixed_buffers = False  # no pre-allocated buffers in test harness
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
                "beam_width": beam_width,
                "beam_decode_steps": 1,
                "prefix_len": prefix_len,
                "cache_len": prefix_len,
                "decode_step": 0,
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

    assert metadata.is_beam_decode
    assert metadata.beam_session is session
    assert metadata.incremental_suffix_kv_len == 1
    assert metadata.prefix_indices.tolist() == [0, 1, 2]
    assert metadata.prefix_cu_seqlens_q_device.tolist() == [0, 3]
    assert metadata.prefix_block_table.tolist() == [[7, 8]]
    assert metadata.suffix_cu_seqlens_q_device.tolist() == [0, 1, 2, 3]
    assert metadata.suffix_cu_seqlens_k_device.tolist() == [0, 1, 2, 3]
