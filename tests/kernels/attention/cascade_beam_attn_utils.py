# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Incremental cascade fixtures and independent dense prefix/suffix references."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadataBuilder,
)
from vllm_gr.v1.beam import BEAM_DATA_VAR
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
# (queries per beam, complete KV length, suffix KV length).
CASES = [(1, 32, 8), (1, 64, 4), (1, 16, 4), (1, 16, 7), (1, 128, 1), (1, 32, 7)]
NO_SHARED_PREFIX_CASE = (1, 8, 8)
GPU_CASES = [NO_SHARED_PREFIX_CASE, *CASES]
NPU_CASES = CASES


@dataclass
class BeamAttnInputs:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    suffix_key: torch.Tensor
    suffix_value: torch.Tensor
    session: BeamSearchSession
    scale: float
    num_reqs: int
    num_seqs: int
    max_seq_len: int
    block_size: int
    num_kv_heads: int
    batch_size: int
    beam_width: int
    q_seq_len: int
    head_size: int
    shared_tokens_len: int
    suffix_kv_len: int
    slot_mapping: torch.Tensor
    soft_cap: float = 0.0
    fa_version: int = 0


def prepare_inputs(
    batch_size: int,
    beam_width: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
    suffix_kv_len: int = 0,
    device: torch.device | str = "cuda",
    soft_cap: float | None = None,
    fa_version: int = 0,
) -> BeamAttnInputs:
    if q_seq_len != 1 or not 0 < suffix_kv_len <= kv_seq_len:
        raise ValueError("Incremental attention requires q_seq_len=1 and valid suffix KV length")
    device = torch.device(device)
    rows = batch_size * beam_width
    prefix_len = kv_seq_len - suffix_kv_len
    blocks = max(1, (prefix_len + block_size - 1) // block_size)
    scale = float(head_size**-0.5)
    query = torch.randn(rows, num_heads, head_size, dtype=dtype, device=device) * scale
    cache_shape = (batch_size * blocks, block_size, num_kv_heads, head_size)
    key = torch.randn(cache_shape, dtype=dtype, device=device) * scale
    value = torch.randn(cache_shape, dtype=dtype, device=device) * scale
    suffix_shape = (batch_size, beam_width, suffix_kv_len, num_kv_heads, head_size)
    suffix_key = torch.randn(suffix_shape, dtype=dtype, device=device) * scale
    suffix_value = torch.randn(suffix_shape, dtype=dtype, device=device) * scale

    # The same request-major [1, B*W, H, S, D] view that batch attention consumes.
    # Independent dense suffix tensors remain outside the pool for the oracle.
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=rows,
        num_kv_heads=num_kv_heads,
        max_decode_steps=suffix_kv_len,
        head_dim=head_size,
        num_layers=1,
        device=device,
        dtype=dtype,
    )
    slot = context.registry.allocate("attention-test", beam_width=rows, prefix_len=prefix_len)
    session = BeamSearchSession("attention-test", slot, context)
    session.begin_session(rows, prefix_len)
    session._state.decode_step = suffix_kv_len - 1
    context.decode_step_buf.fill_(suffix_kv_len - 1)
    context.decode_step_1based_buf.fill_(suffix_kv_len)
    pool = session.pools[0]
    for dst, src in ((pool.unshared_key, suffix_key), (pool.unshared_value, suffix_value)):
        dst[0].copy_(src.reshape(rows, suffix_kv_len, num_kv_heads, head_size).transpose(1, 2))
    return BeamAttnInputs(
        query=query,
        key_cache=key,
        value_cache=value,
        block_tables=torch.arange(batch_size * blocks, dtype=torch.int32, device=device).view(
            batch_size, blocks
        ),
        seq_lens=torch.full((batch_size,), kv_seq_len, dtype=torch.int32, device=device),
        suffix_key=suffix_key,
        suffix_value=suffix_value,
        session=session,
        scale=scale,
        num_reqs=batch_size,
        num_seqs=rows,
        max_seq_len=kv_seq_len,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        batch_size=batch_size,
        beam_width=beam_width,
        q_seq_len=1,
        head_size=head_size,
        shared_tokens_len=prefix_len,
        suffix_kv_len=suffix_kv_len,
        slot_mapping=torch.full((rows,), -1, dtype=torch.long, device=device),
        soft_cap=soft_cap or 0.0,
        fa_version=fa_version,
    )


def iter_beam_segments(
    inputs: BeamAttnInputs,
) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor]]:
    """Concatenate each beam's dense history without using production layout helpers."""
    for request in range(inputs.batch_size):
        blocks = inputs.block_tables[request].long()
        prefix_k = inputs.key_cache[blocks].flatten(0, 1)[: inputs.shared_tokens_len]
        prefix_v = inputs.value_cache[blocks].flatten(0, 1)[: inputs.shared_tokens_len]
        for beam in range(inputs.beam_width):
            yield (
                request * inputs.beam_width + beam,
                1,
                torch.cat((prefix_k, inputs.suffix_key[request, beam]), dim=0),
                torch.cat((prefix_v, inputs.suffix_value[request, beam]), dim=0),
            )


class MockCommonAttentionMetadata:
    def __init__(self, inputs: BeamAttnInputs, device: torch.device | str):
        self.num_reqs = inputs.num_reqs
        self.query_start_loc_cpu = torch.arange(
            0, inputs.num_seqs + 1, inputs.beam_width, dtype=torch.int32
        )
        self.query_start_loc = self.query_start_loc_cpu.to(device)
        self.seq_lens = inputs.seq_lens
        self._seq_lens_cpu = torch.full((inputs.num_reqs,), inputs.max_seq_len, dtype=torch.int32)
        self.block_table_tensor = inputs.block_tables
        self.num_actual_tokens = inputs.num_seqs
        self.max_query_len = inputs.beam_width
        self.max_seq_len = inputs.max_seq_len
        self.causal = True
        self.slot_mapping = inputs.slot_mapping


def build_beam_attn_metadata(
    inputs: BeamAttnInputs,
    device: torch.device | str,
    npu_causal_atten_mask: torch.Tensor | None = None,
) -> Any:
    req_ids = [f"req_{i}" for i in range(inputs.num_reqs)]
    beam_data = {
        rid: {
            "is_beam_decode": True,
            "prefix_len": inputs.shared_tokens_len,
            "beam_width": inputs.beam_width,
            "cache_len": inputs.shared_tokens_len,
            "decode_step": inputs.session.decode_step,
        }
        for rid in req_ids
    }
    token = BEAM_DATA_VAR.set(
        {"beam_data": beam_data, "req_ids": req_ids, "beam_session": inputs.session}
    )
    try:
        builder = BeamAttentionMetadataBuilder.__new__(BeamAttentionMetadataBuilder)
        builder.vllm_config = SimpleNamespace(speculative_config=None)
        builder.device = torch.device(device)
        builder.block_size = inputs.block_size
        builder.aot_schedule = False
        builder.aot_sliding_window = None
        builder.use_full_cuda_graph = False
        builder.max_cudagraph_size = None
        builder.max_num_splits = 0
        builder.num_heads_q = inputs.query.shape[1]
        builder.num_kv_heads = inputs.num_kv_heads
        builder._is_cuda = builder.device.type == "cuda"
        builder._is_npu = builder.device.type == "npu"
        builder._fixed_buffers = False
        builder.npu_causal_atten_mask = npu_causal_atten_mask
        return builder.build(0, MockCommonAttentionMetadata(inputs, device))
    finally:
        BEAM_DATA_VAR.reset(token)


def run_cascade_beam_attention(
    inputs: BeamAttnInputs,
    device: torch.device | str,
    vllm_flash_attn_version: int,
    logits_soft_cap: float | None = None,
    npu_causal_atten_mask: torch.Tensor | None = None,
    *,
    metadata: Any = None,
) -> torch.Tensor:
    """Exercise the platform's incremental cascade (including on xllm installations)."""
    if torch.device(device).type == "npu":
        from vllm_gr.v1.attention.backends import beam_attn_npu as backend
    else:
        from vllm_gr.v1.attention.backends import beam_attn_gpu as backend
    if metadata is None:
        metadata = build_beam_attn_metadata(inputs, device, npu_causal_atten_mask)
    output = torch.empty_like(inputs.query)
    backend.cascade_attention(
        output=output,
        query=inputs.query,
        key_cache=inputs.key_cache,
        value_cache=inputs.value_cache,
        attn_metadata=metadata,
        sliding_window_size=(-1, -1),
        scale=inputs.scale,
        logits_soft_cap=logits_soft_cap or 0.0,
        vllm_flash_attn_version=vllm_flash_attn_version,
        beam_pool=inputs.session.pools[0],
    )
    return output
