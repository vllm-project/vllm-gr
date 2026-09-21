# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared test infrastructure for the cascade beam-attention correctness tests.

Used by:
  - test_cascade_beam_attn_gpu.py   (GPU, flash-attn reference kernel)
  - test_cascade_beam_attn_npu.py   (NPU, pure-torch reference kernel)

Keeps the device-agnostic input construction, the beam prefix/suffix gather
(the same decomposition the real metadata builder performs, replayed here via
``_compute_beam_bounds``), and the backend invocation plumbing in one place,
so the two device-specific test files only contain what actually differs
between them: the reference kernel's math and each device's parameter sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, cast

import torch

from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl
from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadataBuilder,
)
from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.geometry import _compute_beam_bounds

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]  # (query_heads, kv_heads); (4, 4) is MHA

# (q_seq_len, kv_seq_len, suffix_kv_len) beam-decode scenarios common to both
# device tests.
CASES: list[tuple[int, int, int]] = [
    (8, 32, 8),  # Standard: prefix complete, beams decode full suffixes.
    (4, 64, 4),  # Large general case with a longer prefix.
    (4, 16, 4),  # General small-context generation.
    (1, 16, 4),  # Chunked execution, strict budget (partial/skipped beams).
    (1, 128, 1),  # Pure single-token decode over a large context.
    (4, 32, 7),  # Partial block-alignment: prime/unaligned lengths.
]

# No shared prefix (shared_tokens_len == 0): pure beam decode. GPU-only --
# NPU excludes it because CANN rejects the resulting empty block table.
NO_SHARED_PREFIX_CASE: tuple[int, int, int] = (8, 8, 8)

GPU_CASES: list[tuple[int, int, int]] = [NO_SHARED_PREFIX_CASE, *CASES]
NPU_CASES: list[tuple[int, int, int]] = CASES


@dataclass
class BeamAttnInputs:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
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
    num_seqs = batch_size * beam_width
    scale = float(1.0 / (head_size**0.5))

    shared_tokens_len = kv_seq_len - suffix_kv_len
    beam_seq_len = shared_tokens_len + beam_width * suffix_kv_len
    blocks_per_req = (beam_seq_len + block_size - 1) // block_size
    num_reqs = batch_size

    query = (
        torch.randn(num_seqs * q_seq_len, num_heads, head_size, dtype=dtype, device=device) * scale
    )

    total_blocks = num_reqs * blocks_per_req
    key_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )
    value_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )

    block_tables = torch.zeros((num_reqs, blocks_per_req), dtype=torch.int32, device=device)
    current_block = 0
    for i in range(num_reqs):
        block_tables[i] = torch.arange(
            current_block, current_block + blocks_per_req, dtype=torch.int32, device=device
        )
        current_block += blocks_per_req

    seq_lens = torch.tensor([beam_seq_len] * num_reqs, dtype=torch.int32, device=device)

    num_actual_tokens = num_seqs * q_seq_len
    slot_mapping = torch.zeros(num_actual_tokens, dtype=torch.long, device=device)

    chunk_budget = beam_width * q_seq_len
    cache_len = (shared_tokens_len // block_size) * block_size
    remaining_prefix = shared_tokens_len - cache_len
    total_uncached_req = remaining_prefix + beam_width * suffix_kv_len
    write_len = min(total_uncached_req, chunk_budget)

    all_physical_slots = (
        block_tables.to(torch.long).unsqueeze(-1) * block_size
        + torch.arange(block_size, device=device)
    ).view(num_reqs, -1)
    slot_mapping.view(num_reqs, chunk_budget)[:, :write_len] = all_physical_slots[
        :, cache_len : cache_len + write_len
    ]

    logits_soft_cap = soft_cap if soft_cap is not None else 0
    return BeamAttnInputs(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        num_reqs=num_reqs,
        num_seqs=num_seqs,
        max_seq_len=beam_seq_len,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        batch_size=batch_size,
        beam_width=beam_width,
        q_seq_len=q_seq_len,
        head_size=head_size,
        shared_tokens_len=shared_tokens_len,
        suffix_kv_len=suffix_kv_len,
        slot_mapping=slot_mapping,
        soft_cap=logits_soft_cap,
        fa_version=fa_version,
    )


def iter_beam_segments(
    inputs: BeamAttnInputs,
    cache_len: int | None = None,
) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor]]:
    """Replays the cascade metadata builder's per-beam bookkeeping.

    Yields ``(q_start, q_uncached, beam_k, beam_v)`` for each beam that still
    has uncached tokens in this chunk, where ``beam_k``/``beam_v`` is the
    ground-truth ``[prefix || suffix]`` KV segment (KV-head layout, not yet
    head-expanded) that beam attends to, and
    ``query[q_start : q_start + q_uncached]`` is the query slice it produces.

    *cache_len* overrides the recompute-path default (block-aligned prefix
    length) so a caller can replay a chunked-prefill step that already has
    cached suffix KV in the paged cache (the ``suffix_past_coords`` case).

    Both device references (flash-attn packing on GPU, per-beam causal
    softmax on NPU) build on top of this same decomposition; only what they
    do with each segment differs.
    """
    seq_lens_cpu = inputs.seq_lens.cpu().tolist()
    q_ptr = 0
    for i in range(inputs.batch_size):
        phys_blocks = inputs.block_tables[i]
        seq_len = seq_lens_cpu[i]
        k_seq = inputs.key_cache[phys_blocks].flatten(0, 1)[:seq_len]
        v_seq = inputs.value_cache[phys_blocks].flatten(0, 1)[:seq_len]

        prefix_len = inputs.shared_tokens_len
        steps = inputs.suffix_kv_len
        w = inputs.beam_width

        b_cache_len = (
            cache_len if cache_len is not None else prefix_len // inputs.block_size * inputs.block_size
        )
        chunk_budget = inputs.q_seq_len * inputs.beam_width

        for b in range(w):
            if chunk_budget <= 0:
                break

            b_uncached, b_suffix_len, b_past_suffix, b_physical = _compute_beam_bounds(
                b, prefix_len, b_cache_len, steps, chunk_budget
            )
            if b_uncached <= 0:
                continue

            remaining_prefix = prefix_len - b_cache_len
            if remaining_prefix >= 0:
                eff_prefix_len = b_cache_len
                b_suffix_start = b_cache_len
            else:
                eff_prefix_len = prefix_len
                b_suffix_start = prefix_len + b_physical * steps

            prefix_k = k_seq[:eff_prefix_len]
            prefix_v = v_seq[:eff_prefix_len]
            suffix_k = k_seq[b_suffix_start : b_suffix_start + b_suffix_len]
            suffix_v = v_seq[b_suffix_start : b_suffix_start + b_suffix_len]

            beam_k = torch.cat([prefix_k, suffix_k], dim=0)
            beam_v = torch.cat([prefix_v, suffix_v], dim=0)

            yield q_ptr, b_uncached, beam_k, beam_v

            q_ptr += b_uncached
            chunk_budget -= b_uncached
            b_cache_len += b_uncached


def run_incremental_reference(
    inputs: BeamAttnInputs,
    vllm_flash_attn_version: int = 2,
) -> torch.Tensor:
    """Dense attention reference for incremental decode semantics.

    Each beam contributes exactly 1 query token and attends to its complete
    shared prefix + ``suffix_kv_len`` suffix KV rows (``decode_step + 1``).

    This is what ``cascade_attention`` with the pool path or the gather path
    must produce.  The existing ``iter_beam_segments`` uses recompute
    semantics (chunking) and is NOT suitable for the incremental case.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    device = inputs.query.device
    w = inputs.beam_width
    prefix_len = inputs.shared_tokens_len
    suffix_len = inputs.suffix_kv_len
    beam_seq_len = prefix_len + w * suffix_len

    # Unpack the first request's block table to get dense KV.
    phys_blocks = inputs.block_tables[0]
    k_seq = inputs.key_cache[phys_blocks].flatten(0, 1)[:beam_seq_len]
    v_seq = inputs.value_cache[phys_blocks].flatten(0, 1)[:beam_seq_len]
    prefix_k = k_seq[:prefix_len]   # [prefix_len, H, D]
    prefix_v = v_seq[:prefix_len]

    # Per-beam suffix slice: [prefix_len + i*suffix_len : prefix_len + (i+1)*suffix_len]
    dense_k_parts = []
    dense_v_parts = []
    seqlens_k = []
    for i in range(w):
        start = prefix_len + i * suffix_len
        beam_k = torch.cat([prefix_k, k_seq[start : start + suffix_len]], dim=0)
        beam_v = torch.cat([prefix_v, v_seq[start : start + suffix_len]], dim=0)
        dense_k_parts.append(beam_k)
        dense_v_parts.append(beam_v)
        seqlens_k.append(prefix_len + suffix_len)

    k_tensor = torch.cat(dense_k_parts, dim=0)
    v_tensor = torch.cat(dense_v_parts, dim=0)

    cu_seqlens_k = (
        torch.tensor([0] + seqlens_k, dtype=torch.int32, device=device)
        .cumsum(dim=0)
        .to(torch.int32)
    )
    cu_seqlens_q = torch.arange(0, w + 1, dtype=torch.int32, device=device)

    output = torch.zeros_like(inputs.query)
    flash_attn_varlen_func(
        q=inputs.query,
        k=k_tensor,
        v=v_tensor,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=1,
        max_seqlen_k=prefix_len + suffix_len,
        softmax_scale=inputs.scale,
        causal=True,
        window_size=(-1, -1),
        softcap=inputs.soft_cap,
        fa_version=vllm_flash_attn_version,
    )
    torch.accelerator.synchronize()
    return output


class MockCommonAttentionMetadata:
    def __init__(self, inputs: BeamAttnInputs, device: torch.device | str):
        self.num_reqs = inputs.num_reqs
        self.query_start_loc_cpu = torch.arange(
            0,
            (inputs.num_seqs + 1) * inputs.q_seq_len,
            inputs.beam_width * inputs.q_seq_len,
            dtype=torch.int32,
        )
        self.query_start_loc = self.query_start_loc_cpu.to(device)
        self.seq_lens = inputs.seq_lens
        self.block_table_tensor = inputs.block_tables
        self.num_actual_tokens = inputs.num_seqs * inputs.q_seq_len
        self.max_query_len = inputs.q_seq_len * inputs.beam_width
        self.max_seq_len = inputs.max_seq_len
        self.causal = True
        self.slot_mapping = inputs.slot_mapping


def build_beam_attn_metadata(
    inputs: BeamAttnInputs,
    device: torch.device | str,
    npu_causal_atten_mask: torch.Tensor | None = None,
    beam_session: Any = None,
    decode_step: int | None = None,
) -> Any:
    """Builds attn_metadata via BeamAttentionMetadataBuilder, wiring the
    BEAM_DATA_VAR contextvar the same way both device tests need.

    When *beam_session* and *decode_step* are supplied the builder takes
    the incremental decode path (one query per beam + decode_step+1 suffix
    KV rows).  Otherwise it falls back to the recompute/gather path.
    """
    common_attn_metadata = MockCommonAttentionMetadata(inputs, device)

    beam_info: dict[str, Any] = {
        "beam_data": {},
        "req_ids": [f"req_{i}" for i in range(inputs.num_reqs)],
    }
    if beam_session is not None:
        beam_info["beam_session"] = beam_session
    beam_data = cast(dict[str, Any], beam_info["beam_data"])
    for i in range(inputs.num_reqs):
        entry: dict[str, Any] = {
            "is_beam_decode": True,
            "prefix_len": inputs.shared_tokens_len,
            "beam_width": inputs.beam_width,
            "beam_decode_steps": inputs.suffix_kv_len,
            "cache_len": inputs.shared_tokens_len // inputs.block_size * inputs.block_size,
        }
        if decode_step is not None:
            entry["decode_step"] = decode_step
        beam_data[f"req_{i}"] = entry

    token = BEAM_DATA_VAR.set(beam_info)
    try:
        builder = BeamAttentionMetadataBuilder.__new__(BeamAttentionMetadataBuilder)
        builder.device = device
        builder.block_size = inputs.block_size
        builder.aot_schedule = False
        builder.aot_sliding_window = None
        builder.use_full_cuda_graph = False
        builder.max_cudagraph_size = None
        builder.max_num_splits = 0
        builder.num_heads_q = inputs.query.shape[1]
        builder.num_kv_heads = inputs.num_kv_heads
        device_type = getattr(device, "type", str(device).partition(":")[0])
        builder._is_cuda = device_type == "cuda"
        builder._is_npu = device_type == "npu"
        builder._fixed_buffers = False  # no pre-allocated buffers in test harness
        if npu_causal_atten_mask is not None:
            # NPU cascade execution reads this for the causal STD/suffix calls.
            builder.npu_causal_atten_mask = npu_causal_atten_mask

        return builder.build(
            common_prefix_len=inputs.shared_tokens_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=False,
        )
    finally:
        BEAM_DATA_VAR.reset(token)


def run_cascade_beam_attention(
    inputs: BeamAttnInputs,
    device: torch.device | str,
    vllm_flash_attn_version: int,
    logits_soft_cap: float | None = None,
    npu_causal_atten_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Runs the real BeamAttentionImpl.cascade_attention backend under test.

    Shared by both device tests; only ``vllm_flash_attn_version``,
    ``logits_soft_cap`` and (NPU-only) ``npu_causal_atten_mask`` differ.
    """
    output = torch.zeros_like(inputs.query)
    attn_metadata = build_beam_attn_metadata(
        inputs, device, npu_causal_atten_mask=npu_causal_atten_mask
    )

    impl = BeamAttentionImpl(
        num_heads=inputs.query.shape[1],
        head_size=inputs.head_size,
        scale=inputs.scale,
        num_kv_heads=inputs.num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=logits_soft_cap,
    )
    impl.vllm_flash_attn_version = vllm_flash_attn_version

    BeamAttentionImpl.cascade_attention(
        output=output,
        query=inputs.query,
        key_cache=inputs.key_cache,
        value_cache=inputs.value_cache,
        attn_metadata=attn_metadata,
        sliding_window_size=impl.sliding_window,
        scale=impl.scale,
        logits_soft_cap=impl.logits_soft_cap,
        vllm_flash_attn_version=impl.vllm_flash_attn_version,
        alibi_slopes=impl.alibi_slopes,
    )
    return output
