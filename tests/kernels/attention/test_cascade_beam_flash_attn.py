# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm_gr.v1.engine.engine_core_patch import _compute_beam_bounds
except ImportError:
    pytest.skip(
        "beam_attn backend is not supported.",
        allow_module_level=True,
    )

try:
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        flash_attn_varlen_func,
        is_fa_version_supported,
    )
except ImportError:
    pytest.skip(
        "vllm_flash_attn is not supported.",
        allow_module_level=True,
    )

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.float16, torch.bfloat16]

CASES = [
    # Scenario: Standard Prefix Complete.
    # The shared prefix is fully cached, and beams independently decode their full suffixes.
    (8, 32, 8),
    # Scenario: No Shared Prefix.
    # Completely independent parallel sequence generation where shared_tokens_len = 0.
    (8, 8, 8),
    # Scenario: Large sequence general case with longer prefix.
    (4, 64, 4),
    # Scenario: General small context generation.
    (4, 16, 4),
    # Scenario: Chunked execution with strict budget (Partially Executed / Skipped beams).
    # Budget (1*16=16) runs out before all beams can process their 4-token suffixes.
    (1, 16, 4),
    # Scenario: Pure single-token decoding (Typical autoregressive decoding phase).
    # Each beam processes exactly 1 token over a large context.
    (1, 128, 1),
    # Scenario: Partial block-alignment anomaly.
    # Tests memory boundary safety when sequence lengths are prime/unaligned numbers.
    (4, 32, 7),
]


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
    soft_cap: float
    fa_version: int
    shared_tokens_len: int
    suffix_kv_len: int
    slot_mapping: torch.Tensor


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
    soft_cap: float | None,
    fa_version: int,
    suffix_kv_len: int = 0,
    device: str = "cuda",
) -> BeamAttnInputs:
    num_seqs = batch_size * beam_width
    scale = float(1.0 / (head_size**0.5))

    shared_tokens_len = kv_seq_len - suffix_kv_len
    beam_seq_len = shared_tokens_len + beam_width * suffix_kv_len
    blocks_per_req = (beam_seq_len + block_size - 1) // block_size

    num_reqs = batch_size

    # Query: [num_seqs * q_seq_len, num_heads, head_size]
    query = (
        torch.randn(num_seqs * q_seq_len, num_heads, head_size, dtype=dtype, device=device) * scale
    )

    # KV Cache
    total_blocks = num_reqs * blocks_per_req

    key_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )
    value_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )

    # Block Tables
    block_tables = torch.zeros((num_reqs, blocks_per_req), dtype=torch.int32, device=device)

    current_block = 0
    for i in range(num_reqs):
        block_tables[i] = torch.arange(
            current_block, current_block + blocks_per_req, dtype=torch.int32, device=device
        )
        current_block += blocks_per_req

    seq_lens_list = []
    for i in range(num_reqs):
        seq_lens_list.append(beam_seq_len)

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    # Calculate slot_mapping using the full block table slice approach
    num_actual_tokens = num_seqs * q_seq_len
    slot_mapping = torch.zeros(num_actual_tokens, dtype=torch.long, device=device)

    chunk_budget = beam_width * q_seq_len
    cache_len = (shared_tokens_len // block_size) * block_size
    remaining_prefix = shared_tokens_len - cache_len
    total_uncached_req = remaining_prefix + beam_width * suffix_kv_len
    write_len = min(total_uncached_req, chunk_budget)

    for i in range(num_reqs):
        physical_slots = (
            block_tables[i].to(torch.long).unsqueeze(-1) * block_size
            + torch.arange(block_size, device=device)
        ).view(-1)
        req_slots = physical_slots[cache_len : cache_len + write_len]
        q_start = i * chunk_budget
        slot_mapping[q_start : q_start + write_len] = req_slots

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
        soft_cap=logits_soft_cap,
        fa_version=fa_version,
        shared_tokens_len=shared_tokens_len,
        suffix_kv_len=suffix_kv_len,
        slot_mapping=slot_mapping,
    )


def run_reference_attention(inputs: BeamAttnInputs) -> torch.Tensor:
    dense_k = []
    dense_v = []
    seqlens_k = []
    seqlens_q = []

    for i in range(inputs.batch_size):
        # Retrieve the flat layout of the key and value cache for this specific request
        phys_blocks = inputs.block_tables[i].cpu().tolist()
        k_seq = torch.cat([inputs.key_cache[b] for b in phys_blocks], dim=0)[: inputs.seq_lens[i]]
        v_seq = torch.cat([inputs.value_cache[b] for b in phys_blocks], dim=0)[: inputs.seq_lens[i]]

        prefix_len = inputs.shared_tokens_len
        steps = inputs.suffix_kv_len
        w = inputs.beam_width
        q_len = inputs.q_seq_len * inputs.beam_width

        b_cache_len = prefix_len // inputs.block_size * inputs.block_size
        chunk_budget = q_len

        for b in range(w):
            # Stop processing if we run out of token budget for this chunk
            if chunk_budget <= 0:
                break

            b_uncached, b_suffix_len, b_past_suffix, b_physical = _compute_beam_bounds(
                b, prefix_len, b_cache_len, steps, chunk_budget
            )

            # If there's no uncached tokens to compute for this beam, skip it
            if b_uncached <= 0:
                continue

            prefix_k = k_seq[:prefix_len]
            prefix_v = v_seq[:prefix_len]
            if b_cache_len <= prefix_len:
                b_suffix_start = b_cache_len
            else:
                b_suffix_start = prefix_len + b_physical * steps

            suffix_k = k_seq[b_suffix_start : b_suffix_start + b_suffix_len]
            suffix_v = v_seq[b_suffix_start : b_suffix_start + b_suffix_len]

            # Concatenate prefix and suffix to form the contiguous beam key and value tensor
            beam_k = torch.cat([prefix_k, suffix_k], dim=0)
            beam_v = torch.cat([prefix_v, suffix_v], dim=0)

            dense_k.append(beam_k)
            dense_v.append(beam_v)
            seqlens_k.append(beam_k.shape[0])
            seqlens_q.append(b_uncached)

            # Deduct the processed tokens from the chunk budget
            chunk_budget -= b_uncached
            b_cache_len += b_uncached

    k_tensor = torch.cat(dense_k, dim=0)
    v_tensor = torch.cat(dense_v, dim=0)

    cu_seqlens_k = (
        torch.tensor([0] + seqlens_k, dtype=torch.int32, device=inputs.query.device)
        .cumsum(dim=0)
        .to(torch.int32)
    )
    cu_seqlens_q = (
        torch.tensor([0] + seqlens_q, dtype=torch.int32, device=inputs.query.device)
        .cumsum(dim=0)
        .to(torch.int32)
    )

    output = torch.zeros_like(inputs.query)

    def run_kernel() -> None:
        flash_attn_varlen_func(
            q=inputs.query[: cu_seqlens_q[-1]],
            k=k_tensor,
            v=v_tensor,
            out=output[: cu_seqlens_q[-1]],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            softmax_scale=inputs.scale,
            causal=True,
            window_size=(-1, -1),
            softcap=inputs.soft_cap,
            fa_version=inputs.fa_version,
        )

    run_kernel()
    torch.cuda.synchronize()
    return output


def run_cascade_beam_attention(
    inputs: BeamAttnInputs, shuffle_processing_order: bool = False
) -> torch.Tensor:
    output = torch.zeros_like(inputs.query)
    device = inputs.query.device

    from vllm_gr.v1.attention.backends.beam_attn import (
        BEAM_DATA_VAR,
        BeamAttentionImpl,
        BeamAttentionMetadataBuilder,
    )

    class MockCommonAttentionMetadata:
        def __init__(self, inputs: BeamAttnInputs):
            self.num_reqs = inputs.num_reqs

            # If shuffle_processing_order is True, we can simulate out-of-order execution
            # by reversing the query grouping lengths. But since we need them to match the
            # input shapes exactly, we just keep the sequential layout for queries.

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

    common_attn_metadata = MockCommonAttentionMetadata(inputs)

    beam_info: dict[str, Any] = {
        "beam_data": {},
        "req_ids": [f"req_{i}" for i in range(inputs.num_reqs)],
    }

    beam_data = cast(dict[str, Any], beam_info["beam_data"])
    for i in range(inputs.num_reqs):
        beam_data[f"req_{i}"] = {
            "is_beam_decode": True,
            "prefix_len": inputs.shared_tokens_len,
            "beam_beam_width": inputs.beam_width,
            "beam_decode_steps": inputs.suffix_kv_len,
            "cache_len": inputs.shared_tokens_len // inputs.block_size * inputs.block_size,
        }

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
        builder.num_heads_kv = inputs.num_kv_heads

        attn_metadata = builder.build(
            common_prefix_len=inputs.shared_tokens_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=False,
        )
    finally:
        BEAM_DATA_VAR.reset(token)

    impl = BeamAttentionImpl(
        num_heads=inputs.query.shape[1],
        head_size=inputs.head_size,
        scale=inputs.scale,
        num_kv_heads=inputs.num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=inputs.soft_cap,
    )

    impl.vllm_flash_attn_version = inputs.fa_version

    def run_kernel() -> None:
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

    run_kernel()
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("beam_width", [16, 32])  # type: ignore[misc]
@pytest.mark.parametrize("seq_lens_and_common_prefix", CASES)  # type: ignore[misc]
@pytest.mark.parametrize("num_heads", NUM_HEADS)  # type: ignore[misc]
@pytest.mark.parametrize("head_size", HEAD_SIZES)  # type: ignore[misc]
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@pytest.mark.parametrize("block_size", BLOCK_SIZES)  # type: ignore[misc]
@pytest.mark.parametrize("soft_cap", [None, 50])  # type: ignore[misc]
@pytest.mark.parametrize("fa_version", [2, 3])  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_beam_cascade(
    batch_size: int,
    beam_width: int,
    seq_lens_and_common_prefix: tuple[int, int, int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float,
    fa_version: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")
    device = torch.device("cuda")

    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    set_random_seed(0)

    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    assert num_kv_heads != 0
    q_seq_len = seq_lens_and_common_prefix[0]
    kv_seq_len = seq_lens_and_common_prefix[1]
    suffix_kv_len = seq_lens_and_common_prefix[2]
    assert suffix_kv_len > 0

    inputs = prepare_inputs(
        batch_size=batch_size,
        beam_width=beam_width,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_size=block_size,
        dtype=dtype,
        soft_cap=soft_cap,
        fa_version=fa_version,
        suffix_kv_len=suffix_kv_len,
        device=device,
    )

    # Run the regular attention.
    ref_output = run_reference_attention(inputs)
    # Run beam attention
    output = run_cascade_beam_attention(inputs)
    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [8])  # type: ignore[misc]
@pytest.mark.parametrize("beam_width", [16, 32, 128])  # type: ignore[misc]
@pytest.mark.parametrize("seq_lens_and_common_prefix", [(8, 2048, 8)])  # type: ignore[misc]
@pytest.mark.parametrize("num_heads", NUM_HEADS)  # type: ignore[misc]
@pytest.mark.parametrize("head_size", HEAD_SIZES)  # type: ignore[misc]
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@pytest.mark.parametrize("block_size", BLOCK_SIZES)  # type: ignore[misc]
@pytest.mark.parametrize("soft_cap", [None, 50])  # type: ignore[misc]
@pytest.mark.parametrize("fa_version", [2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("iter_num", [100])  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_beam_cascade_perf(
    batch_size: int,
    beam_width: int,
    seq_lens_and_common_prefix: tuple[int, int, int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float,
    fa_version: int,
    iter_num: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")
    device = torch.device("cuda")

    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    set_random_seed(0)

    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    assert num_kv_heads != 0
    q_seq_len = seq_lens_and_common_prefix[0]
    kv_seq_len = seq_lens_and_common_prefix[1]
    suffix_kv_len = seq_lens_and_common_prefix[2]
    assert suffix_kv_len > 0

    inputs = prepare_inputs(
        batch_size=batch_size,
        beam_width=beam_width,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_size=block_size,
        dtype=dtype,
        soft_cap=soft_cap,
        fa_version=fa_version,
        suffix_kv_len=suffix_kv_len,
        device=device,
    )

    # Warmup
    for _ in range(5):
        _ = run_cascade_beam_attention(inputs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iter_num):
        _ = run_cascade_beam_attention(inputs)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event) / iter_num
    print(
        f"\nPerf: batch_size={batch_size}, beam_width={beam_width}, "
        f"seq_lens={seq_lens_and_common_prefix}, num_heads={num_heads}, "
        f"head_size={head_size}, dtype={dtype}, block_size={block_size}, "
        f"avg_time={elapsed_time_ms:.3f} ms"
    )
