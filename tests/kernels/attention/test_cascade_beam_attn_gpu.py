# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from types import SimpleNamespace

import pytest
import torch
from vllm.utils.torch_utils import set_random_seed

try:
    from cascade_beam_attn_utils import (
        BEAM_DATA_VAR,
        GPU_CASES,
        NUM_HEADS,
        BeamAttentionImpl,
        BeamAttentionMetadataBuilder,
        BeamAttnInputs,
        _compute_beam_bounds,
        iter_beam_segments,
        prepare_inputs,
        run_cascade_beam_attention,
    )
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

HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.float16, torch.bfloat16]


def run_reference_attention(inputs: BeamAttnInputs) -> torch.Tensor:
    dense_k = []
    dense_v = []
    seqlens_k = []
    seqlens_q = []

    for _, b_uncached, beam_k, beam_v in iter_beam_segments(inputs):
        dense_k.append(beam_k)
        dense_v.append(beam_v)
        seqlens_k.append(beam_k.shape[0])
        seqlens_q.append(b_uncached)

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
    torch.accelerator.synchronize()
    return output


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("beam_width", [16, 32])  # type: ignore[misc]
@pytest.mark.parametrize("seq_lens_and_common_prefix", GPU_CASES)  # type: ignore[misc]
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
    output = run_cascade_beam_attention(
        inputs,
        device=device,
        vllm_flash_attn_version=inputs.fa_version,
        logits_soft_cap=inputs.soft_cap,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-5)


@torch.inference_mode()  # type: ignore[misc]
def test_beam_cascade_two_requests_with_unequal_progress() -> None:
    """Compare packed requests at different points in scheduler chunking.

    The broad parameter sweep above gives every request identical prefix and
    cache progress. This case instead makes request 0 resume three tokens into
    a beam suffix while request 1 starts at its suffix boundary. Their KV block
    tables are disjoint and non-contiguous so request-relative slot aliases
    cannot accidentally read the expected history.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")

    fa_version = 3 if is_fa_version_supported(3) else 2
    if not is_fa_version_supported(fa_version):
        pytest.skip(f"Flash attention is unsupported: {fa_version_unsupported_reason(fa_version)}")

    set_random_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    block_size = 16
    beam_width = 32
    decode_steps = 7
    prefix_lens = [31, 48]
    cache_lens = [prefix_lens[0] + 3, prefix_lens[1]]
    chunk_budgets = [71, 53]
    seq_lens_cpu = [
        cache_len + chunk_budget for cache_len, chunk_budget in zip(cache_lens, chunk_budgets)
    ]
    q_starts_cpu = [0, chunk_budgets[0], sum(chunk_budgets)]

    num_heads = 8
    num_kv_heads = 2
    head_size = 128
    scale = float(1.0 / (head_size**0.5))

    max_logical_tokens = max(prefix_len + beam_width * decode_steps for prefix_len in prefix_lens)
    blocks_per_req = (max_logical_tokens + block_size - 1) // block_size
    num_physical_blocks = blocks_per_req * len(prefix_lens)
    physical_blocks = torch.randperm(num_physical_blocks, device=device).to(torch.int32)
    block_table = physical_blocks.view(len(prefix_lens), blocks_per_req)

    key_cache = torch.randn(
        num_physical_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(
        sum(chunk_budgets),
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )

    slot_mapping_parts = []
    for req_idx, (cache_len, chunk_budget) in enumerate(zip(cache_lens, chunk_budgets)):
        logical_positions = torch.arange(
            cache_len,
            cache_len + chunk_budget,
            dtype=torch.long,
            device=device,
        )
        slots = (
            block_table[req_idx, logical_positions // block_size].to(torch.long) * block_size
            + logical_positions % block_size
        )
        slot_mapping_parts.append(slots)
    slot_mapping = torch.cat(slot_mapping_parts)

    query_start_loc_cpu = torch.tensor(q_starts_cpu, dtype=torch.int32)
    common_metadata = SimpleNamespace(
        num_reqs=len(prefix_lens),
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(device),
        seq_lens=torch.tensor(seq_lens_cpu, dtype=torch.int32, device=device),
        block_table_tensor=block_table,
        num_actual_tokens=sum(chunk_budgets),
        max_query_len=max(chunk_budgets),
        max_seq_len=max(seq_lens_cpu),
        causal=True,
        slot_mapping=slot_mapping,
    )

    beam_info = {
        "req_ids": ["resumed", "suffix-boundary"],
        "beam_data": {
            "resumed": {
                "is_beam_decode": True,
                "prefix_len": prefix_lens[0],
                "beam_width": beam_width,
                "beam_decode_steps": decode_steps,
                "cache_len": cache_lens[0],
            },
            "suffix-boundary": {
                "is_beam_decode": True,
                "prefix_len": prefix_lens[1],
                "beam_width": beam_width,
                "beam_decode_steps": decode_steps,
                "cache_len": cache_lens[1],
            },
        },
    }

    builder = BeamAttentionMetadataBuilder.__new__(BeamAttentionMetadataBuilder)
    builder.device = device
    builder.block_size = block_size
    builder.aot_schedule = False
    builder.aot_sliding_window = None
    builder.use_full_cuda_graph = False
    builder.max_cudagraph_size = None
    builder.max_num_splits = 0
    builder.num_heads_q = num_heads
    builder.num_kv_heads = num_kv_heads
    builder._is_cuda = True

    context_token = BEAM_DATA_VAR.set(beam_info)
    try:
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata,
            fast_build=False,
        )
    finally:
        BEAM_DATA_VAR.reset(context_token)

    output = torch.zeros_like(query)
    impl = BeamAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    BeamAttentionImpl.cascade_attention(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=attn_metadata,
        sliding_window_size=impl.sliding_window,
        scale=impl.scale,
        logits_soft_cap=impl.logits_soft_cap,
        vllm_flash_attn_version=fa_version,
        alibi_slopes=impl.alibi_slopes,
    )

    dense_k = []
    dense_v = []
    query_lens = []
    kv_lens = []
    for req_idx, (prefix_len, cache_len, chunk_budget, seq_len) in enumerate(
        zip(prefix_lens, cache_lens, chunk_budgets, seq_lens_cpu)
    ):
        logical_k = key_cache[block_table[req_idx]].flatten(0, 1)[:seq_len]
        logical_v = value_cache[block_table[req_idx]].flatten(0, 1)[:seq_len]
        remaining_budget = chunk_budget
        beam_cache_len = cache_len
        for beam_idx in range(beam_width):
            if remaining_budget <= 0:
                break
            uncached, suffix_len, _, physical_beam = _compute_beam_bounds(
                beam_idx,
                prefix_len,
                beam_cache_len,
                decode_steps,
                remaining_budget,
            )
            if uncached <= 0:
                continue
            suffix_start = prefix_len + physical_beam * decode_steps
            dense_k.append(
                torch.cat(
                    [
                        logical_k[:prefix_len],
                        logical_k[suffix_start : suffix_start + suffix_len],
                    ]
                )
            )
            dense_v.append(
                torch.cat(
                    [
                        logical_v[:prefix_len],
                        logical_v[suffix_start : suffix_start + suffix_len],
                    ]
                )
            )
            query_lens.append(uncached)
            kv_lens.append(prefix_len + suffix_len)
            beam_cache_len += uncached
            remaining_budget -= uncached

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32, device=device).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_kv_lens = torch.tensor([0] + kv_lens, dtype=torch.int32, device=device).cumsum(
        dim=0, dtype=torch.int32
    )
    reference = torch.zeros_like(query)
    flash_attn_varlen_func(
        q=query,
        k=torch.cat(dense_k),
        v=torch.cat(dense_v),
        out=reference,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        fa_version=fa_version,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-5)
