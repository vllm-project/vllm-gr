# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
from types import SimpleNamespace
from vllm.utils.torch_utils import set_random_seed

from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

try:
    from cascade_beam_attn_utils import (
        GPU_CASES,
        NUM_HEADS,
        BeamAttnInputs,
        build_beam_attn_metadata,
        iter_beam_segments,
        prepare_inputs,
        run_cascade_beam_attention,
        run_incremental_reference,
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


def run_reference_attention(inputs: BeamAttnInputs, cache_len: int | None = None) -> torch.Tensor:
    dense_k = []
    dense_v = []
    seqlens_k = []
    seqlens_q = []

    for _, b_uncached, beam_k, beam_v in iter_beam_segments(inputs, cache_len=cache_len):
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
    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("beam_width", [1, 128])  # type: ignore[misc]
@pytest.mark.parametrize("suffix_len", [1, 2])  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_incremental_pool_matches_dense_on_onerec_shape(
    beam_width: int,
    suffix_len: int,
) -> None:
    """Cover the production paged-prefix + persistent-pool suffix path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")
    if not is_fa_version_supported(2):
        pytest.skip(f"Flash attention 2 unsupported: {fa_version_unsupported_reason(2)}")

    from vllm_gr.v1.beam import BeamAttentionContext, BeamSearchSession

    device = torch.device("cuda")
    prefix_len = 256
    set_random_seed(0)
    inputs = prepare_inputs(
        batch_size=1,
        beam_width=beam_width,
        num_heads=16,
        num_kv_heads=8,
        head_size=128,
        q_seq_len=1,
        kv_seq_len=prefix_len + suffix_len,
        suffix_kv_len=suffix_len,
        block_size=32,
        dtype=torch.bfloat16,
        soft_cap=None,
        fa_version=2,
        device=device,
    )

    context = BeamAttentionContext(
        max_batch=1,
        max_beams=beam_width,
        num_kv_heads=8,
        max_decode_steps=3,
        head_dim=128,
        num_layers=1,
        device=device,
        dtype=torch.bfloat16,
    )
    slot = context.registry.allocate("request", beam_width=0, prefix_len=0)
    session = BeamSearchSession("request", slot, context)
    session.begin_session(beam_width, prefix_len)
    session._state.decode_step = suffix_len - 1
    context.decode_step_buf.fill_(suffix_len - 1)

    physical_k = inputs.key_cache[inputs.block_tables[0]].flatten(0, 1)
    physical_v = inputs.value_cache[inputs.block_tables[0]].flatten(0, 1)
    pool_k = physical_k[prefix_len : prefix_len + beam_width * suffix_len]
    pool_v = physical_v[prefix_len : prefix_len + beam_width * suffix_len]
    pool = context.pools[0]
    pool.unshared_key[0, :beam_width, :, :suffix_len, :] = pool_k.reshape(
        beam_width, suffix_len, 8, 128
    ).permute(0, 2, 1, 3)
    pool.unshared_value[0, :beam_width, :, :suffix_len, :] = pool_v.reshape(
        beam_width, suffix_len, 8, 128
    ).permute(0, 2, 1, 3)

    # Build metadata WITH the active session and per-request decode_step so
    # the builder takes the incremental path (one query per beam, decode_step+1
    # suffix KV rows) instead of the recompute path.  Patching these fields
    # after build() only switches the execution branch while leaving the
    # recompute query/KV boundaries intact.
    reference = run_incremental_reference(inputs)
    metadata = build_beam_attn_metadata(inputs, device, beam_session=session,
                                        decode_step=suffix_len - 1)
    output = torch.zeros_like(inputs.query)
    BeamAttentionImpl.cascade_attention(
        output=output,
        query=inputs.query,
        key_cache=inputs.key_cache,
        value_cache=inputs.value_cache,
        attn_metadata=metadata,
        sliding_window_size=(-1, -1),
        scale=inputs.scale,
        logits_soft_cap=0,
        vllm_flash_attn_version=2,
        beam_pool=pool,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("beam_width", [2, 4])  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_gather_path_past_current_suffix_projection(
    beam_width: int,
) -> None:
    """Output-level regression for the cached/current suffix gather path.

    Builds a chunked-prefill step where some suffix KV is already cached
    (``b_past_suffix > 0``) — the builder must project those tokens through
    the block table while projecting the current chunk's tokens through
    ``slot_mapping``.  No ``beam_session`` is provided, so
    ``cascade_attention`` takes the gather path (``extract_suffix_kv`` with
    ``suffix_slot_mapping_out``) instead of the pool path.

    The cached-suffix case would fail with the old mapping (all suffix tokens
    treated as query offsets via ``np.arange(out_slot, ...)``) and must pass
    with this PR's projection fix.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")
    if not is_fa_version_supported(2):
        pytest.skip(f"Flash attention 2 unsupported: {fa_version_unsupported_reason(2)}")

    block_size = 16
    prefix_len = 16
    suffix_len = 4
    q_seq_len = 2
    cache_len = 18  # 2 already-cached suffix tokens → b_past_suffix > 0
    device = torch.device("cuda")
    num_heads = 4
    num_kv_heads = 2
    head_size = 128
    dtype = torch.bfloat16
    set_random_seed(0)

    beam_seq_len = prefix_len + beam_width * suffix_len
    blocks_per_req = (beam_seq_len + block_size - 1) // block_size
    scale = float(1.0 / (head_size ** 0.5))
    num_actual_tokens = beam_width * q_seq_len

    query = (
        torch.randn(num_actual_tokens, num_heads, head_size, dtype=dtype, device=device)
        * scale
    )
    # Non-contiguous physical blocks (gap=3) so past/current projection is
    # distinguishable from a trivial contiguous arange.
    block_tables = torch.zeros((1, blocks_per_req), dtype=torch.int32, device=device)
    for i in range(blocks_per_req):
        block_tables[0, i] = i * 3
    cache_blocks = int(block_tables.max().item()) + block_size
    key_cache = (
        torch.randn(cache_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )
    value_cache = (
        torch.randn(cache_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )

    seq_lens = torch.tensor([beam_seq_len], dtype=torch.int32, device=device)
    query_start_loc_cpu = torch.tensor([0, num_actual_tokens], dtype=torch.int32)
    all_physical_slots = (
        block_tables.to(torch.long).unsqueeze(-1) * block_size
        + torch.arange(block_size, device=device)
    ).view(-1)
    slot_mapping = torch.zeros(num_actual_tokens, dtype=torch.long, device=device)
    chunk_budget = q_seq_len * beam_width
    write_len = min(chunk_budget, beam_seq_len - (prefix_len // block_size * block_size))
    slot_mapping[:write_len] = all_physical_slots[cache_len:cache_len + write_len]

    # Dense reference via iter_beam_segments with the same cache_len.
    inputs_source = SimpleNamespace(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        num_reqs=1,
        num_seqs=beam_width,
        max_seq_len=beam_seq_len,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        batch_size=1,
        beam_width=beam_width,
        q_seq_len=q_seq_len,
        head_size=head_size,
        shared_tokens_len=prefix_len,
        suffix_kv_len=suffix_len,
        slot_mapping=slot_mapping,
        soft_cap=0.0,
        fa_version=2,
    )
    reference = run_reference_attention(inputs_source, cache_len=cache_len)

    # Build metadata (recompute/gather path — no beam_session).
    from vllm_gr.v1.attention.backends.beam_attn_metadata import (
        BeamAttentionMetadataBuilder,
    )
    from vllm_gr.v1.beam.context import BEAM_DATA_VAR

    beam_info = {
        "beam_data": {
            "req_0": {
                "is_beam_decode": True,
                "prefix_len": prefix_len,
                "beam_width": beam_width,
                "beam_decode_steps": suffix_len,
                "cache_len": cache_len,
            },
        },
        "req_ids": ["req_0"],
    }
    token = BEAM_DATA_VAR.set(beam_info)
    try:
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
        builder._is_npu = False
        builder._fixed_buffers = False
        builder.npu_causal_atten_mask = None
        common = SimpleNamespace(
            num_reqs=1,
            query_start_loc_cpu=query_start_loc_cpu,
            query_start_loc=query_start_loc_cpu.to(device),
            seq_lens=seq_lens,
            block_table_tensor=block_tables,
            num_actual_tokens=num_actual_tokens,
            max_query_len=chunk_budget,
            max_seq_len=beam_seq_len,
            causal=True,
            slot_mapping=slot_mapping,
        )
        metadata = builder.build(
            common_prefix_len=prefix_len,
            common_attn_metadata=common,
            fast_build=False,
        )
    finally:
        BEAM_DATA_VAR.reset(token)

    has_past = metadata.suffix_past_coords is not None and metadata.suffix_past_coords.numel() > 0
    has_curr = metadata.suffix_curr_coords is not None and metadata.suffix_curr_coords.numel() > 0
    assert has_past and has_curr, (
        "Test needs both cached-suffix (past) and current-query coordinates. "
        f"past={has_past} curr={has_curr}"
    )

    output = torch.zeros_like(query)
    BeamAttentionImpl.cascade_attention(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=metadata,
        sliding_window_size=(-1, -1),
        scale=scale,
        logits_soft_cap=0,
        vllm_flash_attn_version=2,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-5)

    # Repeat with a different block table to exercise update_block_table.
    bt2 = torch.zeros((1, blocks_per_req), dtype=torch.int32, device=device)
    base = int(block_tables.max().item()) + 5
    for i in range(blocks_per_req):
        bt2[0, i] = base + i * 3
    for i in range(blocks_per_req):
        orig = int(block_tables[0, i].item())
        dst = int(bt2[0, i].item())
        key_cache[dst] = key_cache[orig].clone()
        value_cache[dst] = value_cache[orig].clone()
    new_slots = (
        bt2.to(torch.long).unsqueeze(-1) * block_size
        + torch.arange(block_size, device=device)
    ).view(-1)
    new_slot_mapping = torch.zeros(num_actual_tokens, dtype=torch.long, device=device)
    new_slot_mapping[:write_len] = new_slots[cache_len:cache_len + write_len]

    m2 = builder.update_block_table(metadata, bt2, new_slot_mapping)
    out2 = torch.zeros_like(query)
    BeamAttentionImpl.cascade_attention(
        output=out2,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=m2,
        sliding_window_size=(-1, -1),
        scale=scale,
        logits_soft_cap=0,
        vllm_flash_attn_version=2,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(out2, reference, atol=1e-3, rtol=1e-5)
