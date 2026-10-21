# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
from vllm.utils.torch_utils import set_random_seed

try:
    from cascade_beam_attn_utils import (
        GPU_CASES,
        NUM_HEADS,
        BeamAttnInputs,
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
    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("graph_replay", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_incremental_forward_append_reorder_matches_dense(
    graph_replay: bool, dtype: torch.dtype
) -> None:
    """Cache writes and repeated-parent reorder agree with independent dense history."""
    from types import SimpleNamespace

    from cascade_beam_attn_utils import build_beam_attn_metadata

    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

    if not torch.cuda.is_available() or not is_fa_version_supported(2):
        pytest.skip("CUDA FlashAttention 2 is required")
    torch.manual_seed(81)
    width, heads, kv_heads, dim, steps = 4, 8, 2, 128, 3
    inputs = prepare_inputs(
        1,
        width,
        heads,
        kv_heads,
        dim,
        1,
        22,
        16,
        dtype,
        suffix_kv_len=steps,
        device="cuda",
        fa_version=2,
    )
    session = inputs.session
    session._state.decode_step = 0
    session.context.decode_step_buf.zero_()
    pool = session.pools[0]
    pool.unshared_key.fill_(float("nan"))
    pool.unshared_value.fill_(float("nan"))
    metadata = build_beam_attn_metadata(inputs, "cuda")
    query = torch.randn_like(inputs.query)
    key = torch.randn(width, kv_heads, dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    kv_cache = torch.stack((inputs.key_cache, inputs.value_cache))
    saved_prefix = kv_cache.clone()
    impl = BeamAttentionImpl(heads, dim, inputs.scale, kv_heads, None, None, "auto")
    impl.vllm_flash_attn_version = 2
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn")

    def forward() -> None:
        impl.forward(layer, query, key, value, kv_cache, metadata, output=output)

    if graph_replay:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                forward()
            with torch.cuda.graph(graph, stream=capture_stream):
                forward()
        torch.cuda.current_stream().wait_stream(capture_stream)

    prefix_k = inputs.key_cache[inputs.block_tables[0].long()].flatten(0, 1)[:19]
    prefix_v = inputs.value_cache[inputs.block_tables[0].long()].flatten(0, 1)[:19]
    history_k = key.new_empty(width, 0, kv_heads, dim)
    history_v = value.new_empty(width, 0, kv_heads, dim)
    parents = torch.tensor([1, 1, 3, 0], device="cuda", dtype=torch.long)
    for step in range(steps):
        assert session.decode_step == step
        query.copy_(torch.randn_like(query))
        key.copy_(torch.randn_like(key))
        value.copy_(torch.randn_like(value))
        session.context.decode_step_buf.fill_(step)
        metadata.suffix_cu_seqlens_k_device.copy_(
            torch.arange(width + 1, device="cuda", dtype=torch.int32) * (step + 1)
        )
        history_k = torch.cat((history_k, key[:, None].clone()), dim=1)
        history_v = torch.cat((history_v, value[:, None].clone()), dim=1)
        if graph_replay:
            graph.replay()
        else:
            forward()
        dense_k = torch.cat((prefix_k[None].expand(width, -1, -1, -1), history_k), dim=1)
        dense_v = torch.cat((prefix_v[None].expand(width, -1, -1, -1), history_v), dim=1)
        dense_k = dense_k.float().repeat_interleave(heads // kv_heads, dim=2)
        dense_v = dense_v.float().repeat_interleave(heads // kv_heads, dim=2)
        weights = torch.einsum("whd,wlhd->whl", query.float(), dense_k) * inputs.scale
        expected = torch.einsum("whl,wlhd->whd", weights.softmax(-1), dense_v)
        tolerance = 4e-3 if dtype == torch.bfloat16 else 5e-4
        torch.testing.assert_close(output.float(), expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(kv_cache, saved_prefix, atol=0, rtol=0)
        if step + 1 < steps:
            session.step_complete(parents)
            history_k = history_k[parents].clone()
            history_v = history_v[parents].clone()
