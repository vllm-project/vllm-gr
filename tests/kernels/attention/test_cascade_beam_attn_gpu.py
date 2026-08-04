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
    torch.cuda.synchronize()
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
    torch.cuda.synchronize()
    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-5)