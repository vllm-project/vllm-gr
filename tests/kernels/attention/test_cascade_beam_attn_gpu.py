# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402


import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu

# Skip before importing CUDA-only dependencies or newer vLLM GPU APIs.
if os.environ.get("VLLM_GR_TARGET_DEVICE", "").lower() == "npu" or not torch.cuda.is_available():
    pytest.skip("CUDA is required for this test.", allow_module_level=True)

from vllm.config import CacheConfig, VllmConfig
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import record_kv_cache_layout
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_gr.v1.attention.backends.beam_attn import (
    BeamAttentionBackend,
    BeamAttentionImpl,
)
from vllm_gr.v1.attention.backends.beam_attn_gpu import reshape_and_cache
from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata
from vllm_gr.v1.beam import BeamAttentionContext, BeamSearchSession

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


def _native_layer_caches(
    layout: KVCacheLayout, dtype: torch.dtype, num_blocks: int
) -> dict[str, torch.Tensor]:
    """Use the engine's placement and worker allocation, with three real layers."""
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128, dtype=dtype)
    config = VllmConfig(cache_config=CacheConfig(block_size=16))
    record_kv_cache_layout(config.cache_config, layout.name)
    group = KVCacheGroupSpec(layer_names=["layer0", "layer1", "layer2"], kv_cache_spec=spec)
    cache_config = get_kv_cache_config_from_groups(
        config, [group], available_memory=3 * num_blocks * spec.page_size_bytes
    )
    assert cache_config.num_blocks == num_blocks
    caches: dict[str, torch.Tensor] = allocate_kv_cache(cache_config, torch.device("cuda"), layout)
    return caches


def _native_impl() -> BeamAttentionImpl:
    return BeamAttentionImpl(
        num_heads=4,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


@pytest.mark.parametrize("layout", BeamAttentionBackend.supported_kv_cache_layouts())  # type: ignore[misc]
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_native_kv_layout_cascade(layout: KVCacheLayout, dtype: torch.dtype) -> None:
    """Check native cache writes, nonzero layer offsets and paged suffix gathers."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    set_random_seed(17)
    impl = _native_impl()
    inputs = prepare_inputs(
        batch_size=2,
        beam_width=4,
        num_heads=4,
        num_kv_heads=2,
        head_size=128,
        q_seq_len=1,
        kv_seq_len=33,
        block_size=16,
        dtype=dtype,
        suffix_kv_len=3,
        fa_version=impl.vllm_flash_attn_version,
    )
    reference = run_reference_attention(inputs)
    caches = _native_layer_caches(layout, dtype, inputs.key_cache.shape[0])
    cache = caches["layer1"]
    key_view, value_view = impl._kv_cache_views(cache)
    assert key_view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert value_view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert key_view.stride() == value_view.stride()
    # Include one masked token and an extra padded input row: neither may
    # overwrite a valid page. The native writer uses slot_mapping's length.
    key = inputs.key_cache.flatten(0, 1)
    value = inputs.value_cache.flatten(0, 1)
    padding = torch.full_like(key[:2], 100)
    slots = torch.cat(
        (torch.arange(key.shape[0], device="cuda"), torch.tensor([-1], device="cuda"))
    )
    scale = torch.ones((), device="cuda")
    reshape_and_cache(
        torch.cat((key, padding)),
        torch.cat((value, padding)),
        key_view,
        value_view,
        slots,
        "auto",
        scale,
        scale,
    )
    # Check the upstream logical axes independently of the GR view helper.
    torch.testing.assert_close(cache[..., :128], inputs.key_cache.transpose(1, 2))
    torch.testing.assert_close(cache[..., 128:], inputs.value_cache.transpose(1, 2))
    assert not caches["layer0"].count_nonzero()
    assert not caches["layer2"].count_nonzero()
    inputs.key_cache, inputs.value_cache = key_view, value_view
    output = run_cascade_beam_attention(
        inputs,
        device="cuda",
        vllm_flash_attn_version=impl.vllm_flash_attn_version,
        logits_soft_cap=inputs.soft_cap,
    )
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("layout", BeamAttentionBackend.supported_kv_cache_layouts())  # type: ignore[misc]
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_native_kv_layout_prefill(layout: KVCacheLayout, dtype: torch.dtype) -> None:
    """Cached-prefix prefill crosses a block boundary through the public facade."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    set_random_seed(23)
    impl = _native_impl()
    caches = _native_layer_caches(layout, dtype, 3)
    cache = caches["layer1"]
    # Logical sequence uses physical block 2 followed by block 0.
    key = torch.randn(18, 2, 128, device="cuda", dtype=dtype) * 0.1
    value = torch.randn_like(key)
    query = torch.randn(3, 4, 128, device="cuda", dtype=dtype) * 0.1
    cache[2, :, :15, :128] = key[:15].transpose(0, 1)
    cache[2, :, :15, 128:] = value[:15].transpose(0, 1)
    metadata = BeamAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        max_seq_len=18,
        seq_lens=torch.tensor([18], dtype=torch.int32, device="cuda"),
        block_table=torch.tensor([[2, 0]], dtype=torch.int32, device="cuda"),
        slot_mapping=torch.tensor([47, 0, 1], device="cuda"),
        is_prefill_graph_exec=True,
    )
    scale = torch.ones((), device="cuda")
    layer = SimpleNamespace(_k_scale=scale, _v_scale=scale)
    output = torch.empty_like(query)
    impl.forward(layer, query, key[15:], value[15:], cache, metadata, output=output)
    reference = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=metadata.query_start_loc,
        cu_seqlens_k=torch.tensor([0, 18], dtype=torch.int32, device="cuda"),
        max_seqlen_q=3,
        max_seqlen_k=18,
        softmax_scale=impl.scale,
        causal=True,
        fa_version=impl.vllm_flash_attn_version,
    )
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-5)
    torch.testing.assert_close(cache[2, :, :16, :128], key[:16].transpose(0, 1))
    torch.testing.assert_close(cache[0, :, :2, :128], key[16:].transpose(0, 1))
    torch.testing.assert_close(cache[2, :, :16, 128:], value[:16].transpose(0, 1))
    torch.testing.assert_close(cache[0, :, :2, 128:], value[16:].transpose(0, 1))
    assert not caches["layer0"].count_nonzero()
    assert not caches["layer2"].count_nonzero()


@pytest.mark.parametrize("layout", BeamAttentionBackend.supported_kv_cache_layouts())  # type: ignore[misc]
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_native_kv_layout_incremental_pool(layout: KVCacheLayout, dtype: torch.dtype) -> None:
    """Combine a strided native prefix with the independent, nonzero suffix slot."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    set_random_seed(31)
    width, steps, prefix_len = 4, 3, 18
    impl = _native_impl()
    caches = _native_layer_caches(layout, dtype, 3)
    cache = caches["layer1"]
    prefix_k = torch.randn(prefix_len, 2, 128, device="cuda", dtype=dtype) * 0.1
    prefix_v = torch.randn_like(prefix_k) * 0.1
    for physical, start, end in ((2, 0, 16), (0, 16, 18)):
        cache[physical, :, : end - start, :128] = prefix_k[start:end].transpose(0, 1)
        cache[physical, :, : end - start, 128:] = prefix_v[start:end].transpose(0, 1)
    prefix_snapshot = cache.clone()
    context = BeamAttentionContext(
        max_batch=2,
        max_beams=width,
        num_kv_heads=2,
        max_decode_steps=steps,
        head_dim=128,
        num_layers=3,
        device=torch.device("cuda"),
        dtype=dtype,
    )
    context.registry.allocate("other", beam_width=0, prefix_len=0)
    slot = context.registry.allocate("active", beam_width=0, prefix_len=0)
    assert slot == 1
    session = BeamSearchSession("active", slot, context)
    session.begin_session(width, prefix_len)
    suffix_k = torch.randn(width, steps, 2, 128, device="cuda", dtype=dtype) * 0.1
    suffix_v = torch.randn_like(suffix_k) * 0.1
    query = torch.randn(width, 4, 128, device="cuda", dtype=dtype) * 0.1
    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    metadata = BeamAttentionMetadata(
        num_actual_tokens=width,
        max_query_len=width,
        query_start_loc=torch.tensor([0, width], dtype=torch.int32, device="cuda"),
        max_seq_len=prefix_len + steps,
        seq_lens=torch.tensor([prefix_len], dtype=torch.int32, device="cuda"),
        block_table=block_table,
        slot_mapping=torch.full((width,), -1, device="cuda"),
        is_beam_decode=True,
        beam_session=session,
        num_heads=4,
        num_kv_heads=2,
        prefix_indices=torch.arange(width, device="cuda"),
        num_prefix_indices=width,
        prefix_cu_seqlens_q_device=torch.tensor([0, width], dtype=torch.int32, device="cuda"),
        prefix_seqlens_kv_device=torch.tensor([prefix_len], dtype=torch.int32, device="cuda"),
        prefix_max_q_len=width,
        prefix_max_seq_len=prefix_len,
        prefix_block_table=block_table,
        suffix_cu_seqlens_q_device=torch.arange(width + 1, dtype=torch.int32, device="cuda"),
        suffix_max_q_len=1,
    )
    layer = SimpleNamespace(layer_name="model.layers.1.self_attn")
    output = torch.empty_like(query)
    for step in range(steps):
        context.decode_step_buf.fill_(step)
        metadata.suffix_cu_seqlens_k_device = metadata.suffix_cu_seqlens_q_device * (step + 1)
        impl.forward(
            layer,
            query,
            suffix_k[:, step].contiguous(),
            suffix_v[:, step].contiguous(),
            cache,
            metadata,
            output=output,
        )
        # Independent dense FP32 attention over prefix plus each beam's suffix.
        dense_k = torch.cat((prefix_k.expand(width, -1, -1, -1), suffix_k[:, : step + 1]), 1)
        dense_v = torch.cat((prefix_v.expand(width, -1, -1, -1), suffix_v[:, : step + 1]), 1)
        dense_k = dense_k.float().repeat_interleave(2, dim=2)
        dense_v = dense_v.float().repeat_interleave(2, dim=2)
        scores = torch.einsum("bhd,bthd->bht", query.float(), dense_k) * impl.scale
        reference = torch.einsum("bht,bthd->bhd", scores.softmax(-1), dense_v)
        torch.testing.assert_close(output.float(), reference, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            context.pools[1].unshared_key[slot, :, :, step], suffix_k[:, step]
        )
        torch.testing.assert_close(
            context.pools[1].unshared_value[slot, :, :, step], suffix_v[:, step]
        )
        torch.testing.assert_close(cache, prefix_snapshot)
    assert not caches["layer0"].count_nonzero()
    assert not caches["layer2"].count_nonzero()
    assert not context.pools[1].unshared_key[0].count_nonzero()
    session.end_session()
