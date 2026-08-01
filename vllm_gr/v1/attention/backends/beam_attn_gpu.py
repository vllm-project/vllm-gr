# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU execution primitives for the Beam Attention backend."""

from __future__ import annotations

from typing import Any, NoReturn, cast

import torch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version as _get_flash_attn_version,
)
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from vllm_gr.v1.attention.backends.beam_attn_metadata_gpu import (
    CUDABeamAttentionMetadata,
)
from vllm_gr.v1.attention.backends.beam_attn_triton import extract_suffix_kv

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
else:

    def _flash_attn_varlen_unavailable(
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        raise RuntimeError("BeamAttentionBackend requires FlashAttention varlen support")

    flash_attn_varlen_func = _flash_attn_varlen_unavailable
    reshape_and_cache_flash = _flash_attn_varlen_unavailable
    merge_attn_states = _flash_attn_varlen_unavailable
    flash_attn_supports_sinks = _flash_attn_varlen_unavailable


def is_available() -> bool:
    return bool(is_flash_attn_varlen_func_available())


def get_flash_attn_version(
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
) -> int | None:
    return cast(
        int | None,
        _get_flash_attn_version(
            requires_alibi=requires_alibi,
            head_size=head_size,
        ),
    )


def supports_sinks() -> bool:
    return bool(is_available() and flash_attn_supports_sinks())


def supports_batch_invariance() -> bool:
    return True


def supports_kv_cache_dtype(kv_cache_dtype: Any) -> bool:
    if kv_cache_dtype is None:
        return True
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return get_flash_attn_version() == 3 and current_platform.is_device_capability_family(90)
    return kv_cache_dtype in ("auto", "float16", "bfloat16")


def supports_compute_capability(capability: DeviceCapability) -> bool:
    return capability >= DeviceCapability(8, 0)


def validate_combination(
    *,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Any,
    block_size: int | None,
    use_mla: bool,
    has_sink: bool,
    use_sparse: bool,
    device_capability: DeviceCapability,
) -> str | None:
    if not is_available():
        return "No FlashAttention varlen support."
    if dtype not in (torch.float16, torch.bfloat16):
        return f"Unsupported dtype {dtype}."
    if head_size % 8 != 0 or head_size > 256:
        return f"Unsupported head_size={head_size}."
    if not supports_kv_cache_dtype(kv_cache_dtype):
        return f"Unsupported kv_cache_dtype={kv_cache_dtype}."
    if block_size is not None and block_size % 16 != 0:
        return f"block_size={block_size} not multiple of 16."
    if not supports_compute_capability(device_capability):
        return "Requires GPU compute capability >= 8.0."
    if has_sink and not supports_sinks():
        return "No FlashAttention sink support."
    if use_mla:
        return "No MLA support."
    if use_sparse:
        return "No sparse support."
    return None


def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
) -> tuple[int, ...]:
    del cache_dtype_str
    return (2, num_blocks, block_size, num_kv_heads, head_size)


def get_kv_cache_stride_order(
    include_num_layers_dimension: bool = False,
) -> tuple[int, ...]:
    cache_layout = get_kv_cache_layout()
    if cache_layout == "NHD" and include_num_layers_dimension:
        return (2, 0, 1, 3, 4, 5)
    if cache_layout == "NHD":
        return (0, 1, 2, 3, 4)
    if cache_layout == "HND" and include_num_layers_dimension:
        return (2, 4, 0, 1, 3, 5)
    if cache_layout == "HND":
        return (0, 1, 3, 2, 4)
    raise ValueError(f"Unknown cache layout format {cache_layout}.")


def split_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache, value_cache = kv_cache.unbind(0)
    return key_cache, value_cache


def get_kv_scales(layer: Any) -> tuple[Any, Any]:
    return layer._k_scale, layer._v_scale


def get_fp8_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


def get_metadata_builder_cls() -> type[Any]:
    from vllm_gr.v1.attention.backends.beam_attn_metadata_gpu import (
        CUDABeamAttentionMetadataBuilder,
    )

    return CUDABeamAttentionMetadataBuilder


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: CUDABeamAttentionMetadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> None:
    if attn_metadata.std_indices is not None:
        if attn_metadata.prefix_indices is None:
            std_query = query[: attn_metadata.num_std_indices]
        else:
            std_query = query[attn_metadata.std_indices]
        std_out = flash_attn_varlen_func(
            q=std_query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.std_cu_seqlens_q_device,
            max_seqlen_q=attn_metadata.std_max_q_len,
            seqused_k=attn_metadata.std_seqlens_kv_device,
            max_seqlen_k=attn_metadata.std_max_seq_len,
            softmax_scale=scale,
            causal=attn_metadata.causal,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=attn_metadata.std_block_table,
            softcap=logits_soft_cap,
            fa_version=vllm_flash_attn_version,
        )
        if attn_metadata.prefix_indices is None:
            output[: attn_metadata.num_std_indices] = std_out
        else:
            output[attn_metadata.std_indices] = std_out

    if attn_metadata.prefix_indices is not None:
        if attn_metadata.std_indices is None:
            beam_query = query[: attn_metadata.num_prefix_indices]
        else:
            beam_query = query[attn_metadata.prefix_indices]

        prefix_out, prefix_lse = flash_attn_varlen_func(
            q=beam_query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.prefix_cu_seqlens_q_device,
            seqused_k=attn_metadata.prefix_seqlens_kv_device,
            max_seqlen_q=attn_metadata.prefix_max_q_len,
            max_seqlen_k=attn_metadata.prefix_max_seq_len,
            softmax_scale=scale,
            causal=False,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=attn_metadata.prefix_block_table,
            softcap=logits_soft_cap,
            return_softmax_lse=True,
            fa_version=vllm_flash_attn_version,
            scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            num_splits=attn_metadata.max_num_splits,
            s_aux=s_aux,
        )

        contig_k, contig_v = extract_suffix_kv(
            key_cache,
            value_cache,
            attn_metadata.suffix_slot_mapping_out,
            attn_metadata.suffix_total_tokens,
            attn_metadata.block_size,
        )

        suffix_out, suffix_lse = flash_attn_varlen_func(
            q=beam_query,
            k=contig_k,
            v=contig_v,
            cu_seqlens_q=attn_metadata.suffix_cu_seqlens_q_device,
            cu_seqlens_k=attn_metadata.suffix_cu_seqlens_k_device,
            seqused_k=None,
            max_seqlen_q=attn_metadata.suffix_max_q_len,
            max_seqlen_k=attn_metadata.suffix_max_seq_len,
            softmax_scale=scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=None,
            softcap=logits_soft_cap,
            return_softmax_lse=True,
            fa_version=vllm_flash_attn_version,
        )
        group_size = attn_metadata.num_heads // attn_metadata.num_kv_heads
        if (
            attn_metadata.suffix_max_q_len == 1
            and attn_metadata.suffix_max_seq_len == 1
            and group_size > 1
            and vllm_flash_attn_version != 3
        ):
            suffix_lse = (
                suffix_lse.view(attn_metadata.num_kv_heads, group_size, -1)
                .transpose(1, 2)
                .contiguous()
                .view(attn_metadata.num_heads, -1)
            )

        out = torch.empty_like(beam_query)
        merge_attn_states(out, prefix_out, prefix_lse, suffix_out, suffix_lse)
        if attn_metadata.std_indices is None:
            output[: attn_metadata.num_prefix_indices] = out
        else:
            output[attn_metadata.prefix_indices] = out
