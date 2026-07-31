# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU execution primitives for the Beam Attention backend."""

from __future__ import annotations

from typing import Any

import torch
import torch_npu

from vllm_gr.v1.attention.backends.beam_attn_metadata_npu import (
    NPUBeamAttentionMetadata,
    NPUBeamAttentionMetadataBuilder,
)
from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    extract_suffix_kv_pytorch,
    merge_tnd_attention_outputs,
)


def is_available() -> bool:
    return True


def get_flash_attn_version(
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
) -> int:
    del requires_alibi, head_size
    return 0


def supports_sinks() -> bool:
    return False


def supports_batch_invariance() -> bool:
    return False


def supports_kv_cache_dtype(kv_cache_dtype: Any | None) -> bool:
    if kv_cache_dtype is None:
        return True
    return kv_cache_dtype in ("auto", "float16", "bfloat16")


def supports_compute_capability(capability: Any) -> bool:
    del capability
    return True


def validate_combination(
    *,
    head_size: int,
    dtype: Any,
    kv_cache_dtype: Any,
    block_size: int | None,
    use_mla: bool,
    has_sink: bool,
    use_sparse: bool,
    device_capability: Any,
) -> str | None:
    del device_capability
    if dtype not in (torch.float16, torch.bfloat16):
        return f"Unsupported dtype {dtype}."
    if head_size % 8 != 0 or head_size > 256:
        return f"Unsupported head_size={head_size}."
    if not supports_kv_cache_dtype(kv_cache_dtype):
        return f"Unsupported kv_cache_dtype={kv_cache_dtype}."
    if block_size is not None and block_size % 16 != 0:
        return f"block_size={block_size} not multiple of 16."
    if has_sink:
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
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

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


def split_kv_cache(kv_cache: Any) -> tuple[Any, Any]:
    return kv_cache[0], kv_cache[1]


def get_kv_scales(layer: Any) -> tuple[None, None]:
    del layer
    return None, None


def get_metadata_builder_cls() -> type[NPUBeamAttentionMetadataBuilder]:
    return NPUBeamAttentionMetadataBuilder


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
    del kv_cache_dtype, k_scale, v_scale
    torch_npu._npu_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: NPUBeamAttentionMetadata,
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
        std_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=std_query,
            key=key_cache.view(*key_cache.shape[:-2], -1),
            value=value_cache.view(*value_cache.shape[:-2], -1),
            atten_mask=attn_metadata.npu_causal_atten_mask if attn_metadata.causal else None,
            block_table=attn_metadata.std_block_table,
            input_layout="TND",
            block_size=attn_metadata.block_size,
            actual_seq_qlen=attn_metadata.std_cu_seqlens_q_cpu,
            actual_seq_kvlen=attn_metadata.std_seqlens_kv_cpu,
            num_key_value_heads=attn_metadata.num_kv_heads,
            num_query_heads=attn_metadata.num_heads,
            softmax_scale=scale,
            sparse_mode=3 if attn_metadata.causal else 0,
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
        prefix_out, prefix_lse = torch_npu.npu_fused_infer_attention_score_v2(
            query=beam_query,
            key=key_cache.view(*key_cache.shape[:-2], -1),
            value=value_cache.view(*value_cache.shape[:-2], -1),
            block_table=attn_metadata.prefix_block_table,
            input_layout="TND",
            atten_mask=None,
            block_size=attn_metadata.block_size,
            actual_seq_qlen=attn_metadata.prefix_cu_seqlens_q_cpu,
            actual_seq_kvlen=attn_metadata.prefix_seqlens_kv_cpu,
            num_key_value_heads=attn_metadata.num_kv_heads,
            num_query_heads=attn_metadata.num_heads,
            softmax_scale=scale,
            return_softmax_lse=True,
            sparse_mode=0,
        )

        contig_k, contig_v = extract_suffix_kv_pytorch(
            key_cache,
            value_cache,
            attn_metadata.suffix_slot_mapping_out,
        )

        suffix_out, suffix_lse = torch_npu.npu_fused_infer_attention_score_v2(
            beam_query,
            contig_k,
            contig_v,
            num_query_heads=attn_metadata.num_heads,
            num_key_value_heads=attn_metadata.num_kv_heads,
            actual_seq_qlen=attn_metadata.suffix_cu_seqlens_q_cpu,
            actual_seq_kvlen=attn_metadata.suffix_cu_seqlens_kv_cpu,
            softmax_scale=scale,
            atten_mask=attn_metadata.npu_causal_atten_mask,
            block_table=None,
            block_size=0,
            sparse_mode=3,
            inner_precise=0,
            input_layout="TND",
            return_softmax_lse=True,
        )
        out, _ = merge_tnd_attention_outputs(
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
        )
        if attn_metadata.std_indices is None:
            output[: attn_metadata.num_prefix_indices] = out
        else:
            output[attn_metadata.prefix_indices] = out
