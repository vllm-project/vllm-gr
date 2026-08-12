# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU execution primitives for the Beam Attention backend."""

from __future__ import annotations

import torch
import torch_npu

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
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
    attn_metadata: BeamAttentionMetadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    beam_pool: object = None,  # Per-layer pool, passed explicitly by forward()
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

        # --- Incremental decode path: read suffix KV from pool slice ---
        beam_session = getattr(attn_metadata, "beam_session", None)
        is_beam_decode = getattr(attn_metadata, "is_beam_decode", False)

        if is_beam_decode and beam_session is not None and beam_session.is_active:
            # Incremental decode: read suffix KV from pool slice
            # instead of extract_suffix_kv_pytorch from paged cache.
            # Prefer the explicitly-passed beam_pool; fall back to metadata
            # for backward compatibility with callers that still set
            # beam_layer_idx on the metadata.
            if beam_pool is not None:
                pool = beam_pool
            else:
                layer_idx = getattr(attn_metadata, "beam_layer_idx", 0)
                pool = beam_session.pools[layer_idx]
            step = beam_session.decode_step  # host int, no .item()
            slot = beam_session.slot
            H = pool.num_kv_heads
            D = pool.head_dim
            W = beam_session.beam_width

            # Read pool slice: [W, H, step+1, D]
            pool_k = pool.unshared_key[slot, :W, :, : step + 1, :]
            pool_v = pool.unshared_value[slot, :W, :, : step + 1, :]

            # Copy to pre-allocated fixed buffer (avoiding new allocation)
            # Reshape: [W, H, step+1, D] -> [W*(step+1), H, D]
            total_kv = W * (step + 1)
            suffix_k_buf = beam_session.suffix_k_buf
            suffix_v_buf = beam_session.suffix_v_buf
            suffix_k_buf[:total_kv] = pool_k.permute(0, 2, 1, 3).reshape(-1, H, D)
            suffix_v_buf[:total_kv] = pool_v.permute(0, 2, 1, 3).reshape(-1, H, D)

            suffix_out, suffix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                beam_query,
                suffix_k_buf[:total_kv],
                suffix_v_buf[:total_kv],
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
        else:
            # Standard (recompute) path: extract_suffix_kv from paged cache
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
