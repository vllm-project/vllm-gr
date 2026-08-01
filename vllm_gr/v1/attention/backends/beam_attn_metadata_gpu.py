# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-specific Beam Attention metadata construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BaseBeamAttentionMetadataBuilder,
    BeamAttentionMetadata,
    _get_fp8_dtype_for_flashattn,
)


def get_scheduler_metadata(*args, **kwargs):
    from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available

    if not is_flash_attn_varlen_func_available():
        raise RuntimeError(
            "CUDABeamAttentionMetadataBuilder requires FlashAttention scheduler support"
        )
    from vllm.v1.attention.backends.fa_utils import (
        get_scheduler_metadata as _get_scheduler_metadata,
    )

    return _get_scheduler_metadata(*args, **kwargs)


@dataclass
class CUDABeamAttentionMetadata(BeamAttentionMetadata):
    scheduler_metadata: dict | None = None
    prefix_scheduler_metadata: dict | None = None
    max_num_splits: int = 0


class CUDABeamAttentionMetadataBuilder(BaseBeamAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.max_num_splits = 0
        self.use_full_cuda_graph = self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            if hasattr(vllm_config, "attention_config") and hasattr(
                vllm_config.attention_config,
                "flash_attn_max_num_splits_for_cuda_graph",
            ):
                self.max_num_splits = (
                    vllm_config.attention_config.flash_attn_max_num_splits_for_cuda_graph
                )

    def _get_max_num_splits(self, num_actual_tokens: int) -> int:
        max_num_splits = 0
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            max_num_splits = self.max_num_splits
        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1
        return max_num_splits

    def _get_schedule(
        self,
        batch_size: int,
        cu_query_lens: torch.Tensor,
        max_query_len: int,
        seqlens: torch.Tensor,
        max_seq_len: int,
        causal: bool,
        max_num_splits: int,
    ) -> dict | None:
        if not self.aot_schedule:
            return None

        cache_dtype = self.cache_config.cache_dtype
        qkv_dtype = (
            _get_fp8_dtype_for_flashattn(cache_dtype)
            if cache_dtype.startswith("fp8")
            else self.kv_cache_dtype
        )
        return get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            num_heads_q=self.num_heads_q,
            num_heads_kv=self.num_kv_heads,
            headdim=self.headdim,
            cache_seqlens=seqlens,
            qkv_dtype=qkv_dtype,
            cu_seqlens_q=cu_query_lens,
            page_size=self.block_size,
            causal=causal,
            window_size=self.aot_sliding_window,
            num_splits=max_num_splits,
        )

    def _make_std_device_tensors(
        self,
        std_cu_seqlens_q_cpu: list[int],
        std_seqlens_kv_cpu: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([0] + std_cu_seqlens_q_cpu, dtype=torch.int32, device=device),
            torch.tensor(std_seqlens_kv_cpu, dtype=torch.int32, device=device),
        )

    def _make_beam_device_tensors(
        self,
        prefix_seqlens_kv_cpu: list[int],
        prefix_cu_seqlens_q_cpu: list[int],
        suffix_seqlens_kv_cpu: list[int],
        suffix_cu_seqlens_q_cpu: list[int],
        suffix_cu_seqlens_kv_cpu: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(prefix_seqlens_kv_cpu, dtype=torch.int32, device=device),
            torch.tensor([0] + prefix_cu_seqlens_q_cpu, dtype=torch.int32, device=device),
            torch.tensor(suffix_seqlens_kv_cpu, dtype=torch.int32, device=device),
            torch.tensor([0] + suffix_cu_seqlens_q_cpu, dtype=torch.int32, device=device),
            torch.tensor([0] + suffix_cu_seqlens_kv_cpu, dtype=torch.int32, device=device),
        )

    def _make_metadata(
        self,
        metadata_kwargs: dict,
        *,
        max_num_splits: int,
        prefix_scheduler_metadata: dict | None,
    ) -> CUDABeamAttentionMetadata:
        return CUDABeamAttentionMetadata(
            **metadata_kwargs,
            max_num_splits=max_num_splits,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )
