# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-specific Beam Attention metadata and builder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BaseBeamAttentionMetadata,
    BaseBeamAttentionMetadataBuilder,
)


def get_scheduler_metadata(*args, **kwargs):
    from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available

    if not is_flash_attn_varlen_func_available():
        raise RuntimeError("CUDABeamAttentionMetadataBuilder requires scheduler support")
    from vllm.v1.attention.backends.fa_utils import (
        get_scheduler_metadata as _get_scheduler_metadata,
    )

    return _get_scheduler_metadata(*args, **kwargs)


def _get_fp8_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


@dataclass
class CUDABeamAttentionMetadata(BaseBeamAttentionMetadata):
    std_cu_seqlens_q_device: torch.Tensor | None = None
    std_seqlens_kv_device: torch.Tensor | None = None
    prefix_cu_seqlens_q_device: torch.Tensor | None = None
    prefix_seqlens_kv_device: torch.Tensor | None = None
    suffix_cu_seqlens_q_device: torch.Tensor | None = None
    suffix_cu_seqlens_k_device: torch.Tensor | None = None
    suffix_seqlens_kv_device: torch.Tensor | None = None
    scheduler_metadata: dict | None = None
    prefix_scheduler_metadata: dict | None = None
    max_num_splits: int = 0


class CUDABeamAttentionMetadataBuilder(BaseBeamAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    metadata_cls = CUDABeamAttentionMetadata

    def _init_backend_state(self, vllm_config: VllmConfig) -> None:
        self.compilation_config = vllm_config.compilation_config
        self.max_num_splits = 0
        self.aot_schedule = False
        self.use_full_cuda_graph = self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size
        self.aot_sliding_window: tuple[int, int] | None = None

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

    def _update_backend_state(self, fast_build: bool) -> None:
        self.aot_schedule = self.aot_schedule and not fast_build
        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)

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
        *,
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
            _get_fp8_dtype(cache_dtype) if cache_dtype.startswith("fp8") else self.kv_cache_dtype
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

    def _create_std_backend_metadata(
        self,
        std_cu_seqlens_q_cpu: list[int],
        std_seqlens_kv_cpu: list[int],
        device: torch.device,
    ) -> dict[str, object]:
        return {
            "std_cu_seqlens_q_device": torch.tensor(
                [0] + std_cu_seqlens_q_cpu,
                dtype=torch.int32,
                device=device,
            ),
            "std_seqlens_kv_device": torch.tensor(
                std_seqlens_kv_cpu,
                dtype=torch.int32,
                device=device,
            ),
        }

    def _create_beam_backend_metadata(
        self,
        *,
        prefix_cu_seqlens_q_cpu: list[int],
        prefix_seqlens_kv_cpu: list[int],
        suffix_cu_seqlens_q_cpu: list[int],
        suffix_cu_seqlens_kv_cpu: list[int],
        suffix_seqlens_kv_cpu: list[int],
        prefix_max_q_len: int,
        prefix_max_seq_len: int,
        max_num_splits: int,
        device: torch.device,
    ) -> dict[str, object]:
        prefix_cu_seqlens_q_device = torch.tensor(
            [0] + prefix_cu_seqlens_q_cpu,
            dtype=torch.int32,
            device=device,
        )
        prefix_seqlens_kv_device = torch.tensor(
            prefix_seqlens_kv_cpu,
            dtype=torch.int32,
            device=device,
        )
        return {
            "prefix_cu_seqlens_q_device": prefix_cu_seqlens_q_device,
            "prefix_seqlens_kv_device": prefix_seqlens_kv_device,
            "suffix_seqlens_kv_device": torch.tensor(
                suffix_seqlens_kv_cpu,
                dtype=torch.int32,
                device=device,
            ),
            "suffix_cu_seqlens_q_device": torch.tensor(
                [0] + suffix_cu_seqlens_q_cpu,
                dtype=torch.int32,
                device=device,
            ),
            "suffix_cu_seqlens_k_device": torch.tensor(
                [0] + suffix_cu_seqlens_kv_cpu,
                dtype=torch.int32,
                device=device,
            ),
            "prefix_scheduler_metadata": self._get_schedule(
                batch_size=len(prefix_seqlens_kv_cpu),
                cu_query_lens=prefix_cu_seqlens_q_device,
                max_query_len=prefix_max_q_len,
                seqlens=prefix_seqlens_kv_device,
                max_seq_len=prefix_max_seq_len,
                causal=False,
                max_num_splits=max_num_splits,
            ),
        }

    def _extra_metadata(self, max_num_splits: int) -> dict[str, object]:
        return {"max_num_splits": max_num_splits}
