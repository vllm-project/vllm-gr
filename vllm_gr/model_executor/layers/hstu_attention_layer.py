# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Layer bridge that exposes HSTU attention to vLLM KV cache runtime."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_gr.attention.backends.hstu_attn import (
    HSTUAttentionBackend,
    HSTUAttentionImpl,
    HSTUAttentionMetadata,
)
from vllm_gr.model_executor.kv_cache_utils import (
    _resolve_torch_dtype,
    read_from_paged_cache,
    write_to_paged_cache,
)


class HSTUKVCacheAttentionLayer(nn.Module, AttentionLayerBase):  # type: ignore[misc]
    """Thin bridge so vLLM can discover and bind HSTU KV cache specs."""

    def __init__(
        self,
        *,
        layer_name: str,
        num_heads: int,
        head_size: int,
        head_size_v: int,
        dtype: torch.dtype | str,
        vllm_config: Optional[VllmConfig],
        enable_kv_cache: bool,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.scale = head_size**-0.5
        self.attn_type = AttentionType.DECODER
        self.kv_cache_torch_dtype = _resolve_torch_dtype(dtype)
        self.enable_kv_cache = enable_kv_cache
        self.kv_cache: list[torch.Tensor] = []
        HSTUAttentionBackend.set_head_size_v(head_size_v)
        self.impl = HSTUAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            attn_type=AttentionType.DECODER,
        )

        if vllm_config is not None and enable_kv_cache:
            compilation_config = vllm_config.compilation_config
            if layer_name in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {layer_name}")
            compilation_config.static_forward_context[layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        HSTUAttentionBackend.set_head_size_v(self.head_size_v)
        return HSTUAttentionBackend  # type: ignore[no-any-return]

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if not self.enable_kv_cache:
            return None
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HSTUAttentionMetadata,
        attn_mode: str | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.impl.forward(
            layer=self,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            attn_mode=attn_mode,
            output=output,
        )

    def get_runtime_kv_cache(
        self,
        kv_cache: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if kv_cache is not None:
            return kv_cache
        if len(self.kv_cache) == 0:
            return None

        try:
            from vllm.forward_context import get_forward_context
        except ImportError:
            return None

        forward_context = get_forward_context()
        virtual_engine = getattr(forward_context, "virtual_engine", 0)
        if virtual_engine >= len(self.kv_cache):
            return None
        return self.kv_cache[virtual_engine]

    def write_to_paged_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
    ) -> None:
        write_to_paged_cache(
            runtime_kv_cache=self.get_runtime_kv_cache(kv_cache),
            key=key,
            value=value,
            slot_mapping=slot_mapping,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
        )

    def read_from_paged_cache(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        return read_from_paged_cache(
            runtime_kv_cache=self.get_runtime_kv_cache(kv_cache),
            block_table_tensor=block_table_tensor,
            seq_lens=seq_lens,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            kv_cache_torch_dtype=self.kv_cache_torch_dtype,
        )


__all__ = ["HSTUKVCacheAttentionLayer"]
