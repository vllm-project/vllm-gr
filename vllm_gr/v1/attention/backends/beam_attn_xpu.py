# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native XPU Prefill and independent Torch XPU shared-prefix Beam Decode."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.functional import scaled_dot_product_attention
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)


@dataclass
class XPUDecodeMetadata:
    owner: Any
    step: int
    num_actual_tokens: int


def is_available() -> bool:
    return torch.xpu.is_available()


def beam_attention(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    width, heads, _ = query.shape
    # GQA expansion is explicit; the XPU SDPA implementation need not support
    # enable_gqa. Decode has one query per beam and every cached key is causal.
    key = torch.cat((prefix_key.unsqueeze(0).expand(width, -1, -1, -1), suffix_key), dim=1)
    value = torch.cat((prefix_value.unsqueeze(0).expand(width, -1, -1, -1), suffix_value), dim=1)
    repeats = heads // key.shape[2]
    key = key.repeat_interleave(repeats, dim=2).transpose(1, 2)
    value = value.repeat_interleave(repeats, dim=2).transpose(1, 2)
    return scaled_dot_product_attention(query.unsqueeze(2), key, value, scale=scale).squeeze(2)


class XPUBeamAttentionImpl(FlashAttentionImpl):
    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: Any = None,
        output_block_scale: Any = None,
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, XPUDecodeMetadata):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("XPU Beam does not support quantized attention output")
        if (
            self.sliding_window != (-1, -1)
            or self.logits_soft_cap
            or self.alibi_slopes is not None
            or getattr(self, "sinks", None) is not None
        ):
            raise ValueError("XPU Beam requires ordinary causal decoder attention")
        owner, step = attn_metadata.owner, attn_metadata.step
        if not 0 <= step < owner.max_tokens:
            raise ValueError("XPU Beam suffix step exceeds its reserved capacity")
        name = layer.layer_name
        if name not in owner.layer_kv:
            if kv_cache.ndim != 4 or kv_cache.shape[-1] != 2 * self.head_size:
                raise ValueError("XPU Beam requires the native paged XPU KV layout")
            prefix_k, prefix_v = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
            block_size = prefix_k.shape[1]
            if len(owner.prefix_blocks) * block_size < owner.prompt_length:
                raise RuntimeError("XPU Beam prefix block capacity is insufficient")
            block_ids = torch.tensor(owner.prefix_blocks, device=key.device, dtype=torch.long)
            prefix_k = prefix_k[block_ids].flatten(0, 1)[: owner.prompt_length]
            prefix_v = prefix_v[block_ids].flatten(0, 1)[: owner.prompt_length]
            shape = (owner.width, owner.max_tokens, self.num_kv_heads, self.head_size)
            owner.layer_kv[name] = (
                prefix_k,
                prefix_v,
                torch.empty(shape, device=key.device, dtype=key.dtype),
                torch.empty(shape, device=value.device, dtype=value.dtype),
            )
        prefix_k, prefix_v, suffix_k, suffix_v = owner.layer_kv[name]
        suffix_k[:, step].copy_(key)
        suffix_v[:, step].copy_(value)
        result = beam_attention(
            query,
            prefix_k,
            prefix_v,
            suffix_k[:, : step + 1],
            suffix_v[:, : step + 1],
            self.scale,
        )
        output.copy_(result.reshape_as(output))
        return output


class XPUBeamAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[XPUBeamAttentionImpl]:
        return XPUBeamAttentionImpl
