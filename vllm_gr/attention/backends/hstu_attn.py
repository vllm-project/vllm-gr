# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HSTU attention backend that runs on top of vLLM paged KV cache."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_gr.config.hstu_config import get_hstu_contextual_seq_len
from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.hstu_attention import delta_hstu_mha, hstu_mha


def _build_hstu_token_positions(
    attn_metadata: "HSTUAttentionMetadata",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return attn_metadata.get_partition_positions(device=device)


def _resolve_hstu_attention_mode(
    attn_metadata: "HSTUAttentionMetadata",
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mode: str | None = None,
) -> tuple[str, torch.Tensor]:
    prefix_positions, delta_positions, full_positions = _build_hstu_token_positions(
        attn_metadata,
        device=query.device,
    )
    token_positions = {
        "prefix": prefix_positions,
        "delta": delta_positions,
        "full": full_positions,
    }
    expected_tokens = {mode: int(positions.numel()) for mode, positions in token_positions.items()}

    if attn_mode is None:
        for candidate_mode in ("full", "prefix", "delta"):
            expected = expected_tokens[candidate_mode]
            if query.shape[0] == expected and key.shape[0] == expected:
                return candidate_mode, token_positions[candidate_mode]
        raise ValueError(
            "HSTUAttentionImpl input token count does not match any known "
            f"sequence partition: query={query.shape[0]} key={key.shape[0]}, "
            f"expected one of {expected_tokens!r}."
        )

    if attn_mode not in expected_tokens:
        raise ValueError(f"Unsupported HSTU attention mode: {attn_mode}")
    expected = expected_tokens[attn_mode]
    if query.shape[0] != expected or key.shape[0] != expected:
        raise ValueError(
            f"HSTUAttentionImpl expected {expected} tokens for mode "
            f"{attn_mode}, got query={query.shape[0]} key={key.shape[0]}."
        )
    return attn_mode, token_positions[attn_mode]


@dataclass
class HSTUAttentionMetadata(AttentionMetadata):  # type: ignore[misc]
    """HSTU-specific metadata layered on top of vLLM common attention metadata."""

    common_attn_metadata: CommonAttentionMetadata
    num_targets: torch.Tensor
    history_lengths: torch.Tensor
    contextual_seq_len: int
    layer_names: list[str]
    cache_layers: list[Any] | None = None
    _cached_position_device: tuple[str, int | None] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_prefix_positions: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_delta_positions: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_full_positions: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_prefix_slot_mapping: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_delta_slot_mapping: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_full_slot_mapping: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_prefix_seq_lens: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_full_seq_lens: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def reset_derived_cache(self) -> None:
        self._cached_position_device = None
        self._cached_prefix_positions = None
        self._cached_delta_positions = None
        self._cached_full_positions = None
        self._cached_prefix_slot_mapping = None
        self._cached_delta_slot_mapping = None
        self._cached_full_slot_mapping = None
        self._cached_prefix_seq_lens = None
        self._cached_full_seq_lens = None

    @staticmethod
    def _device_key(device: torch.device) -> tuple[str, int | None]:
        return (device.type, device.index)

    def _ensure_cached_derived_tensors(
        self,
        device: torch.device | None = None,
    ) -> None:
        target_device = (
            device if device is not None else self.common_attn_metadata.slot_mapping.device
        )
        device_key = self._device_key(target_device)
        if (
            self._cached_position_device == device_key
            and self._cached_prefix_positions is not None
            and self._cached_delta_positions is not None
            and self._cached_full_positions is not None
            and self._cached_prefix_slot_mapping is not None
            and self._cached_delta_slot_mapping is not None
            and self._cached_full_slot_mapping is not None
            and self._cached_prefix_seq_lens is not None
            and self._cached_full_seq_lens is not None
        ):
            return

        common = self.common_attn_metadata
        query_start_loc = common.query_start_loc_cpu
        if query_start_loc.device.type != "cpu":
            query_start_loc = query_start_loc.cpu()

        prefix_seq_lens = self.history_lengths.to(device=target_device, dtype=torch.int64) + int(
            self.contextual_seq_len
        )
        delta_lengths = self.num_targets.to(device=target_device, dtype=torch.int64)
        full_seq_lens = common.seq_lens.to(device=target_device, dtype=torch.int64)

        prefix_positions: list[torch.Tensor] = []
        delta_positions: list[torch.Tensor] = []
        full_positions: list[torch.Tensor] = []
        for req_idx in range(common.num_reqs):
            start = int(query_start_loc[req_idx].item())
            prefix_len = int(prefix_seq_lens[req_idx].item())
            delta_len = int(delta_lengths[req_idx].item())
            full_len = prefix_len + delta_len
            if full_len > 0:
                full_positions.append(
                    torch.arange(
                        start,
                        start + full_len,
                        device=target_device,
                        dtype=torch.long,
                    )
                )
            if prefix_len > 0:
                prefix_positions.append(
                    torch.arange(
                        start,
                        start + prefix_len,
                        device=target_device,
                        dtype=torch.long,
                    )
                )
            if delta_len > 0:
                delta_positions.append(
                    torch.arange(
                        start + prefix_len,
                        start + prefix_len + delta_len,
                        device=target_device,
                        dtype=torch.long,
                    )
                )

        empty = torch.empty(0, device=target_device, dtype=torch.long)
        slot_mapping = common.slot_mapping.to(device=target_device, dtype=torch.long)
        self._cached_position_device = device_key
        self._cached_prefix_positions = torch.cat(prefix_positions) if prefix_positions else empty
        self._cached_delta_positions = torch.cat(delta_positions) if delta_positions else empty
        self._cached_full_positions = torch.cat(full_positions) if full_positions else empty
        self._cached_prefix_slot_mapping = slot_mapping[self._cached_prefix_positions]
        self._cached_delta_slot_mapping = slot_mapping[self._cached_delta_positions]
        self._cached_full_slot_mapping = slot_mapping[self._cached_full_positions]
        self._cached_prefix_seq_lens = prefix_seq_lens
        self._cached_full_seq_lens = full_seq_lens

    def get_partition_positions(
        self,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_cached_derived_tensors(device=device)
        assert self._cached_prefix_positions is not None
        assert self._cached_delta_positions is not None
        assert self._cached_full_positions is not None
        return (
            self._cached_prefix_positions,
            self._cached_delta_positions,
            self._cached_full_positions,
        )

    def get_seq_lens(
        self,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cached_derived_tensors(device=device)
        assert self._cached_prefix_seq_lens is not None
        assert self._cached_full_seq_lens is not None
        return self._cached_prefix_seq_lens, self._cached_full_seq_lens

    def get_slot_mapping(
        self,
        mode: str,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        self._ensure_cached_derived_tensors(device=device)
        if mode == "prefix":
            assert self._cached_prefix_slot_mapping is not None
            return self._cached_prefix_slot_mapping
        if mode == "delta":
            assert self._cached_delta_slot_mapping is not None
            return self._cached_delta_slot_mapping
        if mode == "full":
            assert self._cached_full_slot_mapping is not None
            return self._cached_full_slot_mapping
        raise ValueError(f"Unsupported HSTU attention mode: {mode}")


class HSTUAttentionMetadataBuilder(AttentionMetadataBuilder[HSTUAttentionMetadata]):  # type: ignore[misc]
    """Build complete HSTU metadata for vLLM attention layers."""

    supports_update_block_table = True
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: FullAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(reorder_batch_threshold=1)
        self._num_targets: torch.Tensor | None = None
        self._history_lengths: torch.Tensor | None = None
        hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
        self._contextual_seq_len: int = (
            get_hstu_contextual_seq_len(hf_config) if hf_config is not None else 0
        )

    def set_request_lengths(
        self,
        *,
        num_targets: torch.Tensor,
        history_lengths: torch.Tensor,
        contextual_seq_len: int,
    ) -> None:
        self._num_targets = num_targets
        self._history_lengths = history_lengths
        self._contextual_seq_len = int(contextual_seq_len)

    def clear_request_lengths(self) -> None:
        self._num_targets = None
        self._history_lengths = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HSTUAttentionMetadata:
        del common_prefix_len, fast_build
        num_reqs = common_attn_metadata.num_reqs
        zeros = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        num_targets = (
            self._num_targets.to(device=self.device, dtype=torch.int32)
            if self._num_targets is not None
            else zeros
        )
        history_lengths = (
            self._history_lengths.to(device=self.device, dtype=torch.int32)
            if self._history_lengths is not None
            else zeros.clone()
        )
        return HSTUAttentionMetadata(
            common_attn_metadata=common_attn_metadata,
            num_targets=num_targets,
            history_lengths=history_lengths,
            contextual_seq_len=self._contextual_seq_len,
            layer_names=list(self.layer_names),
            cache_layers=None,
        )

    def update_block_table(
        self,
        metadata: HSTUAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> HSTUAttentionMetadata:
        common_attn_metadata = copy.copy(metadata.common_attn_metadata)
        common_attn_metadata.block_table_tensor = blk_table
        common_attn_metadata.slot_mapping = slot_mapping
        return HSTUAttentionMetadata(
            common_attn_metadata=common_attn_metadata,
            num_targets=metadata.num_targets,
            history_lengths=metadata.history_lengths,
            contextual_seq_len=metadata.contextual_seq_len,
            layer_names=list(metadata.layer_names),
            cache_layers=metadata.cache_layers,
        )


class HSTUAttentionImpl(AttentionImpl[HSTUAttentionMetadata]):  # type: ignore[misc]
    """HSTU attention impl that consumes vLLM paged KV cache."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        del alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap
        del kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.attn_type = attn_type

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HSTUAttentionMetadata,
        attn_mode: str | None = None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "HSTUAttentionImpl does not support fused output quantization."
            )

        if attn_metadata is None:
            if output is not None:
                return output
            return torch.zeros(
                query.shape[0],
                self.num_heads,
                getattr(layer, "head_size_v", value.shape[-1]),
                dtype=value.dtype,
                device=value.device,
            )

        prefix_seq_lens, full_seq_lens = attn_metadata.get_seq_lens(
            device=query.device,
        )
        total_delta_tokens = int(attn_metadata.num_targets.to(dtype=torch.int64).sum().item())
        total_full_tokens = int(full_seq_lens.sum().item())
        if kv_cache.numel() == 0:
            if output is not None:
                return output
            target_tokens = (
                total_delta_tokens if query.shape[0] == total_delta_tokens else total_full_tokens
            )
            return torch.zeros(
                target_tokens,
                self.num_heads,
                getattr(layer, "head_size_v", value.shape[-1]),
                dtype=value.dtype,
                device=value.device,
            )

        mode, _token_positions = _resolve_hstu_attention_mode(
            attn_metadata=attn_metadata,
            query=query,
            key=key,
            attn_mode=attn_mode,
        )
        slot_mapping = attn_metadata.get_slot_mapping(
            mode=mode,
            device=query.device,
        )
        layer.write_to_paged_cache(key, value, slot_mapping, kv_cache=kv_cache)

        # Match the current non-KV STU path exactly: contextual-prefix tokens
        # participate in the logical full sequence and paged-KV partitioning,
        # but the underlying attention kernels are still invoked with
        # contextual_seq_len=0 in the regular full-forward implementation.
        contextual_seq_len = 0
        runtime_kernel = HammerKernel.NPU if query.device.type == "npu" else HammerKernel.PYTORCH

        if mode == "delta":
            if total_delta_tokens == 0:
                result = query.new_empty((0, self.num_heads, value.shape[-1]))
                if output is not None:
                    output[:0] = result
                    return output
                return result

            full_k, full_v, max_seq_len, seq_offsets = layer.read_from_paged_cache(
                attn_metadata.common_attn_metadata.block_table_tensor,
                full_seq_lens,
                kv_cache=kv_cache,
            )
            delta_output = delta_hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self.scale,
                delta_q=query,
                k=full_k,
                v=full_v,
                seq_offsets=seq_offsets,
                num_targets=attn_metadata.num_targets,
                contextual_seq_len=contextual_seq_len,
                kernel=runtime_kernel,
            )
            if output is not None:
                output[: delta_output.shape[0]].copy_(delta_output)
                return output
            return delta_output

        if mode == "prefix":
            prefix_k, prefix_v, prefix_max_seq_len, prefix_offsets = layer.read_from_paged_cache(
                attn_metadata.common_attn_metadata.block_table_tensor,
                prefix_seq_lens,
                kv_cache=kv_cache,
            )
            del prefix_max_seq_len
            full_max_seq_len = int(full_seq_lens.max().item()) if full_seq_lens.numel() > 0 else 0
            prefix_output = hstu_mha(
                max_seq_len=full_max_seq_len,
                alpha=self.scale,
                q=query,
                k=prefix_k,
                v=prefix_v,
                seq_offsets=prefix_offsets,
                causal=True,
                num_targets=torch.zeros_like(attn_metadata.num_targets),
                contextual_seq_len=contextual_seq_len,
                kernel=runtime_kernel,
            )
            if output is not None:
                output[: prefix_output.shape[0]].copy_(prefix_output)
                return output
            return prefix_output

        full_k, full_v, max_seq_len, seq_offsets = layer.read_from_paged_cache(
            attn_metadata.common_attn_metadata.block_table_tensor,
            full_seq_lens,
            kv_cache=kv_cache,
        )
        full_output = hstu_mha(
            max_seq_len=max_seq_len,
            alpha=self.scale,
            q=query,
            k=full_k,
            v=full_v,
            seq_offsets=seq_offsets,
            causal=True,
            num_targets=attn_metadata.num_targets,
            contextual_seq_len=contextual_seq_len,
            kernel=runtime_kernel,
        )
        if output is not None:
            output[: full_output.shape[0]].copy_(full_output)
            return output
        return full_output


class HSTUAttentionBackend(AttentionBackend):  # type: ignore[misc]
    """vLLM backend descriptor for HSTU KV cache pages."""

    head_size_v: int = 128

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        return "HSTU_KV_CACHE"

    @staticmethod
    def get_impl_cls() -> type[HSTUAttentionImpl]:
        return HSTUAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[HSTUAttentionMetadataBuilder]:
        return HSTUAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + HSTUAttentionBackend.head_size_v,
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)


__all__ = [
    "HSTUAttentionBackend",
    "HSTUAttentionImpl",
    "HSTUAttentionMetadata",
    "HSTUAttentionMetadataBuilder",
    "_build_hstu_token_positions",
]
