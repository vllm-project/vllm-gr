# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""STU (Sequential Transducer Unit) module for HSTU model."""

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from torch.autograd.profiler import record_function

from vllm_gr.models.dlrm_v3.common import HammerModule
from vllm_gr.ops.hstu_compute import hstu_compute_output, hstu_compute_uqvk, hstu_preprocess_and_attention
from vllm_gr.utils import ReliableOps


class STU(HammerModule, abc.ABC):
    """Abstract base class for STU modules.

    Subclasses support the full-sequence path and the native paged-KV
    incremental path used by HSTU under vLLM attention metadata.
    """

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any] = None,
        hstu_cache_layer: Optional[Any] = None,
        hstu_attention_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Run the incremental forward path for cached HSTU attention."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any] = None,
        hstu_cache_layer: Optional[Any] = None,
        hstu_attention_mode: Optional[str] = None,
    ) -> torch.Tensor:
        pass


@dataclass
class STULayerConfig:
    """Configuration for STU layer."""
    embedding_dim: int
    num_heads: int
    hidden_dim: int
    attention_dim: int
    output_dropout_ratio: float = 0.3
    causal: bool = True
    target_aware: bool = True
    max_attn_len: Optional[int] = None
    attn_alpha: Optional[float] = None
    use_group_norm: bool = False
    recompute_normed_x: bool = True
    recompute_uvqk: bool = True
    recompute_y: bool = True
    sort_by_length: bool = True
    contextual_seq_len: int = 0


@torch.fx.wrap
def _build_subset_row_indices(
    full_seq_lengths: torch.Tensor,
    subset_lengths: torch.Tensor,
    from_tail: bool,
) -> torch.Tensor:
    full_offsets = ReliableOps.asynchronous_complete_cumsum(full_seq_lengths)
    indices: list[torch.Tensor] = []
    for batch_idx in range(int(full_seq_lengths.numel())):
        full_start = int(full_offsets[batch_idx].item())
        full_end = int(full_offsets[batch_idx + 1].item())
        subset_len = int(subset_lengths[batch_idx].item())
        if subset_len == 0:
            continue
        if from_tail:
            subset_start = full_end - subset_len
            subset_end = full_end
        else:
            subset_start = full_start
            subset_end = full_start + subset_len
        indices.append(
            torch.arange(
                subset_start,
                subset_end,
                device=full_seq_lengths.device,
                dtype=full_offsets.dtype,
            )
        )
    if not indices:
        return torch.empty(0, device=full_seq_lengths.device, dtype=full_offsets.dtype)
    return torch.cat(indices, dim=0)


@torch.fx.wrap
def _pad_subset_rows_to_full(
    subset_x: torch.Tensor,
    full_seq_lengths: torch.Tensor,
    subset_lengths: torch.Tensor,
    from_tail: bool,
    row_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if row_indices is None:
        row_indices = _build_subset_row_indices(
            full_seq_lengths=full_seq_lengths,
            subset_lengths=subset_lengths,
            from_tail=from_tail,
        )
    full_shape = (int(full_seq_lengths.sum().item()),) + tuple(subset_x.shape[1:])
    full_x = torch.zeros(
        full_shape,
        device=subset_x.device,
        dtype=subset_x.dtype,
    )
    if row_indices.numel() > 0:
        full_x[row_indices.long()] = subset_x
    return full_x, row_indices.long()


class STULayer(STU):
    """Single STU layer implementation."""

    def __init__(
        self,
        config: STULayerConfig,
    ) -> None:
        super().__init__()

        self._num_heads: int = config.num_heads
        self._embedding_dim: int = config.embedding_dim
        self._hidden_dim: int = config.hidden_dim
        self._attention_dim: int = config.attention_dim
        self._output_dropout_ratio: float = config.output_dropout_ratio
        self._target_aware: bool = config.target_aware
        self._causal: bool = config.causal
        self._max_attn_len: int = config.max_attn_len or 0
        self._attn_alpha: float = config.attn_alpha or 1.0 / (self._attention_dim ** 0.5)
        self._use_group_norm: bool = config.use_group_norm
        self._recompute_normed_x: bool = config.recompute_normed_x
        self._recompute_uvqk: bool = config.recompute_uvqk
        self._recompute_y: bool = config.recompute_y
        self._sort_by_length: bool = config.sort_by_length
        self._contextual_seq_len: int = config.contextual_seq_len

        # U, V, Q, K projection weights
        self._uvqk_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((
                self._embedding_dim,
                (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
            )),
        )
        torch.nn.init.xavier_uniform_(self._uvqk_weight)
        
        self._uvqk_beta: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads),
        )
        
        # Input layer norm
        self._input_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((self._embedding_dim,)),
        )
        self._input_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._embedding_dim,)),
        )
        
        # Output projection
        # When concat_u=True and concat_x=True in pytorch_norm_mul_dropout:
        #   y = torch.cat([u, x, y], dim=1)  where x is actually attn
        # u dim: num_heads * hidden_dim
        # x (attn) dim: num_heads * hidden_dim  
        # y (u * normalized(attn)) dim: num_heads * hidden_dim
        # Total: 3 * num_heads * hidden_dim
        # output_weight shape: (y_dim, embedding_dim) for torch.addmm
        self._output_weight = torch.nn.Parameter(
            torch.empty((self._hidden_dim * self._num_heads * 3, self._embedding_dim)),
        )
        torch.nn.init.xavier_uniform_(self._output_weight)
        
        # Output layer norm
        output_norm_shape: int = (
            self._hidden_dim * self._num_heads
            if not self._use_group_norm
            else self._num_heads
        )
        self._output_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((output_norm_shape,)),
        )
        self._output_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((output_norm_shape,)),
        )

    def _get_runtime_vllm_kv_cache(
        self,
        hstu_attn_metadata: Optional[Any],
        hstu_cache_layer: Optional[Any],
    ) -> Optional[torch.Tensor]:
        if hstu_attn_metadata is None or hstu_cache_layer is None:
            return None
        runtime_kv_cache = hstu_cache_layer.get_runtime_kv_cache()
        if runtime_kv_cache is None or runtime_kv_cache.numel() == 0:
            return None
        return runtime_kv_cache

    def _forward_with_paged_kv_cache(
        self,
        x: torch.Tensor,
        hstu_cache_layer: Any,
        runtime_kv_cache: torch.Tensor,
        hstu_attn_metadata: Any,
        hstu_attention_mode: Optional[str],
    ) -> torch.Tensor:
        attn_mode = hstu_attention_mode or "full"
        full_padded_x = x
        row_indices: Optional[torch.Tensor] = None
        if attn_mode in {"prefix", "delta"}:
            prefix_seq_lens, full_seq_lens = hstu_attn_metadata.get_seq_lens(
                device=x.device,
            )
            subset_lengths = (
                prefix_seq_lens
                if attn_mode == "prefix"
                else hstu_attn_metadata.num_targets.to(
                    device=x.device,
                    dtype=full_seq_lens.dtype,
                )
            )
            full_padded_x, row_indices = _pad_subset_rows_to_full(
                subset_x=x,
                full_seq_lengths=full_seq_lens,
                subset_lengths=subset_lengths,
                from_tail=(attn_mode == "delta"),
            )

        with record_function("## stu_compute_uqvk ##"):
            full_u, full_q, full_k, full_v = hstu_compute_uqvk(
                x=full_padded_x,
                norm_weight=self._input_norm_weight.to(full_padded_x.dtype),
                norm_bias=self._input_norm_bias.to(full_padded_x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(full_padded_x.dtype),
                uvqk_bias=self._uvqk_beta.to(full_padded_x.dtype),
                kernel=self.hammer_kernel(),
            )
        if row_indices is not None:
            u = full_u[row_indices]
            q = full_q[row_indices]
            k = full_k[row_indices]
            v = full_v[row_indices]
        else:
            u = full_u
            q = full_q
            k = full_k
            v = full_v

        attn_output = hstu_cache_layer(
            query=q,
            key=k,
            value=v,
            kv_cache=runtime_kv_cache,
            attn_metadata=hstu_attn_metadata,
            attn_mode=hstu_attention_mode,
        ).reshape(-1, self._hidden_dim * self._num_heads)

        with record_function("## stu_compute_output ##"):
            if row_indices is not None:
                full_padded_u, _ = _pad_subset_rows_to_full(
                    subset_x=u,
                    full_seq_lengths=full_seq_lens,
                    subset_lengths=subset_lengths,
                    from_tail=(attn_mode == "delta"),
                    row_indices=row_indices,
                )
                full_padded_attn, _ = _pad_subset_rows_to_full(
                    subset_x=attn_output,
                    full_seq_lengths=full_seq_lens,
                    subset_lengths=subset_lengths,
                    from_tail=(attn_mode == "delta"),
                    row_indices=row_indices,
                )
                full_output = hstu_compute_output(
                    attn=full_padded_attn,
                    u=full_padded_u,
                    x=full_padded_x,
                    norm_weight=self._output_norm_weight.to(full_padded_x.dtype),
                    norm_bias=self._output_norm_bias.to(full_padded_x.dtype),
                    norm_eps=1e-6,
                    dropout_ratio=self._output_dropout_ratio,
                    output_weight=self._output_weight.to(full_padded_x.dtype),
                    group_norm=self._use_group_norm,
                    num_heads=self._num_heads,
                    linear_dim=self._hidden_dim,
                    concat_u=True,
                    concat_x=True,
                    training=self.training,
                    kernel=self.hammer_kernel(),
                    recompute_y_in_backward=self._recompute_y,
                )
                return full_output[row_indices]

            return hstu_compute_output(
                attn=attn_output,
                u=u,
                x=full_padded_x,
                norm_weight=self._output_norm_weight.to(full_padded_x.dtype),
                norm_bias=self._output_norm_bias.to(full_padded_x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(full_padded_x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any] = None,
        hstu_cache_layer: Optional[Any] = None,
        hstu_attention_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward pass of STU layer."""
        runtime_kv_cache = self._get_runtime_vllm_kv_cache(
            hstu_attn_metadata=hstu_attn_metadata,
            hstu_cache_layer=hstu_cache_layer,
        )
        if runtime_kv_cache is not None:
            return self._forward_with_paged_kv_cache(
                x=x,
                hstu_cache_layer=hstu_cache_layer,
                runtime_kv_cache=runtime_kv_cache,
                hstu_attn_metadata=hstu_attn_metadata,
                hstu_attention_mode=hstu_attention_mode,
            )

        with record_function("## stu_preprocess_and_attention ##"):
            u, attn_output, k, v = hstu_preprocess_and_attention(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                attn_alpha=self._attn_alpha,
                causal=self._causal,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                recompute_uvqk_in_backward=self._recompute_uvqk,
                recompute_normed_x_in_backward=self._recompute_normed_x,
                sort_by_length=self._sort_by_length,
                prefill=False,
                kernel=self.hammer_kernel(),
            )

        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=attn_output,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight.to(x.dtype),
                norm_bias=self._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any] = None,
        hstu_cache_layer: Optional[Any] = None,
        hstu_attention_mode: Optional[str] = None,
    ) -> torch.Tensor:
        runtime_kv_cache = self._get_runtime_vllm_kv_cache(
            hstu_attn_metadata=hstu_attn_metadata,
            hstu_cache_layer=hstu_cache_layer,
        )
        if runtime_kv_cache is not None:
            return self._forward_with_paged_kv_cache(
                x=delta_x,
                hstu_cache_layer=hstu_cache_layer,
                runtime_kv_cache=runtime_kv_cache,
                hstu_attn_metadata=hstu_attn_metadata,
                hstu_attention_mode=hstu_attention_mode,
            )
        raise RuntimeError(
            "STULayer.cached_forward requires native vLLM KV metadata and "
            "runtime paged KV cache."
        )


class STUStack(STU):
    """Stack of multiple STU layers."""
    
    def __init__(
        self,
        stu_list: List[STU],
    ) -> None:
        super().__init__()
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)

    def _resolve_cache_layers(
        self,
        hstu_attn_metadata: Optional[Any],
    ) -> list[Optional[Any]]:
        if hstu_attn_metadata is None:
            return []
        return list(getattr(hstu_attn_metadata, "cache_layers", []) or [])

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any] = None,
        hstu_attention_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward pass through all STU layers."""
        cache_layers = self._resolve_cache_layers(hstu_attn_metadata)
        for layer_idx, layer in enumerate(self._stu_layers):
            cache_layer = (
                cache_layers[layer_idx] if layer_idx < len(cache_layers) else None
            )
            x = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                hstu_attn_metadata=hstu_attn_metadata,
                hstu_cache_layer=cache_layer,
                hstu_attention_mode=hstu_attention_mode,
            )
        return x

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any] = None,
        hstu_attention_mode: Optional[str] = None,
    ) -> torch.Tensor:
        cache_layers = self._resolve_cache_layers(hstu_attn_metadata)
        for layer_idx, layer in enumerate(self._stu_layers):
            cache_layer = (
                cache_layers[layer_idx] if layer_idx < len(cache_layers) else None
            )
            delta_x = layer.cached_forward(
                delta_x=delta_x,
                num_targets=num_targets,
                hstu_attn_metadata=hstu_attn_metadata,
                hstu_cache_layer=cache_layer,
                hstu_attention_mode=hstu_attention_mode,
            )
        return delta_x
