# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FuXi model entry for vLLM ModelRegistry."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_gr.model_executor.models.hstu_vllm import HSTUPooler
from vllm_gr.utils.hstu_runtime import resolve_hstu_runtime_devices

logger = init_logger(__name__)


class FuXiForRecommendation(nn.Module):
    """vLLM-compatible wrapper around FuXiAlphaInferenceModel."""

    supports_multimodal = False
    supports_pp = False
    is_pooling_model = True
    is_text_generation_model = False
    is_attention_free = False

    def __init__(
        self,
        *,
        vllm_config: Any = None,
        prefix: str = "",
        config: Any = None,
        cache_config: Any = None,
        quant_config: Any = None,
        lora_config: Any = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.vllm_config = vllm_config

        if vllm_config is not None:
            config = vllm_config.model_config.hf_config

        self.config = config
        self.enable_kv_cache = bool(getattr(self.config, "enable_kv_cache", False))
        self.kv_cache_dtype = getattr(self.config, "kv_cache_dtype", None) or getattr(
            self.config, "torch_dtype", torch.bfloat16
        )
        dense_device_override = torch.device(device) if device is not None else None
        runtime_devices = resolve_hstu_runtime_devices(
            explicit_dense_device=dense_device_override,
        )
        self._dense_device = runtime_devices.dense_device

        self._create_fuxi_model()
        self._hstu_kv_cache_layers = nn.ModuleList(self._fuxi.kv_cache_layers)
        self._pooler = HSTUPooler(self.config.item_embedding_dim)

        logger.info(
            "FuXiForRecommendation initialized: blocks=%d, heads=%d, "
            "item_dim=%d, enable_kv_cache=%s",
            self.config.num_blocks,
            self.config.num_heads,
            self.config.item_embedding_dim,
            self.enable_kv_cache,
        )

    @property
    def pooler(self):
        return self._pooler

    def _create_fuxi_model(self) -> None:
        from vllm_gr.models.fuxi.fuxi_module import FuXiAlphaInferenceModel

        self._fuxi = FuXiAlphaInferenceModel(
            self.config,
            vllm_config=self.vllm_config,
            prefix=self.prefix,
        )
        self._fuxi.to(self._dense_device)

    @staticmethod
    def _split_pooling_predictions(
        predictions: torch.Tensor,
        candidate_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        if candidate_lengths.numel() == 0:
            return []

        split_sizes = [int(size) for size in candidate_lengths.reshape(-1).tolist()]
        total_candidates = sum(split_sizes)
        if predictions.shape[-1] != total_candidates:
            raise ValueError(
                "FuXi pooling split mismatch: "
                f"predictions.shape={tuple(predictions.shape)}, "
                f"sum(candidate_lengths)={total_candidates}"
            )
        return list(torch.split(predictions, split_sizes, dim=-1))

    def _get_hstu_attention_metadata(
        self,
        uih_seq_lengths: Optional[torch.Tensor] = None,
        num_targets: Optional[torch.Tensor] = None,
    ) -> Optional[Any]:
        if not self.enable_kv_cache or len(self._hstu_kv_cache_layers) == 0:
            return None

        try:
            from vllm.forward_context import (
                get_forward_context,
                is_forward_context_available,
            )
        except ImportError:
            return None

        if not is_forward_context_available():
            return None

        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, list):
            if not attn_metadata:
                return None
            attn_metadata = attn_metadata[0]
        if not isinstance(attn_metadata, dict):
            return None

        for cache_layer in self._hstu_kv_cache_layers:
            layer_name = getattr(cache_layer, "layer_name", None)
            if layer_name is None:
                continue
            metadata = attn_metadata.get(layer_name)
            if metadata is not None:
                runtime_uih_seq_lengths = (
                    uih_seq_lengths.to(
                        device=metadata.history_lengths.device,
                        dtype=metadata.history_lengths.dtype,
                    )
                    if uih_seq_lengths is not None
                    else None
                )
                runtime_num_targets = (
                    num_targets.to(
                        device=metadata.num_targets.device,
                        dtype=metadata.num_targets.dtype,
                    )
                    if num_targets is not None
                    else None
                )
                if (
                    runtime_uih_seq_lengths is not None
                    and int(metadata.history_lengths.to(dtype=torch.int64).sum().item()) == 0
                    and int(runtime_uih_seq_lengths.to(dtype=torch.int64).sum().item()) > 0
                ):
                    metadata.history_lengths = runtime_uih_seq_lengths
                if (
                    runtime_num_targets is not None
                    and int(metadata.num_targets.to(dtype=torch.int64).sum().item()) == 0
                    and int(runtime_num_targets.to(dtype=torch.int64).sum().item()) > 0
                ):
                    metadata.num_targets = runtime_num_targets

                _total_delta_tokens = int(metadata.num_targets.to(dtype=torch.int64).sum().item())
                expected_total_tokens = int(
                    (
                        metadata.history_lengths.to(dtype=torch.int64)
                        + int(getattr(metadata, "contextual_seq_len", 0))
                        + metadata.num_targets.to(dtype=torch.int64)
                    )
                    .sum()
                    .item()
                )
                slot_mapping = getattr(
                    getattr(metadata, "common_attn_metadata", None),
                    "slot_mapping",
                    None,
                )
                num_computed_tokens = getattr(
                    getattr(metadata, "common_attn_metadata", None),
                    "_num_computed_tokens_cpu",
                    None,
                )
                actual_token_slots = int(slot_mapping.numel()) if slot_mapping is not None else 0
                _actual_computed_tokens = (
                    int(num_computed_tokens.to(dtype=torch.int64).sum().item())
                    if num_computed_tokens is not None
                    else 0
                )
                if actual_token_slots < expected_total_tokens:
                    return None
                metadata.cache_layers = list(self._hstu_kv_cache_layers)
                return metadata
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_lengths: Optional[torch.Tensor] = None,
        past_ids: Optional[torch.Tensor] = None,
        past_payloads: Optional[dict[str, torch.Tensor]] = None,
        candidate_ids: Optional[torch.Tensor] = None,
        candidate_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if past_lengths is None or past_ids is None:
            total_tokens = input_ids.numel() if input_ids is not None else 1
            return torch.zeros(
                total_tokens,
                self.config.item_embedding_dim,
                device=self._dense_device,
            )

        dense_device = self._dense_device
        past_lengths = past_lengths.to(dense_device)
        past_ids = past_ids.to(dense_device)
        if candidate_ids is not None:
            candidate_ids = candidate_ids.to(dense_device)
        if candidate_lengths is not None:
            candidate_lengths = candidate_lengths.to(dense_device)
        if past_payloads:
            past_payloads = {key: value.to(dense_device) for key, value in past_payloads.items()}

        hstu_attn_metadata = self._get_hstu_attention_metadata(
            uih_seq_lengths=past_lengths,
            num_targets=torch.zeros_like(past_lengths),
        )
        predictions = self._fuxi(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_payloads=past_payloads,
            candidate_ids=candidate_ids,
            candidate_lengths=candidate_lengths,
            hstu_attn_metadata=hstu_attn_metadata,
        )
        if candidate_lengths is None:
            return predictions
        return self._split_pooling_predictions(predictions, candidate_lengths)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Set[str]:
        from vllm.model_executor.models.utils import AutoWeightsLoader

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["_pooler."],
        )
        return loader.load_weights(weights)

    @classmethod
    def from_model_config(
        cls,
        config: Any,
        device: torch.device,
        **kwargs,
    ) -> "FuXiForRecommendation":
        return cls(config=config, device=device, **kwargs)
