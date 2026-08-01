# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU-specific Beam Attention metadata construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BaseBeamAttentionMetadataBuilder,
    BeamAttentionMetadata,
)


@dataclass
class NPUBeamAttentionMetadata(BeamAttentionMetadata):
    npu_causal_atten_mask: torch.Tensor | None = None


class NPUBeamAttentionMetadataBuilder(BaseBeamAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.npu_causal_atten_mask = (
            torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
        )

    def _make_metadata(
        self,
        metadata_kwargs: dict,
        *,
        max_num_splits: int,
        prefix_scheduler_metadata: dict | None,
    ) -> NPUBeamAttentionMetadata:
        del max_num_splits, prefix_scheduler_metadata
        return NPUBeamAttentionMetadata(
            **metadata_kwargs,
            npu_causal_atten_mask=self.npu_causal_atten_mask,
        )
