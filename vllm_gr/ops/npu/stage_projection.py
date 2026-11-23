# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Fixed stage-local LM-head weights for Ascend."""

from __future__ import annotations

from typing import Any

import torch

from vllm_gr.ops.sliced_logits import SlicedLMHeadProxy


class NPUStageProjectionWeights:
    """Own the fixed padded weights, biases, and head proxies for each stage."""

    def __init__(self, lm_head: Any, stage_vocab: Any) -> None:
        self.vocab_size = int(stage_vocab.vocab_size)
        self._stage_token_ids = stage_vocab.stage_token_ids
        self.stage_token_counts = tuple(int(ids.numel()) for ids in self._stage_token_ids)
        source_weight = lm_head.weight
        source_bias = getattr(lm_head, "bias", None)
        stage_weights = []
        stage_biases = []
        proxies = []
        with torch.no_grad():
            for ids in self._stage_token_ids:
                weight = source_weight.new_zeros((self.vocab_size, source_weight.shape[1]))
                weight[: ids.numel()].copy_(source_weight.index_select(0, ids))
                bias = None
                if source_bias is not None:
                    bias = source_bias.new_zeros(self.vocab_size)
                    bias[: ids.numel()].copy_(source_bias.index_select(0, ids))
                stage_weights.append(weight)
                stage_biases.append(bias)
                proxies.append(SlicedLMHeadProxy(lm_head, weight, self.vocab_size))
        self.stage_weights = tuple(stage_weights)
        self.stage_biases = tuple(stage_biases)
        self.proxies = tuple(proxies)
        self.weight_shapes = tuple(tuple(weight.shape) for weight in self.stage_weights)
        self.device_memory_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.stage_weights, *self.stage_biases)
            if tensor is not None
        )

    def refresh(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        """Refresh values in place so captured graph addresses stay valid."""
        # Model reloads can write inference tensors. Gather every stage before
        # changing any of the persistent projection buffers.
        with torch.inference_mode():
            weights = tuple(weight.index_select(0, ids) for ids in self._stage_token_ids)
            biases = (
                tuple(bias.index_select(0, ids) for ids in self._stage_token_ids)
                if bias is not None
                else ()
            )
            for stage, values in enumerate(weights):
                self.stage_weights[stage][: values.shape[0]].copy_(values)
            for stage, values in enumerate(biases):
                cached_bias = self.stage_biases[stage]
                assert cached_bias is not None
                cached_bias[: values.shape[0]].copy_(values)

    def close(self) -> None:
        self.stage_weights = ()
        self.stage_biases = ()
        self.proxies = ()
        self._stage_token_ids = ()
