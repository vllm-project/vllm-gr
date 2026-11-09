# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refresh derived head storage from the layer that actually loads its weight."""

from functools import wraps
from typing import Any

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


def _refresh_head_cache(head: Any, loaded_layer: Any) -> None:
    # A tied head may not be processed at all during native layerwise reload.
    # Its shared original Parameter is also still old until the loader copies
    # back, so use the loading layer's newly materialized weight here.
    with torch.inference_mode():
        if hasattr(head, "_sliced_weight"):
            offset, _, rank_start, _ = head._sliced_cache_key
            start = max(offset - rank_start, 0) - head._sliced_rel_start
            head._sliced_weight.copy_(
                loaded_layer.weight[start : start + head._sliced_weight.shape[0]]
            )
            if loaded_layer is head and head._sliced_bias is not None:
                head._sliced_bias.copy_(head.bias[start : start + head._sliced_bias.shape[0]])
        if hasattr(head, "_gr_unconstrained_head_layout"):
            weight, ready = head._gr_unconstrained_head_layout
            weight.copy_(loaded_layer.weight.detach().T)
            ready.record(torch.cuda.current_stream(weight.device))


def install_lm_head_cache_reload_hooks(model: Any) -> None:
    """Register once after model load, before projection caches are created."""
    for _, head in model.named_modules():
        # Pipeline placeholders and packed-only heads have no dense head cache.
        if not isinstance(head, ParallelLMHead) or not hasattr(head, "weight"):
            continue
        if getattr(head, "_gr_cache_reload_installed", False):
            continue
        for owner in model.modules():
            if getattr(owner, "weight", None) is not head.weight:
                continue
            method = getattr(owner, "quant_method", None)
            if method is None:
                continue
            _install_owner_hook(owner, head, method)
        head._gr_cache_reload_installed = True


def _install_owner_hook(owner: Any, head: Any, method: Any) -> None:
    original = method.process_weights_after_loading

    @wraps(original)
    def process_weights_after_loading(layer: Any) -> None:
        original(layer)
        if layer is owner:
            _refresh_head_cache(head, layer)

    method.process_weights_after_loading = process_weights_after_loading
