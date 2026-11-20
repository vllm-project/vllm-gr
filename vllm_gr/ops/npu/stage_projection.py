# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Fixed stage-local LM-head projections for Ascend."""

from __future__ import annotations

from collections import Counter
from typing import Any, cast

import torch

from vllm_gr.config.models import deployment_for
from vllm_gr.ops.sliced_logits import (
    SlicedLMHeadProxy,
    is_supported_unquantized_lm_head,
)

_MISSING = object()


def supported_projection_parts(runner: Any) -> tuple[Any, Any] | None:
    """Check model eligibility before allocating optional device tables."""
    settings = deployment_for(getattr(runner, "vllm_config", None))
    if not settings.npu_stage_lm_head:
        return None
    config = getattr(runner, "vllm_config", None)
    parallel = getattr(config, "parallel_config", None)
    if (
        getattr(parallel, "tensor_parallel_size", 1) != 1
        or getattr(parallel, "pipeline_parallel_size", 1) != 1
        or getattr(runner, "lora_config", None)
        or getattr(config, "lora_config", None)
        or getattr(runner, "speculative_config", None) is not None
        or getattr(config, "speculative_config", None) is not None
    ):
        return None
    model = getattr(runner, "model", None)
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not is_supported_unquantized_lm_head(lm_head)
    ):
        return None
    bias = getattr(lm_head, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.ndim != 1
        or bias.shape[0] != weight.shape[0]
        or bias.dtype != weight.dtype
        or bias.device != weight.device
    ):
        return None
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    modules = getattr(model, "modules", None)
    if not callable(modules):
        return None
    processors = tuple(module for module in modules() if isinstance(module, LogitsProcessor))
    if len(processors) != 1:
        return None
    return lm_head, processors[0]


class NPUStageProjection:
    """Own three fixed projection weights and one reversible processor hook.

    The original processor still applies its native projection method; its
    outer forward retains logits scaling, soft capping, and TP handling.
    Only a currently executing, fixed-stage Beam dispatch may use these
    weights. Ordinary requests and startup dummy runs keep the full LM head.
    """

    def __init__(self, runner: Any, processor: Any, lm_head: Any, stage_vocab: Any) -> None:
        self.runner = runner
        self.processor = processor
        self.lm_head = lm_head
        self.vocab_size = int(stage_vocab.vocab_size)
        self.projection_counts = [0, 0, 0]
        self.fallback_counts: Counter[str] = Counter()
        self.closed = False
        self.weights_valid = True
        self.update_active = False
        self.update_failed = False
        self.refresh_count = 0
        self._original = processor._get_logits
        self._previous_method = processor.__dict__.get("_get_logits", _MISSING)
        self._previous_window = processor.__dict__.get("_gr_vocab_slice", _MISSING)
        self._source_bias = getattr(lm_head, "bias", None)
        self._source_weight = lm_head.weight
        self._source_weight_shape = tuple(lm_head.weight.shape)
        self._source_quant_method = getattr(lm_head, "quant_method", None)
        self._stage_token_ids = stage_vocab.stage_token_ids
        self.stage_token_counts = tuple(int(ids.numel()) for ids in stage_vocab.stage_token_ids)
        stage_weights = []
        stage_biases = []
        proxies = []
        with torch.no_grad():
            for ids in stage_vocab.stage_token_ids:
                weight = lm_head.weight.new_zeros((self.vocab_size, lm_head.weight.shape[1]))
                weight[: ids.numel()].copy_(lm_head.weight.index_select(0, ids))
                bias = None
                if self._source_bias is not None:
                    bias = self._source_bias.new_zeros(self.vocab_size)
                    bias[: ids.numel()].copy_(self._source_bias.index_select(0, ids))
                stage_weights.append(weight)
                stage_biases.append(bias)
                proxies.append(SlicedLMHeadProxy(lm_head, weight, self.vocab_size))
        self.stage_weights = tuple(stage_weights)
        self._stage_biases = tuple(stage_biases)
        self._proxies = tuple(proxies)
        self.weight_shapes = tuple(tuple(weight.shape) for weight in self.stage_weights)
        self.device_memory_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.stage_weights, *self._stage_biases)
            if tensor is not None
        )

    def install(self) -> None:
        # Explicit None prevents legacy process-global slicing from being
        # applied a second time to a compact projection (or the fallback).
        self.processor._gr_vocab_slice = None
        self.processor._get_logits = self._get_logits

    def begin_weight_update(self) -> None:
        """Invalidate before native code can write any model parameter."""
        if self.closed or self.update_failed:
            raise RuntimeError("NPU stage LM head requires a complete model reload after failure")
        if self.update_active:
            raise RuntimeError("NPU stage LM head weight update is already active")
        self.update_active = True
        self.weights_valid = False

    def fail_weight_update(self) -> None:
        """Never reuse a cache after a possibly partial native update."""
        self.weights_valid = False
        self.update_failed = True

    def _require_weight_update(self) -> None:
        if self.closed or self.update_failed or not self.update_active:
            raise RuntimeError("NPU stage LM head requires an active, successful weight update")

    def finish_weight_update(self) -> None:
        """Refresh only after native checkpoint finalization has returned."""
        self._require_weight_update()
        try:
            self.refresh()
        except BaseException:
            self.fail_weight_update()
            raise
        self.update_active = False

    def refresh(self) -> bool:
        """Refresh fixed cache addresses at a quiescent native update boundary.

        Native reload APIs must run without concurrent inference. Their writes
        and our copies use the current stream; synchronization here also makes
        refreshed weights visible to subsequent serving streams. There is no
        synchronization or source-tensor inspection in ordinary projection.
        Direct model.load_weights/parameter writes outside these native APIs
        require a complete model reload; they are not an update transaction.
        """
        if self.closed or self.update_failed:
            raise RuntimeError("Cannot refresh NPU stage LM head after a failed update")
        self.weights_valid = False
        parts = supported_projection_parts(self.runner)
        if parts is None or parts[0] is not self.lm_head or parts[1] is not self.processor:
            return False
        weight = self.lm_head.weight
        bias = getattr(self.lm_head, "bias", None)
        if (
            tuple(weight.shape) != self._source_weight_shape
            or weight.dtype != self.stage_weights[0].dtype
            or weight.device != self.stage_weights[0].device
            or getattr(self.lm_head, "quant_method", None) is not self._source_quant_method
            or (bias is None) != (self._source_bias is None)
        ):
            return False
        try:
            # Loaded parameters and cached weights can be inference tensors.
            # Prepare every stage before changing any persistent cache tensor.
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
                    self._stage_biases[stage][: values.shape[0]].copy_(values)
                if weight.device.type == "npu":
                    torch.npu.synchronize(weight.device)
        except BaseException:
            self.fail_weight_update()
            raise
        self._source_weight = weight
        self._source_bias = bias
        self.refresh_count += 1
        self.weights_valid = True
        return True

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: Any,
        embedding_bias: torch.Tensor | None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        if self.update_active or self.update_failed:
            raise RuntimeError(
                "NPU stage LM head cannot run during or after a failed weight update"
            )
        reason = None
        plan = getattr(self.runner, "_gr_npu_stage_plan", None)
        owner = getattr(self.runner, "_gr_npu_stage_runner", None)
        execution = getattr(plan, "execution", None)
        stage = getattr(execution, "stage_index", None)
        rows = getattr(execution, "num_execution_rows", None)
        if self.closed or plan is None or getattr(owner, "pending_plan", None) is not plan:
            reason = "no_active_stage"
        elif not self.weights_valid:
            reason = "weights_invalid"
        elif isinstance(stage, bool) or not isinstance(stage, int) or stage not in (0, 1, 2):
            reason = "unsupported_stage"
        elif hidden_states.ndim != 2 or hidden_states.shape[0] != rows:
            reason = "row_mismatch"
        elif lm_head is not self.lm_head:
            reason = "different_lm_head"
        elif (
            hidden_states.device != self.stage_weights[stage].device
            or hidden_states.dtype != self.stage_weights[stage].dtype
        ):
            reason = "hidden_layout"
        elif embedding_bias is not None and embedding_bias is not self._source_bias:
            reason = "different_bias"
        elif getattr(lm_head, "weight", None) is not self._source_weight:
            reason = "different_weight"
        if reason is not None:
            self.fallback_counts[reason] += 1
            return self._original(hidden_states, lm_head, embedding_bias, skip_gather)
        stage_index = cast(int, stage)
        bias = self._stage_biases[stage_index] if embedding_bias is not None else None
        logits = self._original(hidden_states, self._proxies[stage_index], bias, skip_gather)
        if logits is not None and logits.shape[-1] != self.vocab_size:
            raise RuntimeError("NPU stage LM head returned an unexpected projection width")
        self.projection_counts[stage_index] += 1
        return logits

    def snapshot(self) -> dict[str, Any]:
        """Return host-only diagnostics without touching device tensors."""
        return {
            "enabled": not self.closed and self.weights_valid,
            "weights_valid": self.weights_valid,
            "update_active": self.update_active,
            "update_failed": self.update_failed,
            "refresh_count": self.refresh_count,
            "vocab_size": self.vocab_size,
            "stage_token_counts": list(self.stage_token_counts),
            "weight_shapes": [list(shape) for shape in self.weight_shapes],
            "projection_counts": list(self.projection_counts),
            "fallback_counts": dict(self.fallback_counts),
            "device_memory_bytes": self.device_memory_bytes,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.weights_valid = False
        if self._previous_method is _MISSING:
            delattr(self.processor, "_get_logits")
        else:
            self.processor._get_logits = self._previous_method
        if self._previous_window is _MISSING:
            delattr(self.processor, "_gr_vocab_slice")
        else:
            self.processor._gr_vocab_slice = self._previous_window
        self.stage_weights = ()
        self._stage_biases = ()
        self._proxies = ()
        self._source_bias = None
        self._source_weight = None
        self._source_quant_method = None
        self._stage_token_ids = ()
        self._original = None
        self._previous_method = _MISSING
        self._previous_window = _MISSING
        self.lm_head = None
        self.processor = None
        self.runner = None
