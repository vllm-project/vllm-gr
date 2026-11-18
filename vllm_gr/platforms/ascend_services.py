# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend stage-execution and Beam attention services.

Shared stage dispatch and shared Beam attention ask the selected provider for
these services instead of comparing ``worker.device`` against ``"npu"``. Every
Ascend module import stays inside a method body, so importing this service still
loads no vendor runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: Configuration value for the fused Ascend decode path.
_XLLM_DECODE_MODE = "xllm"


class AscendStageExecutionService:
    """Execute, retire and pre-capture Ascend GR Beam stages."""

    backend_id = "ascend"

    def execute(
        self,
        worker: Any,
        scheduled: Any,
        original: Callable[..., Any],
        *args: Any,
        hooks_type: Any = None,
        **kwargs: Any,
    ) -> Any:
        # The Ascend eager owner owns its own stage state, so the CUDA stage-hooks
        # contract does not apply. Accept and ignore the argument rather than
        # requiring shared dispatch to know which backend takes hooks.
        del hooks_type
        from vllm_gr.v1.worker.npu_beam_stage_runner import get_npu_stage_runner

        stage_runner = get_npu_stage_runner(worker, scheduled)
        try:
            return stage_runner.execute(scheduled, original, *args, **kwargs)
        except Exception as error:
            stage_runner.execution_error = error
            raise

    def retire(self, worker: Any, scheduled: Any) -> Any:
        from vllm_gr.v1.worker.npu_beam_stage_runner import (
            execute_npu_gr_retirement,
        )

        return execute_npu_gr_retirement(worker, scheduled)

    def capture_startup_graph(
        self,
        worker: Any,
        *,
        hooks_type: Any = None,
        allow_eager_fallback: bool = False,
    ) -> bool:
        del hooks_type, allow_eager_fallback
        from vllm_gr.v1.worker.npu_model_runner_patch import (
            capture_beam_decode_graph,
        )

        return bool(capture_beam_decode_graph(worker))

    def admit_unconstrained_beam(self, worker: Any) -> None:
        """Apply the Ascend-only device admission rules for unconstrained Beam."""
        from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

        validate_gr_device_config(worker.vllm_config)


class AscendBeamAttentionService:
    """Own Ascend Beam attention selection and admissibility.

    Decode and Prefill are resolved separately because the fused ``xllm`` path
    serves Decode only: Prefill always needs the cascade module. Shared code
    asks for the path it is about to run instead of branching on a platform
    constant.
    """

    backend_id = "ascend"

    def execution_module(self) -> Any:
        """Return the module serving Decode and ordinary attention."""
        if self.decode_mode() == _XLLM_DECODE_MODE:
            from vllm_gr.v1.attention.backends import beam_attn_npu_xllm

            return beam_attn_npu_xllm
        from vllm_gr.v1.attention.backends import beam_attn_npu

        return beam_attn_npu

    def prefill_module(self) -> Any:
        """Return the cascade module, which owns Ascend Prefill attention."""
        from vllm_gr.v1.attention.backends import beam_attn_npu

        return beam_attn_npu

    def decode_mode(self) -> str:
        """Resolve the configured Decode mode, applying the Ascend default."""
        from vllm.config import get_current_vllm_config_or_none

        from vllm_gr.config import GRConfig

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            # Config not installed yet in this process; the cascade path needs
            # no fused operator, so it is always a valid answer here.
            return "cascade"
        configured = GRConfig.from_vllm_config(vllm_config).npu_decode_execution
        if configured is not None:
            return str(configured.value)
        from vllm_gr.ops.npu import beam_ops

        if beam_ops.is_available():
            return _XLLM_DECODE_MODE
        return "cascade"

    def supports_sinks(self) -> bool:
        # Neither the cascade nor the xllm execution path implements sinks.
        return False

    def supports_fp8_kv_cache(self) -> bool:
        return False

    def supports_batch_invariance(self) -> bool:
        # Ascend has no CUDA-style deterministic reduction mode.
        return False

    def requires_flash_attn_version(self) -> bool:
        # The Ascend cascade path reports a placeholder version and does not
        # require one to be detected before Decode runs.
        return False

    def supports_short_suffix_attention(self) -> bool:
        # The fused short-suffix prefix path is a CUDA FlashAttention feature.
        return False

    def supports_prefill_version_selection(self) -> bool:
        # Ascend has one Prefill attention version; there is nothing to select.
        return False

    def supports_compute_capability(self, capability: Any) -> bool:
        # Ascend does not use CUDA compute capability; accept any value.
        del capability
        return True

    def requires_layer_kv_scale(self) -> bool:
        # The Ascend cache-write operators apply no layer scale.
        return False

    def uses_bsnd_prefill(self, attn_metadata: Any) -> bool:
        return getattr(attn_metadata, "prefill_fia_layout", "TND") == "BSND"

    def beam_row_selector(self) -> Any:
        """Return the xllm dense-row reuse hook when that path is selected."""
        if self.decode_mode() != _XLLM_DECODE_MODE:
            return None
        from vllm_gr.v1.attention.backends.beam_attn_npu_xllm import select_beam_rows

        return select_beam_rows

    def validate_impl_features(
        self,
        *,
        sliding_window: Any,
        alibi_slopes: Any,
        logits_soft_cap: Any,
        has_sinks: bool,
    ) -> None:
        unsupported = []
        if sliding_window is not None:
            unsupported.append("sliding_window")
        if alibi_slopes is not None:
            unsupported.append("alibi_slopes")
        if logits_soft_cap not in (None, 0.0):
            unsupported.append("logits_soft_cap")
        if unsupported:
            raise NotImplementedError(
                "NPU CUSTOM Beam Attention does not support "
                + ", ".join(unsupported)
                + "; neither the cascade nor xllm execution path implements these settings."
            )
        if has_sinks:
            raise NotImplementedError("Sinks are not yet supported on NPU.")

    def resolve_prefill_version(
        self,
        requested: Any,
        decode_version: Any,
        **kwargs: Any,
    ) -> Any:
        # Prefill FlashAttention versioning is a CUDA concept; an explicit
        # request is a configuration error rather than a silently ignored one.
        del kwargs
        if requested not in (None, "auto"):
            raise ValueError("beam.prefill_flash_attn_version is CUDA-only")
        return decode_version

    def update_standard_graph_params(
        self,
        update_stream: Any,
        forward_context: Any,
        num_tokens: Any,
    ) -> None:
        from vllm_gr.v1.attention.backends.npu_standard_graph import (
            update_standard_graph_params,
        )

        update_standard_graph_params(update_stream, forward_context, num_tokens)

    def capture_standard_prefill(
        self,
        output: Any,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        attn_metadata: Any,
        scale: float,
        layer_name: str,
        max_model_len: int,
    ) -> bool:
        """Capture the standard FIA task group during a native full-graph pass.

        Ascend prefill graphs capture ordinary attention tasks that the Beam
        path does not own, so shared attention must hand those to the backend
        instead of running its own Prefill kernel.
        """
        from vllm_gr.v1.attention.backends import beam_attn_npu
        from vllm_gr.v1.attention.backends.npu_standard_graph import (
            capture_standard_fia,
        )

        if beam_attn_npu._GRAPH_CAPTURE_STATE["capturing"]:
            return False
        capture_standard_fia(
            output,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            scale,
            layer_name,
            max_model_len,
        )
        return True

    def has_standard_prefill_capture(self) -> bool:
        # Ascend captures native FIA task groups for the ordinary Prefill path.
        return True


__all__ = ["AscendBeamAttentionService", "AscendStageExecutionService"]
