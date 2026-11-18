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

    def supports_worker_beam_decision(self) -> bool:
        return True

    def uses_fused_beam_decision(self) -> bool:
        """The fused decision operators exist only on the fused decode path."""
        if self.decode_mode() != _XLLM_DECODE_MODE:
            return False
        from vllm_gr.ops.npu import beam_ops

        return bool(beam_ops.is_available())

    def requires_beam_candidate_width_match(self) -> bool:
        # The fused decision operators hard-code one survivor per Beam row.
        return True

    def batches_beam_kv_reorder(self) -> bool:
        # One request-major call keeps the whole B>1 geometry in the operator.
        return True

    def requires_aligned_final_select(self) -> bool:
        # The Ascend final-select kernel tiles rows in aligned blocks.
        return True


class AscendBeamAttentionService:
    """Own Ascend Beam attention selection and admissibility.

    Decode and Prefill are resolved separately because the fused ``xllm`` path
    serves Decode only: Prefill always needs the cascade module. Shared code
    asks for the path it is about to run instead of branching on a platform
    constant.
    """

    backend_id = "ascend"

    def device_type(self) -> str:
        return "npu"

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

    def short_suffix_head_size_supported(self, head_size: int) -> bool:
        del head_size
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


class AscendConfigService:
    """Ascend's configuration-time policy, kept out of shared configuration code."""

    backend_id = "ascend"

    def apply_config_defaults(self, config: Any) -> None:
        """Pin the Ascend-compatible block size without overriding user intent.

        vLLM-Ascend 0.30 sets the block size to 128 when prefix caching or
        chunked prefill is enabled, but leaves the platform default in place
        otherwise, while the Beam attention backends require the 128 kernel
        geometry. This closes that gap.
        """
        cache_config = getattr(config, "cache_config", None)
        model_config = getattr(config, "model_config", None)
        if cache_config is None:
            return
        if getattr(cache_config, "user_specified_block_size", False):
            return
        # Hybrid attention/state models have model-specific cache sizing rules
        # in vLLM-Ascend and must not be forced onto the dense default.
        if model_config is not None and getattr(model_config, "is_hybrid", False):
            return
        if cache_config.block_size != 128:
            import logging

            logging.getLogger("vllm_gr.platforms.ascend_services").info(
                "Using Ascend-compatible default KV cache block size 128 "
                "(vLLM resolved the unspecified value to %s).",
                cache_config.block_size,
            )
            cache_config.block_size = 128

    def supports_worker_decision_by_default(self) -> bool:
        return True

    def promotes_constraint_tables(self) -> bool:
        # Ascend Beam deployments default to Constraint Tables plus worker
        # decisions when the user did not choose a backend explicitly.
        return True

    def supports_constraint_vocab_slicing(self) -> bool:
        # Ascend keeps the original full-vocabulary ABI and rejects slicing.
        return False

    def validate_unconstrained_runner(self, config: Any) -> None:
        from vllm_gr.v1.engine.beam_search_v1 import validate_npu_unconstrained_runner

        validate_npu_unconstrained_runner(config)

    def requires_persistent_v2_protocol(self) -> bool:
        # Ascend may retain the V2 opt-in while constructing a V1 runner.
        return False

    def install_engine_core_config_sanitizer(self) -> None:
        """Filter GR keys before vllm-ascend validates ``additional_config``.

        vllm-ascend wraps ``EngineCoreProc.run_engine_core`` and validates the
        config with pydantic ``extra="forbid"`` in the spawned EngineCore
        process before the vLLM-GR plugin loads, so the filter must be the
        outermost entrypoint.
        """
        from vllm.v1.engine.core import EngineCoreProc

        from vllm_gr.v1.ascend_compat import apply_engine_core_config_sanitizer

        apply_engine_core_config_sanitizer(EngineCoreProc)

    def default_constraint_table_backend(self) -> str | None:
        return "npu"


class AscendBeamMetadataProfile:
    """Ascend's answers for shared Beam metadata construction."""

    backend_id = "ascend"

    def allocates_fixed_buffers(self) -> bool:
        # ACL Graph FULL-mode Beam decode reads them; the cascade path ignores
        # the device fields but still tolerates their presence.
        return True

    def needs_causal_mask_buffer(self) -> bool:
        # BSND padded Prefill consumes a static causal mask.
        return True

    def uses_v2_prefill_capture(self) -> bool:
        # Ascend has no native V2 Prefill capture in this release.
        return False

    def forces_single_reduction_split(self) -> bool:
        return False

    def packs_seqlens_non_blocking(self) -> bool:
        # Ascend stages seqlens through a pinned host buffer as well.
        return True

    def allows_identity_block_table_slice(self) -> bool:
        # A request-major incremental batch already holds its blocks in order.
        return True

    def marks_identity_prefix_rows(self) -> bool:
        # The fused path can reuse dense rows when routing is proven identity.
        return True

    def writes_seqlens_per_tensor(self) -> bool:
        return False


__all__ = [
    "AscendBeamAttentionService",
    "AscendBeamMetadataProfile",
    "AscendConfigService",
    "AscendStageExecutionService",
]
