# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA stage-execution service.

Shared stage dispatch asks the selected provider for this service instead of
comparing ``worker.device`` against ``"cuda"``. Every CUDA module import stays
inside a method body, so importing this service still loads no CUDA module and
``vllm_gr.platforms`` keeps its import isolation guarantee.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_gr.v1.worker.gpu_stage_hooks import CudaStageHooksType


class CudaStageExecutionService:
    """Execute, retire and pre-capture CUDA GR Beam stages."""

    backend_id = "cuda"

    def execute(
        self,
        worker: Any,
        scheduled: Any,
        original: Callable[..., Any],
        *args: Any,
        hooks_type: CudaStageHooksType | None = None,
        **kwargs: Any,
    ) -> Any:
        from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner

        if hooks_type is None:
            raise ValueError("CUDA GR execution requires explicit stage hooks")
        stage_runner = get_stage_runner(worker, scheduled, hooks_type=hooks_type)
        try:
            return stage_runner.execute(scheduled, original, *args, **kwargs)
        except Exception as error:
            # Native BatchQueue consumes sample first. Stale logits must never
            # hide an execute failure from the corresponding sample Future.
            stage_runner.execution_error = error
            raise

    def retire(self, worker: Any, scheduled: Any) -> Any:
        from vllm_gr.v1.worker.gpu_beam_stage_runner import execute_gr_retirement

        return execute_gr_retirement(worker, scheduled)

    def capture_startup_graph(
        self,
        worker: Any,
        *,
        hooks_type: CudaStageHooksType | None = None,
        allow_eager_fallback: bool = False,
    ) -> bool:
        from vllm_gr.v1.worker.gr_startup import capture_cuda_startup_graph

        return capture_cuda_startup_graph(
            worker,
            hooks_type=hooks_type,
            allow_eager_fallback=allow_eager_fallback,
        )

    def admit_unconstrained_beam(self, worker: Any) -> None:
        """CUDA serves every unconstrained Beam V1 configuration this release admits."""
        del worker

    def supports_worker_beam_decision(self) -> bool:
        # CUDA V1 decisions are made by its stage runner, never worker-side.
        return False

    def uses_fused_beam_decision(self) -> bool:
        # The fused decision operators are Ascend-only; CUDA uses the reference.
        return False

    def requires_beam_candidate_width_match(self) -> bool:
        return False

    def batches_beam_kv_reorder(self) -> bool:
        # CUDA reorders each request's pool separately.
        return False

    def requires_aligned_final_select(self) -> bool:
        # CUDA final selection has no row-tiling constraint.
        return False


class CudaBeamAttentionService:
    """Own CUDA Beam attention selection and admissibility.

    Shared Beam attention asks this service instead of reading platform
    constants. FlashAttention availability still lives in the execution module;
    this type only reports it to shared code under a backend-neutral contract.
    """

    backend_id = "cuda"

    def execution_module(self) -> Any:
        self._reject_npu_only_decode_mode()
        from vllm_gr.v1.attention.backends import beam_attn_gpu

        return beam_attn_gpu

    def prefill_module(self) -> Any:
        # CUDA serves Prefill and Decode from the same module.
        return self.execution_module()

    def decode_mode(self) -> str:
        self._reject_npu_only_decode_mode()
        return "cascade"

    def _reject_npu_only_decode_mode(self) -> None:
        """Fail when a fused Ascend-only Decode path was requested on CUDA."""
        if _configured_decode_mode() == "xllm":
            raise RuntimeError(
                "vllm_gr_npu_decode_execution='xllm' requires an Ascend NPU; "
                "unset the option or select 'cascade' on CUDA."
            )

    def supports_sinks(self) -> bool:
        from vllm_gr.v1.attention.backends import beam_attn_gpu

        return bool(beam_attn_gpu.is_available() and beam_attn_gpu.supports_sinks())

    def supports_fp8_kv_cache(self) -> bool:
        from vllm.platforms import current_platform

        # FP8 KV cache needs an FA3 decode path on the 90-class devices; the
        # capability test belongs to the backend, not to shared attention.
        if self.execution_module().get_flash_attn_version() != 3:
            return False
        return bool(current_platform.is_device_capability_family(90))

    def supports_batch_invariance(self) -> bool:
        return True

    def requires_flash_attn_version(self) -> bool:
        return True

    def supports_short_suffix_attention(self) -> bool:
        return True

    def supports_prefill_version_selection(self) -> bool:
        return True

    def supports_compute_capability(self, capability: Any) -> bool:
        from vllm.platforms.interface import DeviceCapability

        return bool(capability >= DeviceCapability(8, 0))

    def requires_layer_kv_scale(self) -> bool:
        return True

    def uses_bsnd_prefill(self, attn_metadata: Any) -> bool:
        del attn_metadata
        return False

    def beam_row_selector(self) -> Any:
        # CUDA always gathers the Beam rows; there is no identity fast path.
        return None

    def validate_impl_features(
        self,
        *,
        sliding_window: Any,
        alibi_slopes: Any,
        logits_soft_cap: Any,
        has_sinks: bool,
    ) -> None:
        # CUDA's FlashAttention paths implement all of these settings.
        del sliding_window, alibi_slopes, logits_soft_cap, has_sinks

    def resolve_prefill_version(
        self,
        requested: Any,
        decode_version: Any,
        **kwargs: Any,
    ) -> Any:
        if requested is None:
            return decode_version
        from vllm_gr.v1.attention.backends.beam_attn_gpu import (
            resolve_prefill_flash_attn_version,
        )

        return resolve_prefill_flash_attn_version(requested, decode_version, **kwargs)

    def update_standard_graph_params(
        self,
        update_stream: Any,
        forward_context: Any,
        num_tokens: Any,
    ) -> None:
        # CUDA refreshes persistent Beam buffers; it has no standard task group.
        del update_stream, forward_context, num_tokens

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
        del output, query, key_cache, value_cache, attn_metadata, scale
        del layer_name, max_model_len
        return False

    def has_standard_prefill_capture(self) -> bool:
        # CUDA captures no native Prefill task groups; Beam owns its own graph.
        return False


def _configured_decode_mode() -> str | None:
    """Read the configured NPU decode mode without importing any vendor code."""
    from vllm.config import get_current_vllm_config_or_none

    from vllm_gr.config import GRConfig

    configured = GRConfig.from_vllm_config(get_current_vllm_config_or_none()).npu_decode_execution
    return None if configured is None else str(configured.value)


class CudaConfigService:
    """CUDA's configuration-time policy, kept out of shared configuration code."""

    backend_id = "cuda"

    def apply_config_defaults(self, config: Any) -> None:
        # CUDA pins no block size: the native default is the kernel geometry.
        del config

    def supports_worker_decision_by_default(self) -> bool:
        return True

    def promotes_constraint_tables(self) -> bool:
        # Promotion is an Ascend deployment default; CUDA keeps the request.
        return False

    def supports_constraint_vocab_slicing(self) -> bool:
        # CUDA carries full-vocabulary table ids with a sliced logits window.
        return True

    def validate_unconstrained_runner(self, config: Any) -> None:
        del config

    def requires_persistent_v2_protocol(self) -> bool:
        return True

    def install_engine_core_config_sanitizer(self) -> None:
        # Only a vendor platform validates additional_config before GR loads.
        return None

    def default_constraint_table_backend(self) -> str | None:
        return "cuda"


__all__ = [
    "CudaBeamAttentionService",
    "CudaConfigService",
    "CudaStageExecutionService",
]
