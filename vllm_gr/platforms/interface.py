# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GR extension points on vLLM's native platform interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from vllm.platforms.interface import Platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_gr.platforms.admission import BeamLimits
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


class BeamAttentionExecution(Protocol):
    """Platform-specific Beam Attention execution primitives."""

    def is_available(self) -> bool: ...

    def get_flash_attn_version(
        self,
        *,
        requires_alibi: bool = False,
        head_size: int | None = None,
    ) -> int | None: ...

    def supports_sinks(self) -> bool: ...

    def reshape_and_cache(
        self,
        key: Any,
        value: Any,
        key_cache: Any,
        value_cache: Any,
        slot_mapping: Any,
        kv_cache_dtype: str,
        k_scale: Any,
        v_scale: Any,
    ) -> None: ...

    def cache_unshared_kv(
        self,
        unshared_key: Any,
        unshared_value: Any,
        curr_key: Any,
        curr_value: Any,
        slot: int,
        decode_step: int,
        beam_session: Any = None,
    ) -> None: ...

    def prefill_attention(
        self,
        output: Any,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        attn_metadata: Any,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: Any = None,
        alibi_slopes: Any = None,
    ) -> None: ...

    def cascade_attention(
        self,
        output: Any,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        attn_metadata: Any,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: Any = None,
        alibi_slopes: Any = None,
        beam_pool: Any = None,  # Per-layer pool (GPU incremental decode only)
    ) -> None: ...


class GRPlatform(Platform):
    """Select concrete GR implementations; engine state belongs to the runner."""

    api_version = 3
    native_platform_type: type[Platform] | None = None
    backend_id = "unspecified"
    plugin_distribution: str | None = None
    vendor_distribution: str | None = None
    beam_attention_layout: str | None = None
    canonical_constraint_backend = "platform"
    default_use_v2_model_runner: bool | None = None
    default_worker_decision = False
    supports_sid_stage_vocab = False
    supports_unconstrained = False
    supports_dynamic_width = False

    def get_gr_worker_cls(self) -> str:
        raise NotImplementedError(f"{self.backend_id} has no GR Worker")

    def check_environment(self) -> dict[str, str]:
        from .compat import check_vllm_runtime

        return check_vllm_runtime()

    def prepare_gr_engine(self) -> int:
        return 0

    def check_and_update_gr_config(self, config: VllmConfig) -> None:
        """Validate platform-specific settings after native configuration."""

    def get_beam_limits(self, config: VllmConfig) -> BeamLimits:
        from .admission import BeamLimits
        from .identity import qualified_class_name

        return BeamLimits(
            device_type=self.device_type,
            platform_class=qualified_class_name(type(self)),
            max_decode_rows=None,
            retain_candidate_widths=False,
            constraint_backend=self.canonical_constraint_backend,
            runner_generation="v2" if getattr(config, "use_v2_model_runner", False) else "v1",
            local_kv_heads=None,
            head_size=None,
            supports_unconstrained=self.supports_unconstrained,
            supports_dynamic_width=self.supports_dynamic_width,
            supports_sid_stage_vocab=self.supports_sid_stage_vocab,
        )

    def validate_gr_request(self, config: VllmConfig, request: BeamBatchRequest | None) -> None:
        """Additional device checks, run by the authoritative EngineCore."""

    @classmethod
    def get_gr_attention_backend_cls(
        cls, selected_backend, attn_selector_config, num_heads: int | None = None
    ) -> str:
        return "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"

    def get_attention_execution(self) -> BeamAttentionExecution:
        raise NotImplementedError(f"{self.backend_id} has no Beam attention implementation")

    def create_constraint_backend(self, table, *, device, dtype, max_rows, max_top_k):
        """Build a worker-owned canonical constraint operator from a validated table."""
        raise NotImplementedError(f"{self.backend_id} has no canonical constraint backend")

    def get_prefill_graph_runner_class(self) -> type:
        raise NotImplementedError(f"{self.backend_id} has no GR prefill graph runner")


class UnspecifiedGRPlatform(GRPlatform):
    """Import and CLI support without an available GR execution platform."""

    device_type = "cpu"
