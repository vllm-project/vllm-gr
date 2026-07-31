# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-neutral contracts for vllm-gr backend providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

BACKEND_API_VERSION = 1


class ModelRunnerPatchHooks(Protocol):
    """Platform-specific hooks used to patch vLLM ModelRunner classes."""

    def discover_worker_runners(self) -> tuple[tuple[str, type[Any]], ...]: ...

    def make_patched_prepare_inputs(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]: ...

    def make_patched_bookkeeping_sync(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]: ...


class BeamAttentionHooks(Protocol):
    """Complete platform contract used by the Beam Attention backend."""

    def is_available(self) -> bool: ...

    def get_flash_attn_version(
        self,
        *,
        requires_alibi: bool = False,
        head_size: int | None = None,
    ) -> int | None: ...

    def supports_sinks(self) -> bool: ...

    def supports_batch_invariance(self) -> bool: ...

    def supports_kv_cache_dtype(self, kv_cache_dtype: Any | None) -> bool: ...

    def supports_compute_capability(self, capability: Any) -> bool: ...

    def validate_combination(
        self,
        *,
        head_size: int,
        dtype: Any,
        kv_cache_dtype: Any,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: Any,
    ) -> str | None: ...

    def get_kv_cache_shape(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...

    def get_kv_cache_stride_order(
        self,
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]: ...

    def split_kv_cache(self, kv_cache: Any) -> tuple[Any, Any]: ...

    def get_kv_scales(self, layer: Any) -> tuple[Any, Any]: ...

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
    ) -> None: ...

    def get_metadata_builder_cls(self) -> type[Any]: ...


class GRBackendProvider(Protocol):
    """Process-local provider for one canonical vLLM device type."""

    @property
    def api_version(self) -> int: ...

    @property
    def device_type(self) -> str: ...

    def configure_engine(self, config: Any) -> None: ...

    def check_environment(self) -> dict[str, str]: ...

    def apply_runtime_patches(self, *, force: bool = False) -> int: ...

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks: ...

    def get_attention_hooks(self) -> BeamAttentionHooks: ...

    def release_accelerator_memory(self) -> None: ...
