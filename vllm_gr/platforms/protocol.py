# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-neutral contracts for vllm-gr backend providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import ConstrainedTopKExecution


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


class GRBackendProvider(Protocol):
    """Process-local provider for one canonical vLLM device type."""

    @property
    def device_type(self) -> str: ...

    def check_environment(self) -> dict[str, str]: ...

    def apply_runtime_patches(self, *, force: bool = False) -> int: ...

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks: ...

    def get_attention_execution(self) -> BeamAttentionExecution: ...

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "ConstrainedTopKExecution | None": ...
