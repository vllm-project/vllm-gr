# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Hardware-neutral contracts for vllm-gr backend providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import ConstrainedTopKExecution

    from .identity import PlatformIdentity
    from .services import StageExecutionService

#: Provider API implemented by built-in and external providers in this release.
PROVIDER_API_VERSION = 2

#: API version assumed for entry points that do not declare one.
LEGACY_PROVIDER_API_VERSION = 1


class ModelRunnerPatchHooks(Protocol):
    """Required hooks for the V1 ModelRunner patch dispatcher.

    Additional factories, including ``make_patched_bookkeeping_sync``, are
    optional capabilities discovered by the dispatcher. CUDA keeps native
    bookkeeping; NPU supplies its six-item bookkeeping wrapper.
    """

    def discover_worker_runners(self) -> tuple[tuple[str, type[Any]], ...]: ...

    def make_patched_prepare_inputs(
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
        beam_pool: Any = None,  # Per-layer pool (GPU incremental decode only)
    ) -> None: ...


class GRBackendProvider(Protocol):
    """Process-local provider for one stable GR backend identity."""

    @property
    def backend_id(self) -> str: ...

    @property
    def device_type(self) -> str:
        """Tensor placement of the supported platform; not the identity."""

    @property
    def api_version(self) -> int: ...

    @property
    def gr_version_range(self) -> str | None: ...

    @property
    def vllm_version_range(self) -> str | None: ...

    @property
    def vendor_version_range(self) -> str | None: ...

    @property
    def vendor_distribution(self) -> str | None: ...

    def matches(self, identity: PlatformIdentity) -> bool: ...

    def check_environment(self) -> dict[str, str]: ...

    def apply_runtime_patches(self, *, force: bool = False) -> int: ...

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks: ...

    def select_runner_patch_module_name(self, *, prefer_v2: bool = False) -> str:
        """Return the module name that owns this backend's ModelRunner patches.

        Only a backend with a ModelRunner V2 implementation honors ``prefer_v2``.
        Answering here keeps the V2 flag and the device name from being combined
        in shared dispatch.
        """
        ...

    def supports_v2_model_runner(self) -> bool:
        """Whether this backend implements a ModelRunner V2 patch module.

        Shared dispatch consults this before reading the V2 opt-in, so the flag
        is never evaluated on a backend that has no V2 runner.
        """
        ...

    def get_attention_execution(self) -> BeamAttentionExecution: ...

    def create_stage_execution_service(self) -> "StageExecutionService":
        """Return the bound GR stage execution service for this backend.

        Shared stage dispatch calls this instead of comparing ``worker.device``
        against a device name. The service owns executor selection, retirement
        and the backend's startup capture policy.
        """
        ...

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "ConstrainedTopKExecution | None": ...
