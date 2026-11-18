# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Conservative adapter for providers written against API version 1."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from .errors import InvalidBackendProviderError
from .identity import PlatformIdentity
from .protocol import (
    LEGACY_PROVIDER_API_VERSION,
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    ModelRunnerPatchHooks,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("vllm_gr.platforms.legacy")


def declared_api_version(provider: Any) -> int:
    """Read a provider's declared API version, defaulting an absent one to v1.

    Version detection is explicit: an object is never classified by which
    optional attributes happen to exist.
    """
    declared = getattr(provider, "api_version", LEGACY_PROVIDER_API_VERSION)
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise InvalidBackendProviderError(
            "Backend provider api_version must be an integer; "
            f"found={declared!r}; source={getattr(provider, 'device_type', None)!r}"
        )
    return declared


@dataclass(frozen=True)
class LegacyProviderAdapter:
    """Adapt an API-v1 provider without upgrading it or inferring capabilities.

    Only the two fixed identities that shipped with version 1 are recognized:
    ``cuda`` for NVIDIA CUDA and ``npu`` for Ascend. A version-1 provider for
    any other device type must be re-registered by its distribution against API
    version 2; there is no inference from its installed attributes.
    """

    legacy: Any
    source: str
    backend_id: str
    api_version: ClassVar[int] = PROVIDER_API_VERSION
    gr_version_range: ClassVar[None] = None
    vllm_version_range: ClassVar[None] = None
    vendor_version_range: ClassVar[None] = None
    vendor_distribution: ClassVar[None] = None

    @property
    def device_type(self) -> str:
        """The version-1 identity this adapter was created for."""
        return cast(str, self.legacy.device_type)

    def matches(self, identity: PlatformIdentity) -> bool:
        """Match exactly the fixed version-1 device identity, nothing else."""
        if self.device_type == "cuda":
            # Version 1 selected every "cuda" placement as NVIDIA. Preserve that
            # behavior for version-1 providers, but say so, because API version 2
            # distinguishes ROCm from CUDA.
            if identity.platform_enum == "ROCM":
                logger.warning(
                    "API-v1 backend %r matched ROCm through the legacy 'cuda' "
                    "alias; the provider may select NVIDIA-only kernels. "
                    "Upgrade the provider to API version 2 and match ROCM explicitly.",
                    self.source,
                )
            return identity.device_type == "cuda"
        if self.device_type == "npu":
            return identity.device_type == "npu"
        return False

    def check_environment(self) -> dict[str, str]:
        return cast(dict[str, str], self.legacy.check_environment())

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        return cast(int, self.legacy.apply_runtime_patches(force=force))

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(ModelRunnerPatchHooks, self.legacy.get_model_runner_hooks())

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(BeamAttentionExecution, self.legacy.get_attention_execution())

    def select_runner_patch_module_name(self, *, prefer_v2: bool = False) -> str:
        """Return the fixed patch module of the version-1 identity.

        A version-1 provider never declared a runner patch module, so the alias
        that version 1 used for selection still applies. Capabilities are not
        inferred from optional attributes: the V2 opt-in is accepted only where
        version 1 already honored it.
        """
        del prefer_v2
        if self.device_type == "npu":
            return "vllm_gr.v1.worker.npu_model_runner_patch"
        return "vllm_gr.v1.worker.gpu_model_runner_patch"

    def supports_v2_model_runner(self) -> bool:
        """A version-1 provider never declared a V2 runner module.

        API version 1 predates the V2 patch layer, so the opt-in is not honored
        rather than inferred from the provider's device type.
        """
        return False

    def create_stage_execution_service(self) -> Any:
        """Return the fixed stage service of the version-1 identity."""
        if self.device_type == "npu":
            from .ascend_services import AscendStageExecutionService

            return AscendStageExecutionService()
        from .cuda_services import CudaStageExecutionService

        return CudaStageExecutionService()

    def create_beam_attention_service(self) -> Any:
        """Return the fixed Beam attention service of the version-1 identity."""
        if self.device_type == "npu":
            from .ascend_services import AscendBeamAttentionService

            return AscendBeamAttentionService()
        from .cuda_services import CudaBeamAttentionService

        return CudaBeamAttentionService()

    def create_config_service(self) -> Any:
        """Return the fixed configuration policy of the version-1 identity."""
        if self.device_type == "npu":
            from .ascend_services import AscendConfigService

            return AscendConfigService()
        from .cuda_services import CudaConfigService

        return CudaConfigService()

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> Any:
        return self.legacy.create_constrained_topk_execution(device=device)


def adapt_legacy_provider(provider: Any, *, source: str) -> LegacyProviderAdapter | None:
    """Return a version-1 adapter, or ``None`` for a provider that is not v1."""
    if declared_api_version(provider) != LEGACY_PROVIDER_API_VERSION:
        return None

    device_type = getattr(provider, "device_type", None)
    backend_id = {"cuda": "cuda", "npu": "ascend"}.get(
        device_type.strip().lower() if isinstance(device_type, str) else ""
    )
    if backend_id is None:
        raise InvalidBackendProviderError(
            "A vllm-gr backend provider without api_version uses the deprecated "
            "API version 1 and can only be adapted for the fixed identities "
            f"'cuda' and 'npu'; found device_type={device_type!r}; source={source!r}. "
            "Upgrade the provider to API version 2 and declare backend_id, "
            "api_version=2 and matches(identity)."
        )

    logger.warning(
        "vllm-gr backend provider %r (%r) uses the deprecated API version 1 "
        "adapter; backend_id=%r. Declare api_version=%d, backend_id and "
        "matches(identity) before the next release.",
        source,
        provider,
        backend_id,
        PROVIDER_API_VERSION,
    )
    return LegacyProviderAdapter(legacy=provider, source=source, backend_id=backend_id)
