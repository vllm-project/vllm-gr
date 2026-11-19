# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Conservative adapter for providers written against API version 1."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

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

    from .config import BackendConfigService

logger = logging.getLogger("vllm_gr.platforms.legacy")


class _LegacyPlacement(Protocol):
    backend_id: str

    def create_config_service(self) -> BackendConfigService: ...

    def matches_legacy_identity(self, identity: PlatformIdentity) -> bool: ...


def _legacy_placement(device_type: str | None) -> _LegacyPlacement | None:
    from .loader import core_entry_points

    for entry in core_entry_points("vllm_gr.legacy_placements"):
        if entry.name == device_type:
            return cast(_LegacyPlacement, entry.load())
    return None


def legacy_config_service(device_type: str | None) -> BackendConfigService:
    """Preserve native configuration defaults for fixed API-v1 placements."""
    from .config import BackendConfigService

    placement = _legacy_placement(device_type)
    return placement.create_config_service() if placement is not None else BackendConfigService()


def declared_api_version(provider: Any) -> int:
    """Read a provider's declared API version, defaulting an absent one to v1.

    Version detection is explicit: an object is never classified by which
    optional attributes happen to exist.
    """
    declared: object = getattr(provider, "api_version", LEGACY_PROVIDER_API_VERSION)
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
        placement = _legacy_placement(self.device_type)
        return placement is not None and placement.matches_legacy_identity(identity)

    def check_environment(self) -> dict[str, str]:
        return cast(dict[str, str], self.legacy.check_environment())

    def create_config_service(self) -> BackendConfigService:
        return legacy_config_service(self.device_type)

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        return cast(int, self.legacy.apply_runtime_patches(force=force))

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(ModelRunnerPatchHooks, self.legacy.get_model_runner_hooks())

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(BeamAttentionExecution, self.legacy.get_attention_execution())

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

    from .registry import _REQUIRED_PROVIDER_METHODS

    if isinstance(provider, type):
        raise InvalidBackendProviderError(
            f"Expected an API-v1 provider object, not a class; {source}"
        )
    missing = [
        name for name in _REQUIRED_PROVIDER_METHODS if not callable(getattr(provider, name, None))
    ]
    if missing:
        raise InvalidBackendProviderError(f"Missing API-v1 provider methods {missing!r}; {source}")

    device_type = getattr(provider, "device_type", None)
    placement = _legacy_placement(
        device_type.strip().lower() if isinstance(device_type, str) else ""
    )
    if placement is None:
        raise InvalidBackendProviderError(
            "A vllm-gr backend provider without api_version uses the deprecated "
            "API version 1 and can only be adapted for its fixed tensor placements; "
            f"found device_type={device_type!r}; source={source!r}. "
            "Upgrade the provider to API version 2 and declare backend_id, "
            "api_version=2 and matches(identity)."
        )

    logger.warning(
        "vllm-gr backend provider %r (%r) uses the deprecated API version 1 "
        "adapter; backend_id=%r. Declare api_version=%d, backend_id and "
        "matches(identity) before the next release.",
        source,
        provider,
        placement.backend_id,
        PROVIDER_API_VERSION,
    )
    return LegacyProviderAdapter(legacy=provider, source=source, backend_id=placement.backend_id)
