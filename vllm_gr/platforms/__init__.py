# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Stable public API for vllm-gr backend providers."""

from .errors import (
    AmbiguousBackendError,
    BackendAlreadyActiveError,
    BackendNotActiveError,
    BackendRegistryError,
    DuplicateBackendRegistrationError,
    IncompatibleBackendError,
    InvalidBackendProviderError,
    InvalidBackendVersionError,
    UnsupportedBackendError,
)
from .identity import PlatformIdentity, current_platform_identity, platform_identity
from .legacy import LegacyProviderAdapter, adapt_legacy_provider, declared_api_version
from .loader import activate_current_backend
from .plan import (
    PLAN_API_VERSION,
    BackendPlan,
    BackendRequest,
    PlanData,
    ServiceRequirements,
)
from .protocol import (
    PROVIDER_API_VERSION,
    ConfigProvider,
    GRBackendProvider,
    GRPlanningProvider,
)
from .registry import (
    activate_backend,
    get_active_backend,
    register_backend,
    resolve_backend,
    select_backend,
)

__all__ = [
    "PROVIDER_API_VERSION",
    "PLAN_API_VERSION",
    "BackendPlan",
    "BackendRequest",
    "PlanData",
    "ServiceRequirements",
    "GRPlanningProvider",
    "AmbiguousBackendError",
    "BackendAlreadyActiveError",
    "BackendNotActiveError",
    "BackendRegistryError",
    "DuplicateBackendRegistrationError",
    "ConfigProvider",
    "GRBackendProvider",
    "IncompatibleBackendError",
    "InvalidBackendProviderError",
    "InvalidBackendVersionError",
    "LegacyProviderAdapter",
    "PlatformIdentity",
    "UnsupportedBackendError",
    "activate_backend",
    "activate_current_backend",
    "adapt_legacy_provider",
    "current_platform_identity",
    "declared_api_version",
    "get_active_backend",
    "platform_identity",
    "register_backend",
    "resolve_backend",
    "select_backend",
]
