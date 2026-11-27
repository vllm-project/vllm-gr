# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Resolve the optional canonical Constraint Table factory on the active provider."""

from typing import cast

from .errors import BackendNotActiveError, UnsupportedBackendError
from .interface import CONSTRAINT_BACKEND_API_VERSION, CanonicalConstraintProvider
from .registry import get_active_backend


def canonical_constraint_provider(device_type: str) -> CanonicalConstraintProvider:
    """Require an active provider for the Worker device; do not probe or fall back."""
    try:
        provider = get_active_backend()
    except BackendNotActiveError as exc:
        raise UnsupportedBackendError(
            f"No active GR provider for canonical constraint placement {device_type!r}; "
            "install and activate its vllm-gr platform plugin first"
        ) from exc
    if provider.device_type != device_type:
        raise UnsupportedBackendError(
            f"Active GR backend {provider.backend_id!r} serves {provider.device_type!r}, "
            f"not canonical constraint placement {device_type!r}"
        )
    version = getattr(provider, "constraint_backend_api_version", None)
    if type(version) is not int or version != CONSTRAINT_BACKEND_API_VERSION:
        raise UnsupportedBackendError(
            f"GR backend {provider.backend_id!r} declares constraint_backend_api_version="
            f"{version!r}; canonical constraint placement {device_type!r} requires "
            f"version {CONSTRAINT_BACKEND_API_VERSION}"
        )
    if not callable(getattr(provider, "create_constraint_backend", None)):
        raise UnsupportedBackendError(
            f"GR backend {provider.backend_id!r} is missing create_constraint_backend()"
        )
    return cast(CanonicalConstraintProvider, provider)
