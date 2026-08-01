# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demand-loaded backend discovery and current-platform activation."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from .errors import (
    DuplicateBackendRegistrationError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from .protocol import GRBackendProvider
from .registry import (
    _normalize_device_type,
    _registry_context,
    _validate_backend_registration,
    activate_backend,
    register_backend,
)

ENTRYPOINT_GROUP = "vllm_gr.platform_backends"

_LOADED_PROVIDERS: dict[str, GRBackendProvider] = {}


def _entry_point_source(entry_point: Any) -> str:
    distribution = getattr(entry_point, "dist", None)
    distribution_name = getattr(distribution, "name", None)
    owner = distribution_name or "unknown-distribution"
    value = getattr(entry_point, "value", "<unknown>")
    return f"entrypoint:{owner}:{entry_point.name}={value}"


def load_backend_plugin(device_type: str) -> GRBackendProvider:
    """Load only the provider whose entry-point name matches ``device_type``."""
    normalized = _normalize_device_type(device_type)
    cached = _LOADED_PROVIDERS.get(normalized)
    if cached is not None:
        return cached

    matching = tuple(
        entry_point
        for entry_point in entry_points(group=ENTRYPOINT_GROUP)
        if entry_point.name == normalized
    )
    if not matching:
        raise UnsupportedBackendError(
            "No vllm-gr backend entry point matches the current platform; "
            f"entrypoint_group={ENTRYPOINT_GROUP!r}; "
            f"{_registry_context(requested=normalized)}"
        )
    if len(matching) > 1:
        sources = tuple(_entry_point_source(entry_point) for entry_point in matching)
        raise DuplicateBackendRegistrationError(
            "Multiple vllm-gr backend entry points match the current platform; "
            f"sources={sources!r}; {_registry_context(requested=normalized)}"
        )

    entry_point = matching[0]
    source = _entry_point_source(entry_point)
    try:
        provider: Any = entry_point.load()
    except Exception as exc:
        raise InvalidBackendProviderError(
            "Failed to load the selected vllm-gr backend provider; "
            f"source={source!r}; {_registry_context(requested=normalized)}"
        ) from exc

    provider_device_type = getattr(provider, "device_type", None)
    if entry_point.name != provider_device_type:
        raise InvalidBackendProviderError(
            "Backend entry point name must exactly match provider.device_type; "
            f"entrypoint_name={entry_point.name!r}; "
            f"provider_device_type={provider_device_type!r}; "
            f"source={source!r}; {_registry_context(requested=normalized)}"
        )
    _validate_backend_registration(provider, source=source)
    register_backend(provider, source=source)
    _LOADED_PROVIDERS[normalized] = provider
    return provider


def activate_current_backend() -> GRBackendProvider:
    """Demand-load and activate vLLM's current canonical device type."""
    from vllm.platforms import current_platform

    device_type = str(current_platform.device_type)
    load_backend_plugin(device_type)
    return activate_backend(device_type)


def _reset_backend_loader_for_tests() -> None:
    """Clear the process-local loader cache for isolated tests."""
    _LOADED_PROVIDERS.clear()
