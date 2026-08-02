# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External backend discovery and current-platform activation."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from .errors import InvalidBackendProviderError
from .protocol import GRBackendProvider
from .registry import (
    _registry_context,
    _validate_backend_registration,
    activate_backend,
    register_backend,
)

ENTRYPOINT_GROUP = "vllm_gr.platform_backends"

_PLUGINS_LOADED = False


def load_backend_plugins() -> None:
    """Load provider objects exposed through the backend entry-point group."""
    global _PLUGINS_LOADED

    if _PLUGINS_LOADED:
        return

    pending: list[tuple[GRBackendProvider, str, str]] = []
    pending_backends: dict[str, GRBackendProvider] = {}
    pending_sources: dict[str, str] = {}

    for entry_point in entry_points(group=ENTRYPOINT_GROUP):
        provider: Any = entry_point.load()
        source = f"entrypoint:{entry_point.name}"
        normalized = _validate_backend_registration(
            provider,
            source=source,
            pending_backends=pending_backends,
            pending_sources=pending_sources,
        )
        if entry_point.name != getattr(provider, "device_type", None):
            raise InvalidBackendProviderError(
                "Backend entry point name must exactly match provider.device_type; "
                f"entrypoint_name={entry_point.name!r}; "
                f"provider_device_type={getattr(provider, 'device_type', None)!r}; "
                f"source={source!r}; {_registry_context(requested=normalized)}"
            )
        pending.append((provider, source, normalized))
        pending_backends[normalized] = provider
        pending_sources[normalized] = source

    for provider, source, _ in pending:
        register_backend(provider, source=source)
    _PLUGINS_LOADED = True


def activate_current_backend() -> GRBackendProvider:
    """Register providers and activate vLLM's current canonical device type."""
    from vllm.platforms import current_platform

    from .builtins import register_builtin_backends

    register_builtin_backends()
    load_backend_plugins()
    return activate_backend(str(current_platform.device_type))
