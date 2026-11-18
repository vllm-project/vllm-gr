# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""External backend discovery and current-platform activation."""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from typing import Any

from .errors import InvalidBackendProviderError
from .identity import current_platform_identity
from .protocol import GRBackendProvider
from .registry import (
    RegisteredBackend,
    _registry_context,
    _validate_backend_registration,
    activate_selected_backend,
    get_active_backend,
    register_backend,
)

logger = logging.getLogger("vllm_gr.platforms.loader")

ENTRYPOINT_GROUP = "vllm_gr.platform_backends"

#: Comma-separated backend IDs whose built-in provider must not be registered.
DISABLE_BUILTIN_ENV = "VLLM_GR_DISABLE_BUILTIN_BACKENDS"
#: Explicit backend ID. An explicit choice never switches the native platform.
REQUESTED_BACKEND_ENV = "VLLM_GR_BACKEND"

_PLUGINS_LOADED = False


def disabled_builtin_backend_ids() -> frozenset[str]:
    """Read the documented built-in disable list from the environment."""
    raw = os.environ.get(DISABLE_BUILTIN_ENV, "")
    return frozenset(item.strip().lower() for item in raw.split(",") if item.strip())


def requested_backend_id() -> str | None:
    """Read the explicit backend request, if the process made one."""
    raw = os.environ.get(REQUESTED_BACKEND_ENV)
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        raise InvalidBackendProviderError(
            f"{REQUESTED_BACKEND_ENV} is set but empty; unset it to select the "
            "backend from the native platform, or name a backend ID"
        )
    return value


def load_backend_plugins() -> None:
    """Load provider objects exposed through the backend entry-point group.

    An entry point either declares ``api_version`` explicitly or is adapted
    through the deprecated version-1 path. Duplicates are rejected with both
    origins unless an explicitly disabled built-in ID made room for it.
    """
    global _PLUGINS_LOADED

    if _PLUGINS_LOADED:
        return

    from .legacy import adapt_legacy_provider

    pending: list[tuple[GRBackendProvider, str]] = []
    pending_backends: dict[str, RegisteredBackend] = {}

    for entry_point in entry_points(group=ENTRYPOINT_GROUP):
        provider: Any = entry_point.load()
        source = f"entrypoint:{entry_point.name}"
        adapted = adapt_legacy_provider(provider, source=source)
        candidate_provider = provider if adapted is None else adapted
        declared_name = getattr(candidate_provider, "backend_id", None)
        if entry_point.name != declared_name:
            raise InvalidBackendProviderError(
                "Backend entry point name must exactly match provider.backend_id; "
                f"entrypoint_name={entry_point.name!r}; "
                f"provider_backend_id={declared_name!r}; "
                f"source={source!r}; {_registry_context()}"
            )
        registration = _validate_backend_registration(
            candidate_provider,
            source=source,
            pending_backends=pending_backends,
        )
        pending.append((candidate_provider, source))
        pending_backends[registration.backend_id] = registration

    for provider, source in pending:
        register_backend(provider, source=source)
    _PLUGINS_LOADED = True


def activate_current_backend() -> GRBackendProvider:
    """Register providers and activate the one matching native vLLM.

    Registration order is built-ins first, then external entry points, with the
    documented disable list applied before any registration. Selection then
    matches providers against the identity of the platform vLLM already chose;
    an explicit ``VLLM_GR_BACKEND`` may pick among compatible matches but cannot
    make an incompatible platform match.

    Once a backend is active, the answer cannot change in this process, so the
    call returns it without repeating discovery. Execution entry points call this
    per stage and per attention layer, and re-running registration there would
    add registry walks to the token hot path.
    """
    active = _already_active()
    if active is not None:
        return active

    from .builtins import register_builtin_backends
    from .compat import enforce_declared_ranges

    register_builtin_backends()
    load_backend_plugins()

    provider = activate_selected_backend(
        current_platform_identity(),
        requested=requested_backend_id(),
    )
    enforce_declared_ranges(provider)
    return provider


def _already_active() -> GRBackendProvider | None:
    """Return the provider activated earlier in this process, if any."""
    from .errors import BackendNotActiveError

    try:
        return get_active_backend()
    except BackendNotActiveError:
        return None


#: Bound services, keyed by factory, together with the provider that owns them.
_SERVICES: dict[str, tuple[object, object]] = {}


def active_backend_service(factory: str) -> object:
    """Return one bound service of the active backend, created once per process.

    RFC #706 requires execution to call services that are already bound: no
    entry-point discovery, reflection, or backend probing per token. Activation
    walks the registry and re-checks declared version ranges, so shared code that
    runs per attention layer or per decode step must not repeat it.

    The cached entry records the provider that produced the service and is
    rebuilt when that provider changes, so a test that selects a different
    backend never receives the previous backend's service. Holding the provider
    reference also keeps its identity stable for the comparison.
    """
    provider = activate_current_backend()

    entry = _SERVICES.get(factory)
    if entry is not None and entry[0] is provider:
        return entry[1]
    service = getattr(provider, factory)()
    _SERVICES[factory] = (provider, service)
    return service


def _reset_backend_services_for_tests() -> None:
    """Drop cached services so an isolated test selects its own backend."""
    _SERVICES.clear()

