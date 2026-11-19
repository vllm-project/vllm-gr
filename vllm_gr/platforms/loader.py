# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""External backend discovery and current-platform activation."""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Iterator
from functools import cache
from importlib.metadata import (
    EntryPoint,
    PackageNotFoundError,
    distribution,
    entry_points,
)
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name

from .errors import InvalidBackendProviderError
from .identity import current_platform_identity
from .protocol import GRBackendProvider
from .registry import (
    RegisteredBackend,
    _registry_context,
    _validate_backend_registration,
    activate_backend,
    get_active_backend,
    register_backend,
    select_backend,
)

logger = logging.getLogger("vllm_gr.platforms.loader")

ENTRYPOINT_GROUP = "vllm_gr.platform_backends"

#: Comma-separated backend IDs whose built-in provider must not be registered.
DISABLE_BUILTIN_ENV = "VLLM_GR_DISABLE_BUILTIN_BACKENDS"
#: Explicit backend ID. An explicit choice never switches the native platform.
REQUESTED_BACKEND_ENV = "VLLM_GR_BACKEND"

_PLUGINS_LOADED = False


@cache
def core_entry_points(group: str) -> tuple[EntryPoint, ...]:
    """Read this checkout's packaging declarations or its installed wheel metadata."""
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if project_file.is_file():
        with project_file.open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
        if project.get("name") == "vllm-gr":
            return tuple(
                EntryPoint(name=name, value=value, group=group)
                for name, value in project.get("entry-points", {}).get(group, {}).items()
            )
    try:
        return tuple(distribution("vllm-gr").entry_points.select(group=group))
    except PackageNotFoundError:
        return ()


def external_provider_entry_points() -> Iterator[EntryPoint]:
    """Discover provider distributions separately from GR's own metadata."""
    for entry in entry_points(group=ENTRYPOINT_GROUP):
        dist = getattr(entry, "dist", None)
        if dist is None or canonicalize_name(dist.metadata["Name"]) != "vllm-gr":
            yield entry


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

    disabled = disabled_builtin_backend_ids()
    entries = []
    for entry_point in core_entry_points(ENTRYPOINT_GROUP):
        if entry_point.name in disabled:
            logger.warning("vllm-gr built-in backend %r is disabled", entry_point.name)
            continue
        entries.append((entry_point, f"builtin:{entry_point.name}"))
    for entry_point in external_provider_entry_points():
        source = f"entrypoint:{entry_point.name}"
        dist = getattr(entry_point, "dist", None)
        if dist is not None:
            name = dist.metadata["Name"]
            source += f" [distribution={name}=={dist.version}]"
        entries.append((entry_point, source))

    for entry_point, source in entries:
        provider: Any = entry_point.load()
        adapted = adapt_legacy_provider(provider, source=source)
        candidate_provider = provider if adapted is None else adapted
        declared_name = (
            getattr(provider, "backend_id", None) if adapted is None else adapted.device_type
        )
        if entry_point.name != declared_name:
            raise InvalidBackendProviderError(
                "Backend entry point name must exactly match provider.backend_id "
                "(provider.device_type for API v1); "
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
    """
    from .compat import enforce_declared_ranges

    load_backend_plugins()

    active = _already_active()
    if active is not None:
        enforce_declared_ranges(active)
        return active

    registration = select_backend(
        current_platform_identity(),
        requested=requested_backend_id(),
    )
    enforce_declared_ranges(registration.provider)
    return activate_backend(registration.backend_id)


def _already_active() -> GRBackendProvider | None:
    """Return the provider activated earlier in this process, if any."""
    from .errors import BackendNotActiveError

    try:
        return get_active_backend()
    except BackendNotActiveError:
        return None
