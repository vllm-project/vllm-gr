# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local registry for vllm-gr backend providers."""

from __future__ import annotations

from typing import Any

from .errors import (
    BackendAlreadyActiveError,
    BackendNotActiveError,
    DuplicateBackendRegistrationError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from .protocol import GRBackendProvider

_REQUIRED_PROVIDER_METHODS = (
    "check_environment",
    "apply_runtime_patches",
    "get_model_runner_hooks",
    "get_attention_execution",
    "create_constrained_topk_execution",
)

_BACKENDS: dict[str, GRBackendProvider] = {}
_ACTIVE_BACKEND: GRBackendProvider | None = None
_BACKEND_SOURCES: dict[str, str] = {}


def _normalize_device_type(device_type: str) -> str:
    normalized = device_type.strip().lower()
    if not normalized:
        raise ValueError("backend device_type must not be empty")
    return normalized


def _active_device_type() -> str | None:
    if _ACTIVE_BACKEND is None:
        return None
    for device_type, provider in _BACKENDS.items():
        if provider is _ACTIVE_BACKEND:
            return device_type
    value = getattr(_ACTIVE_BACKEND, "device_type", None)
    return value if isinstance(value, str) else "<invalid>"


def _registry_context(*, requested: str | None = None) -> str:
    fields = []
    if requested is not None:
        fields.append(f"requested={requested!r}")
    fields.extend(
        (
            f"registered={tuple(sorted(_BACKENDS))!r}",
            f"active={_active_device_type()!r}",
        )
    )
    return "; ".join(fields)


def _validate_backend_provider(provider: Any, *, source: str) -> str:
    if isinstance(provider, type):
        raise InvalidBackendProviderError(
            "Backend entry must be a provider object, not a class; "
            f"source={source!r}; {_registry_context()}"
        )

    device_type = getattr(provider, "device_type", None)
    if not isinstance(device_type, str):
        raise InvalidBackendProviderError(
            "Backend provider device_type must be a non-empty string; "
            f"found={device_type!r}; source={source!r}; {_registry_context()}"
        )

    try:
        normalized = _normalize_device_type(device_type)
    except ValueError as exc:
        raise InvalidBackendProviderError(
            f"{exc}; source={source!r}; {_registry_context(requested=device_type)}"
        ) from exc

    missing = tuple(
        method_name
        for method_name in _REQUIRED_PROVIDER_METHODS
        if not callable(getattr(provider, method_name, None))
    )
    if missing:
        raise InvalidBackendProviderError(
            "Backend provider is missing required callable methods; "
            f"missing={missing!r}; source={source!r}; "
            f"{_registry_context(requested=normalized)}"
        )
    return normalized


def _validate_backend_registration(
    provider: Any,
    *,
    source: str,
    pending_backends: dict[str, GRBackendProvider] | None = None,
    pending_sources: dict[str, str] | None = None,
) -> str:
    """Validate one registration without mutating process-global state."""
    normalized = _validate_backend_provider(provider, source=source)

    existing = _BACKENDS.get(normalized)
    existing_source = _BACKEND_SOURCES.get(normalized)
    if existing is None and pending_backends is not None:
        existing = pending_backends.get(normalized)
        if pending_sources is not None:
            existing_source = pending_sources.get(normalized)

    if existing is not None and existing is not provider:
        raise DuplicateBackendRegistrationError(
            "A different vllm-gr backend is already registered; "
            f"existing_source={existing_source!r}; attempted_source={source!r}; "
            f"{_registry_context(requested=normalized)}"
        )
    return normalized


def register_backend(
    provider: GRBackendProvider,
    *,
    source: str = "manual",
) -> None:
    """Register a provider without loading any of its hardware modules."""
    normalized = _validate_backend_registration(provider, source=source)
    existing = _BACKENDS.get(normalized)
    if existing is provider:
        return
    _BACKENDS[normalized] = provider
    _BACKEND_SOURCES[normalized] = source


def resolve_backend(device_type: str) -> GRBackendProvider:
    """Resolve an explicitly registered canonical vLLM device type."""
    normalized = _normalize_device_type(device_type)
    try:
        return _BACKENDS[normalized]
    except KeyError as exc:
        raise UnsupportedBackendError(
            "No vllm-gr backend registered for "
            f"{normalized!r}; available={tuple(sorted(_BACKENDS))!r}; "
            f"{_registry_context(requested=normalized)}"
        ) from exc


def activate_backend(device_type: str) -> GRBackendProvider:
    """Activate one backend for the lifetime of the current process."""
    global _ACTIVE_BACKEND

    normalized = _normalize_device_type(device_type)
    provider = resolve_backend(normalized)
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = provider
        return provider
    if _ACTIVE_BACKEND is provider:
        return provider

    active_device_type = _active_device_type()
    raise BackendAlreadyActiveError(
        "A different vllm-gr backend is already active; "
        f"active_source={_BACKEND_SOURCES.get(active_device_type or '')!r}; "
        f"requested_source={_BACKEND_SOURCES.get(normalized)!r}; "
        f"{_registry_context(requested=normalized)}"
    )


def get_active_backend() -> GRBackendProvider:
    """Return the active provider or fail when activation has not occurred."""
    if _ACTIVE_BACKEND is None:
        raise BackendNotActiveError(f"No vllm-gr backend is active; {_registry_context()}")
    return _ACTIVE_BACKEND


def _reset_backend_registry_for_tests() -> None:
    """Clear process-local backend state for isolated tests."""
    global _ACTIVE_BACKEND

    _BACKENDS.clear()
    _BACKEND_SOURCES.clear()
    _ACTIVE_BACKEND = None
