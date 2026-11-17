# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Process-local registry for vllm-gr backend providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import (
    AmbiguousBackendError,
    BackendAlreadyActiveError,
    BackendNotActiveError,
    DuplicateBackendRegistrationError,
    IncompatibleBackendError,
    InvalidBackendProviderError,
    InvalidBackendVersionError,
    UnsupportedBackendError,
)
from .identity import PlatformIdentity
from .protocol import PROVIDER_API_VERSION, GRBackendProvider

_REQUIRED_PROVIDER_METHODS = (
    "check_environment",
    "apply_runtime_patches",
    "get_model_runner_hooks",
    "get_attention_execution",
    "create_constrained_topk_execution",
)
_REQUIRED_PROVIDER_METHODS_REQUIRED = ("matches",)


@dataclass(frozen=True)
class RegisteredBackend:
    """One validated provider together with the origin that registered it."""

    provider: GRBackendProvider
    backend_id: str
    source: str
    origin: str


_BACKENDS: dict[str, RegisteredBackend] = {}
_ACTIVE_BACKEND: RegisteredBackend | None = None


def _normalize_backend_id(backend_id: str) -> str:
    normalized = backend_id.strip().lower()
    if not normalized:
        raise ValueError("backend ID must not be empty")
    return normalized


def _active_backend_id() -> str | None:
    return None if _ACTIVE_BACKEND is None else _ACTIVE_BACKEND.backend_id


def _registry_context(*, requested: str | None = None) -> str:
    fields = []
    if requested is not None:
        fields.append(f"requested={requested!r}")
    fields.extend(
        (
            f"registered={tuple(sorted(_BACKENDS))!r}",
            f"active={_active_backend_id()!r}",
        )
    )
    return "; ".join(fields)


def _origin(backend_id: str, source: str) -> str:
    """Return a diagnostic origin naming both the source and the distribution."""
    distribution = ""
    if source.startswith("entrypoint:"):
        distribution = _entrypoint_distribution(source.removeprefix("entrypoint:"))
    if distribution:
        return f"{source} [distribution={distribution}; backend_id={backend_id}]"
    return f"{source} [backend_id={backend_id}]"


def _entrypoint_distribution(entry_point_name: str) -> str:
    """Read the installed distribution of an entry point, if it can be found."""
    try:
        from importlib.metadata import entry_points

        for entry_point in entry_points(group="vllm_gr.platform_backends"):
            if entry_point.name != entry_point_name:
                continue
            distribution = getattr(entry_point, "dist", None)
            if distribution is None:
                return ""
            name = distribution.metadata["Name"]
            return f"{name}=={distribution.version}"
    except Exception:  # pragma: no cover - diagnostics must never mask the error
        return ""
    return ""


def _validate_backend_provider(provider: Any, *, source: str) -> str:
    if isinstance(provider, type):
        raise InvalidBackendProviderError(
            "Backend entry must be a provider object, not a class; "
            f"source={source!r}; {_registry_context()}"
        )

    backend_id = getattr(provider, "backend_id", None)
    if not isinstance(backend_id, str):
        raise InvalidBackendProviderError(
            "Backend provider backend_id must be a non-empty string; "
            f"found={backend_id!r}; source={source!r}; {_registry_context()}"
        )

    try:
        normalized = _normalize_backend_id(backend_id)
    except ValueError as exc:
        raise InvalidBackendProviderError(
            f"{exc}; source={source!r}; {_registry_context(requested=backend_id)}"
        ) from exc

    api_version = getattr(provider, "api_version", None)
    if not isinstance(api_version, int) or isinstance(api_version, bool):
        raise InvalidBackendVersionError(
            "Backend provider api_version must be an integer; "
            f"found={api_version!r}; {_origin(normalized, source)}; "
            f"{_registry_context(requested=normalized)}"
        )
    if api_version != PROVIDER_API_VERSION:
        raise InvalidBackendVersionError(
            "Unsupported vllm-gr backend provider API version; "
            f"found={api_version!r}; supported={PROVIDER_API_VERSION!r}; "
            f"{_origin(normalized, source)}; "
            f"{_registry_context(requested=normalized)}"
        )

    missing = tuple(
        name
        for name in (*_REQUIRED_PROVIDER_METHODS_REQUIRED, *_REQUIRED_PROVIDER_METHODS)
        if not callable(getattr(provider, name, None))
    )
    if missing:
        raise InvalidBackendProviderError(
            "Backend provider is missing required callable attributes; "
            f"missing={missing!r}; {_origin(normalized, source)}; "
            f"{_registry_context(requested=normalized)}"
        )
    from .services import validate_planning_api

    try:
        validate_planning_api(provider)
    except ValueError as exc:
        raise InvalidBackendProviderError(f"{exc}; {_origin(normalized, source)}") from exc
    return normalized


def _validate_backend_registration(
    provider: Any,
    *,
    source: str,
    pending_backends: dict[str, RegisteredBackend] | None = None,
) -> RegisteredBackend:
    """Validate one registration without mutating process-global state."""
    normalized = _validate_backend_provider(provider, source=source)
    candidate = RegisteredBackend(
        provider=provider,
        backend_id=normalized,
        source=source,
        origin=_origin(normalized, source),
    )

    existing = _BACKENDS.get(normalized)
    if existing is None and pending_backends is not None:
        existing = pending_backends.get(normalized)

    if existing is not None and existing.provider is not provider:
        raise DuplicateBackendRegistrationError(
            "A different vllm-gr backend is already registered for this ID; "
            f"existing_origin={existing.origin!r}; "
            f"attempted_origin={candidate.origin!r}; "
            f"{_registry_context(requested=normalized)}"
        )
    return candidate


def register_backend(
    provider: GRBackendProvider,
    *,
    source: str = "manual",
) -> None:
    """Register a provider without loading any of its hardware modules."""
    candidate = _validate_backend_registration(provider, source=source)
    existing = _BACKENDS.get(candidate.backend_id)
    if existing is not None and existing.provider is provider:
        return
    _BACKENDS[candidate.backend_id] = candidate


def resolve_backend(backend_id: str) -> GRBackendProvider:
    """Resolve an explicitly registered GR backend ID."""
    normalized = _normalize_backend_id(backend_id)
    try:
        return _BACKENDS[normalized].provider
    except KeyError as exc:
        raise UnsupportedBackendError(
            "No vllm-gr backend registered for "
            f"{normalized!r}; available={tuple(sorted(_BACKENDS))!r}; "
            f"{_registry_context(requested=normalized)}"
        ) from exc


def resolve_backend_id(backend_id: str) -> RegisteredBackend:
    """Resolve the registration record of an explicit GR backend ID."""
    normalized = _normalize_backend_id(backend_id)
    try:
        return _BACKENDS[normalized]
    except KeyError as exc:
        raise UnsupportedBackendError(
            "No vllm-gr backend registered for "
            f"{normalized!r}; available={tuple(sorted(_BACKENDS))!r}; "
            f"{_registry_context(requested=normalized)}"
        ) from exc


def select_backend(
    identity: PlatformIdentity,
    *,
    requested: str | None = None,
) -> RegisteredBackend:
    """Select exactly one provider for an already-selected native platform.

    An explicit request may choose between compatible matches, but it cannot
    suppress a duplicate-ID error or make an incompatible platform match. With
    no request, zero matches is unsupported and several matches is ambiguous;
    discovery order never breaks the tie.
    """
    if requested is not None:
        normalized = _normalize_backend_id(requested)
        registration = resolve_backend_id(normalized)
        if not registration.provider.matches(identity):
            raise IncompatibleBackendError(
                "Explicitly requested vllm-gr backend is incompatible with the "
                f"selected native platform; {identity.describe()}; "
                f"requested_origin={registration.origin!r}; "
                f"{_registry_context(requested=normalized)}"
            )
        return registration

    matches = [
        registration
        for registration in _BACKENDS.values()
        if registration.provider.matches(identity)
    ]
    if not matches:
        raise UnsupportedBackendError(
            "No vllm-gr backend matches the selected native platform; "
            f"{identity.describe()}; available={tuple(sorted(_BACKENDS))!r}; "
            "set VLLM_GR_BACKEND to choose one explicitly; "
            f"{_registry_context()}"
        )
    if len(matches) > 1:
        raise AmbiguousBackendError(
            "Several vllm-gr backends match the selected native platform; "
            f"{identity.describe()}; matches={[m.origin for m in matches]!r}; "
            "set VLLM_GR_BACKEND to choose one explicitly; "
            f"{_registry_context()}"
        )
    return matches[0]


def activate_backend(
    backend_id: str,
    *,
    identity: PlatformIdentity | None = None,
) -> GRBackendProvider:
    """Activate one backend for the lifetime of the current process.

    When ``identity`` is provided the request must also match that native
    platform; an explicit ID never switches the platform vLLM selected.
    """
    global _ACTIVE_BACKEND

    normalized = _normalize_backend_id(backend_id)
    if identity is not None:
        registration = select_backend(identity, requested=normalized)
    else:
        registration = resolve_backend_id(normalized)

    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = registration
        return registration.provider
    if _ACTIVE_BACKEND.provider is registration.provider:
        return registration.provider

    raise BackendAlreadyActiveError(
        "A different vllm-gr backend is already active; "
        f"active_origin={_ACTIVE_BACKEND.origin!r}; "
        f"requested_origin={registration.origin!r}; "
        f"{_registry_context(requested=normalized)}"
    )


def activate_selected_backend(
    identity: PlatformIdentity,
    *,
    requested: str | None = None,
) -> GRBackendProvider:
    """Select and activate the provider for a native platform identity."""
    global _ACTIVE_BACKEND

    registration = select_backend(identity, requested=requested)
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = registration
        return registration.provider
    if _ACTIVE_BACKEND.provider is registration.provider:
        return registration.provider

    raise BackendAlreadyActiveError(
        "A different vllm-gr backend is already active; "
        f"active_origin={_ACTIVE_BACKEND.origin!r}; "
        f"requested_origin={registration.origin!r}; "
        f"platform={identity.describe()}; {_registry_context()}"
    )


def get_active_backend() -> GRBackendProvider:
    """Return the active provider or fail when activation has not occurred."""
    if _ACTIVE_BACKEND is None:
        raise BackendNotActiveError(f"No vllm-gr backend is active; {_registry_context()}")
    return _ACTIVE_BACKEND.provider


def _reset_backend_registry_for_tests() -> None:
    """Clear process-local backend state for isolated tests."""
    global _ACTIVE_BACKEND

    _BACKENDS.clear()
    _ACTIVE_BACKEND = None
