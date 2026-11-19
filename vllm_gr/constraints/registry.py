# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Registration of Constraint Table implementations by identity.

A canonical Constraint Table is executed by exactly one implementation. This
module answers "which one", so no call site carries a closed list of them.

The registry is data-driven on purpose: an implementation registers an object
that knows how to admit a configuration and how to build a Backend, and the
loader looks it up by ID. Adding an implementation therefore means registering
it, not editing an enum or a branch in shared code.

``legacy_enum`` keeps historical payloads working. A configuration written
before ``implementation_id`` existed names only ``backend``; that enum maps to
the implementation that historically served it, and nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_gr.config.models import ConstraintTableBackendEnum

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.backend import ConstraintBackend
    from vllm_gr.constraints.table import ConstraintTable


class ConstraintImplementationError(RuntimeError):
    """A configuration names an implementation that cannot serve it."""


@dataclass(frozen=True, slots=True)
class ConstraintTableRequest:
    """The resolved inputs one implementation needs to build a Backend.

    Everything here is already validated by the caller, so an implementation
    never re-derives placement, dtype or shape policy from a device name.
    """

    table: ConstraintTable
    device: torch.device
    dtype: torch.dtype
    max_top_k: int
    max_rows: int
    enable_stage_vocab: bool = False


@dataclass(frozen=True, slots=True)
class ConstraintImplementation:
    """One registered way of executing a canonical Constraint Table.

    ``admit`` runs before the Artifact is read, so a configuration this
    implementation cannot serve fails without paying artifact I/O. It returns
    whatever the implementation resolved for itself — a fused operator handle,
    for instance — and ``build`` receives that value unchanged. Keeping the two
    steps separate preserves the ordering callers already rely on: an
    implementation's placement, capacity and operator checks all happen before
    the Artifact is opened.

    ``build`` owns the device upload, readiness and readiness-failure cleanup.
    """

    implementation_id: str
    legacy_enum: ConstraintTableBackendEnum | None
    admit: Callable[[Any, torch.device, torch.dtype], Any]
    build: Callable[[ConstraintTableRequest, Any], ConstraintBackend]


_REGISTRY: dict[str, ConstraintImplementation] = {}
_LEGACY: dict[ConstraintTableBackendEnum, str] = {}


def normalize_implementation_id(value: str) -> str:
    """Return the canonical spelling of an implementation ID."""
    normalized = value.strip().lower()
    if not normalized:
        raise ConstraintImplementationError("constraint implementation_id must not be empty")
    return normalized


def register_implementation(implementation: ConstraintImplementation) -> None:
    """Register one implementation, rejecting a duplicate or ambiguous identity."""
    implementation_id = normalize_implementation_id(implementation.implementation_id)
    existing = _REGISTRY.get(implementation_id)
    if existing is not None and existing is not implementation:
        raise ConstraintImplementationError(
            f"constraint implementation_id {implementation_id!r} is already registered"
        )
    legacy = implementation.legacy_enum
    if legacy is not None:
        owner = _LEGACY.get(legacy)
        if owner is not None and owner != implementation_id:
            raise ConstraintImplementationError(
                f"constraint backend enum {legacy.value!r} already maps to {owner!r}; "
                f"cannot also map to {implementation_id!r}"
            )
        _LEGACY[legacy] = implementation_id
    _REGISTRY[implementation_id] = implementation


def registered_implementation_ids() -> tuple[str, ...]:
    """Return every registered ID, sorted for a stable diagnostic."""
    return tuple(sorted(_REGISTRY))


def implementation_for_legacy_enum(
    backend: ConstraintTableBackendEnum,
) -> ConstraintImplementation:
    """Return the implementation that historically served ``backend``."""
    implementation_id = _LEGACY.get(backend)
    if implementation_id is None:
        raise ConstraintImplementationError(
            f"no registered constraint implementation serves legacy backend {backend.value!r}; "
            f"registered implementations: {', '.join(registered_implementation_ids()) or 'none'}"
        )
    return _REGISTRY[implementation_id]


def resolve_constraint_implementation(config: Any) -> ConstraintImplementation:
    """Return the implementation a configuration asks for.

    An explicit ``implementation_id`` wins. Otherwise the legacy ``backend``
    enum selects the historical implementation, so a payload written before
    ``implementation_id`` existed keeps its established behavior.
    """
    requested = getattr(config, "implementation_id", None)
    if requested is not None:
        implementation_id = normalize_implementation_id(requested)
        implementation = _REGISTRY.get(implementation_id)
        if implementation is None:
            raise ConstraintImplementationError(
                f"unsupported constraint implementation_id {implementation_id!r}; "
                f"registered implementations: "
                f"{', '.join(registered_implementation_ids()) or 'none'}"
            )
        return implementation
    return implementation_for_legacy_enum(config.backend)


__all__ = [
    "ConstraintImplementation",
    "ConstraintImplementationError",
    "ConstraintTableRequest",
    "implementation_for_legacy_enum",
    "normalize_implementation_id",
    "register_implementation",
    "registered_implementation_ids",
    "resolve_constraint_implementation",
]
