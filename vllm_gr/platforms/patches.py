# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ownership and installation checks for the selected process's early patches."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

from .interface import GRBackendProvider
from .registry import resolve_backend_id

# Semantic ownership, independent of which process orchestrates an installer.
# Built-in providers declare their hardware-owned IDs in ``patch_ids``.
# These independent architectural assertions are checked against the authoritative
# docs/patch_inventory.yaml; they do not drive installation. See the ownership
# rationale in docs/runtime_patch_inventory.md.
CORE_PATCH_IDS = (
    "GR-PATCH-000",
    "GR-PATCH-001",
    "GR-PATCH-003",
    "GR-PATCH-004",
    "GR-PATCH-005",
    "GR-PATCH-007",
    "GR-PATCH-008",
    "GR-PATCH-009",
    "GR-PATCH-010",
    "GR-PATCH-013",
    "GR-PATCH-015",
    "GR-PATCH-020",
)


@dataclass(frozen=True)
class SelectedPatchClaims:
    provider: GRBackendProvider
    origin: str
    checks: tuple[tuple[str, Callable[[], bool | None]], ...]
    deferred: frozenset[str]

    @classmethod
    def from_provider(
        cls, provider: GRBackendProvider, core_patch_ids: Collection[str]
    ) -> SelectedPatchClaims:
        """Reject missing probes and conflicting owners before the first mutation."""
        registration = resolve_backend_id(provider.backend_id)
        origin = registration.origin
        if registration.provider is not provider:
            raise RuntimeError(f"Patch provider differs from registered owner {origin}")
        declared = getattr(provider, "patch_ids", ())
        checks = getattr(provider, "runtime_patch_checks", {})
        if not isinstance(declared, (tuple, list)) or not isinstance(checks, Mapping):
            raise ValueError(f"Invalid patch_ids or runtime_patch_checks for {origin}")
        claimed = set(core_patch_ids)
        if len(claimed) != len(core_patch_ids):
            raise ValueError(f"Duplicate active core patch IDs for {origin}")
        for patch_id in declared:
            if not isinstance(patch_id, str) or not patch_id.strip():
                raise ValueError(f"Invalid patch ID {patch_id!r} for {origin}")
            if patch_id in claimed:
                raise ValueError(f"Duplicate active patch ID {patch_id!r} for {origin}")
            claimed.add(patch_id)
        # The shipped inventory includes native hooks whose activation and
        # validators run in EngineCore or Worker. External early hooks must
        # supply a process-local probe for every ID, including replacement
        # providers registered under a disabled built-in's backend ID.
        deferred = (
            getattr(provider, "deferred_patch_ids", ())
            if registration.source == f"builtin:{registration.backend_id}"
            else ()
        )
        missing = set(declared).difference(checks, deferred)
        if missing:
            raise ValueError(f"Missing runtime patch checks {sorted(missing)!r} for {origin}")
        for patch_id, check in checks.items():
            if patch_id not in declared or not callable(check):
                raise ValueError(f"Invalid runtime patch check {patch_id!r} for {origin}")
        return cls(provider, origin, tuple(checks.items()), frozenset(deferred))

    def install(self, *, force: bool = False) -> int:
        try:
            count: int = self.provider.apply_runtime_patches(force=force)
        except Exception as exc:
            raise RuntimeError(
                f"Runtime patches for backend {self.provider.backend_id!r} ({self.origin}) "
                f"failed during apply_runtime_patches for {getattr(self.provider, 'patch_ids', ())!r}; "
                "initialization may be partial, restart the process"
            ) from exc
        for patch_id, check in self.checks:
            try:
                verified = check()
            except Exception as exc:
                raise RuntimeError(f"Patch {patch_id!r} check raised for {self.origin}") from exc
            if verified is None and patch_id in self.deferred:
                logging.getLogger(__name__).debug("Deferred patch %s for %s", patch_id, self.origin)
                continue
            if verified is not True:
                raise RuntimeError(f"Patch {patch_id!r} failed verification for {self.origin}")
        return count
