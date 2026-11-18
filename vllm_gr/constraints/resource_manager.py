# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Process-local Constraint Tables resource lifecycle.

The driver compiles a
:class:`~vllm_gr.constraints.compiler.ConstraintTableResource` once and injects
it into every worker.  This manager is the process-local cache that owns those
CPU-side resources keyed by digest:

- ``ensure`` is idempotent: injecting the same digest reuses the cached
  resource; injecting a different digest is rejected rather than silently
  replacing the current tables.
- ``get`` looks a resource up by digest for the platform adapter to load onto
  the device.

The manager deliberately stays CPU-side and does not own device tensors; moving
a resource to a device is the platform adapter's job
(:class:`~vllm_gr.constraints.execution.ConstrainedTopKExecution.load_tables`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm_gr.constraints.compiler import ConstraintTableResource


class ConstraintResourceManager:
    """Idempotent, digest-keyed cache of constraint table resources."""

    def __init__(self) -> None:
        self._resources: dict[str, "ConstraintTableResource"] = {}
        self._current_digest: str | None = None

    @property
    def current_digest(self) -> str | None:
        """Digest of the currently loaded resource, if any."""
        return self._current_digest

    @property
    def resources(self) -> dict[str, "ConstraintTableResource"]:
        """All cached resources, keyed by digest."""
        return dict(self._resources)

    @property
    def enabled(self) -> bool:
        """Whether constraint tables have been injected."""
        return self._current_digest is not None

    def ensure(self, resource: "ConstraintTableResource") -> None:
        """Idempotently register a resource by digest.

        Same digest: no-op (the resource is already cached).
        Different digest: rejected, because silently swapping constraint
        tables mid-service would change decoding semantics without notice.
        """
        if self._current_digest == resource.digest:
            return
        if self._current_digest is not None:
            raise ValueError(
                f"cannot replace loaded constraint tables digest "
                f"{self._current_digest} with {resource.digest}; "
                "constraint tables are immutable per service run"
            )
        self._resources[resource.digest] = resource
        self._current_digest = resource.digest

    def get(self, digest: str) -> Optional["ConstraintTableResource"]:
        """Return the resource for a digest, or ``None`` if not injected."""
        return self._resources.get(digest)

    def require(self, digest: str) -> "ConstraintTableResource":
        """Like :meth:`get` but raises when the digest is unknown."""
        resource = self._resources.get(digest)
        if resource is None:
            raise ValueError(
                f"constraint tables digest {digest} is not loaded; call ensure() first"
            )
        return resource

    def clear(self) -> None:
        """Drop all cached resources (used on shutdown / explicit reset)."""
        self._resources.clear()
        self._current_digest = None
