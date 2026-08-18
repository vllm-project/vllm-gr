# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-side injection of Constraint Tables resources.

The driver compiles a
:class:`~vllm_gr.constraints.compiler.ConstraintTableResource` and injects it
into every worker via ``EngineCore.collective_rpc``
(``vLLM 0.22.1 v1/engine/core.py``).  The worker method
``set_gr_constraint_resources`` is dynamically mounted on the worker class by
:func:`install_worker_resource_hook`, which runs during worker ``init_device``
(before any request can arrive).

The RPC payload is a base-type wire dict (see
:func:`~vllm_gr.constraints.compiler.to_wire`) because vLLM v1 decodes utility
payloads with an untyped msgpack decoder; the worker rebuilds a
:class:`~vllm_gr.constraints.compiler.ConstraintTableResource` via
:func:`~vllm_gr.constraints.compiler.from_wire` (which restores the read-only
arrays, validates the table contract, and verifies the digest).  The worker
stores it in a process-local :class:`ConstraintResourceManager` keyed by
digest: same digest reuses, a different digest is rejected.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm_gr.constraints.resource_manager import ConstraintResourceManager

logger = logging.getLogger(__name__)

# Marker used to make the hook installation idempotent across re-apply cycles.
_WORKER_INJECT_MARKER = "_gr_constraint_resource_hook_installed"

_WORKER_MANAGER: ConstraintResourceManager | None = None


def get_worker_manager() -> ConstraintResourceManager:
    """Return the process-local constraint resource manager for this worker."""
    global _WORKER_MANAGER
    if _WORKER_MANAGER is None:
        _WORKER_MANAGER = ConstraintResourceManager()
    return _WORKER_MANAGER


def set_gr_constraint_resources(
    self: Any,
    payload: dict,
    manager: ConstraintResourceManager | None = None,
) -> dict:
    """Worker RPC method: idempotently cache constraint tables by digest.

    Args:
        payload: the :func:`~vllm_gr.constraints.compiler.to_wire` wire dict
            broadcast by the driver via ``EngineCore.collective_rpc``.  It is a
            dict of base types so it survives the untyped msgpack decode.
        manager: optional manager override (tests).  Defaults to the
            process-local singleton.

    Returns:
        A small status dict: ``{"digest": ..., "cached": bool}``.  ``cached``
        is ``True`` the first time a digest is stored and ``False`` on a
        same-digest reuse.

    Raises:
        ValueError: if the payload is malformed / digest mismatch, or a
            different digest is already cached (no silent swap).

    Note:
        This setter only caches the CPU-side resource (rebuilt via
        :func:`~vllm_gr.constraints.compiler.from_wire`, which restores the
        read-only arrays and verifies the digest).  It does *not* move tables
        to a device; the platform provider loads them via
        ``ConstrainedTopKExecution.load_tables`` (e.g. lazily on the first
        ``run``), reading the resource from the manager by digest.
    """
    from vllm_gr.constraints.compiler import from_wire

    resource = from_wire(payload)
    target = manager if manager is not None else get_worker_manager()
    already = target.current_digest == resource.digest
    target.ensure(resource)
    # Fail at startup when the active runner cannot run the raw-logits hook
    # (e.g. a vllm-ascend build without ``_sample``): a Constraint Table
    # request must never silently sample unconstrained.  Inert when ``self``
    # has no model_runner yet (unit tests / mocked workers).
    _verify_runner_sample_hook(self)
    logger.info(
        "vllm-gr: worker constraint tables digest=%s cached=%s",
        resource.digest,
        not already,
    )
    return {"digest": resource.digest, "cached": not already}


def _verify_runner_sample_hook(worker: Any) -> None:
    """Raise when the worker's runner lacks the Constraint Table ``_sample`` hook.

    Only meaningful once ``init_device`` has built ``model_runner`` (the driver
    injects after that point); skipped otherwise so unit tests and mocked
    workers keep working.
    """
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER

    sample = getattr(type(runner), "_sample", None)
    if sample is None or not getattr(sample, _WORKER_SAMPLE_PATCH_MARKER, False):
        raise RuntimeError(
            "constraint tables require the vllm-gr raw-logits hook on the "
            f"active runner {type(runner).__name__}, but its _sample is not "
            "patched (this platform lacks the hook). Constraint Tables cannot "
            "start on this worker; refusing to degrade to unconstrained "
            "sampling."
        )


def install_worker_resource_hook(worker_cls: type) -> int:
    """Mount ``set_gr_constraint_resources`` on a worker class, once.

    Must run before any ``collective_rpc`` that injects tables; worker
    ``init_device`` is the natural point.
    """
    if getattr(worker_cls, _WORKER_INJECT_MARKER, False):
        return 0
    worker_cls.set_gr_constraint_resources = set_gr_constraint_resources  # type: ignore[attr-defined]
    setattr(worker_cls, _WORKER_INJECT_MARKER, True)
    return 1
