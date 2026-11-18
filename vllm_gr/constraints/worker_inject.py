# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Worker-side injection of Constraint Tables resources.

The driver compiles a
:class:`~vllm_gr.constraints.compiler.ConstraintTableResource` and injects it
into every worker via ``EngineCore.collective_rpc``
(``vLLM 0.30.0 v1/engine/core.py``). The worker method
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


def load_constraint_tables_on_worker(worker: Any) -> bool:
    """Preload the injected constraint tables onto the runner device.

    Idempotent: a runner that already owns a loaded adapter (``_gr_constraint_execution``)
    is left untouched.  When the worker has injected tables (the driver
    broadcast them via ``EngineCore.collective_rpc``) and a runner is
    available, the platform adapter is created and ``load_tables`` moves the
    resource to the device — at weight-load time (``init_device``) or as soon
    as the injection arrives (``set_gr_constraint_resources``), so the runtime
    never pays a lazy build / H2D on the first constrained step.

    ``worker`` may be the worker object (with ``model_runner``) or the runner
    itself (which owns ``device`` / ``_gr_constraint_execution`` directly).

    Returns:
        ``True`` when tables were loaded (or already present), ``False`` when
        nothing was injected yet or the runner is not available (the caller
        keeps the current lazy path as a fallback).
    """
    runner = getattr(worker, "model_runner", None)
    if runner is None and getattr(worker, "device", None) is not None:
        # The caller passed the runner itself (constraint_hook path).
        runner = worker
    if runner is None:
        return False
    if getattr(runner, "_gr_constraint_execution", None) is not None:
        return True
    if getattr(runner, "device", None) is None:
        # No device yet (mocked runners in tests, or device not initialised):
        # keep the lazy runtime path — do not build an adapter against the
        # default device here.
        return False

    manager = get_worker_manager()
    digest = manager.current_digest
    if digest is None:
        return False

    from vllm_gr.constraints.startup import create_constraint_execution

    adapter = create_constraint_execution(device=getattr(runner, "device", None))
    adapter.load_tables(manager.require(digest))
    runner._gr_constraint_execution = adapter
    logger.info(
        "vllm-gr: worker preloaded constraint tables digest=%s onto %s",
        digest,
        getattr(runner, "device", None),
    )
    return True


def set_gr_constraint_resources(
    self: Any,
    payload: dict,
    manager: ConstraintResourceManager | None = None,
) -> dict:
    """Worker RPC method: idempotently cache and preload constraint tables.

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
        RuntimeError: if the active runner lacks the required constraint hooks.

    Note:
        This setter caches the CPU-side resource (rebuilt via
        :func:`~vllm_gr.constraints.compiler.from_wire`, which restores the
        read-only arrays and verifies the digest), validates the active runner's
        hooks, and preloads tables when its device is available.
    """
    from vllm_gr.constraints.compiler import from_wire

    resource = from_wire(payload)
    target = manager if manager is not None else get_worker_manager()
    already = target.current_digest == resource.digest
    target.ensure(resource)
    # Fail at startup when the active runner lacks a supported constraint hook
    # combination: a Constraint Table request must never silently sample
    # unconstrained.  Inert when ``self`` has no model_runner yet
    # (unit tests / mocked workers).
    _verify_runner_sample_hook(self)
    # Preload the tables onto the device now that the resource is registered
    # (the runner is built when injection happens after init_device).  This is
    # the Step-B half of moving load_tables to weight-load time: if the tables
    # arrived after init_device, this closes the gap with no lazy runtime load.
    load_constraint_tables_on_worker(self)
    logger.info(
        "vllm-gr: worker constraint tables digest=%s cached=%s",
        resource.digest,
        not already,
    )
    return {"digest": resource.digest, "cached": not already}


def _verify_runner_sample_hook(worker: Any) -> None:
    """Raise when the worker's runner lacks the Constraint Table sample hook.

    Accepts the retained V1 ``_sample`` hook or a V1/V2 persistent-stage
    execute_model / sample_tokens pair. CUDA ModelRunner V1 and V2 leave
    native sampling untouched: their beam_search_v1 stage runner owns
    constrained sampling directly.

    Only meaningful once ``init_device`` has built ``model_runner`` (the driver
    injects after that point); skipped otherwise so unit tests and mocked
    workers keep working.
    """
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return
    from vllm_gr.v1.worker.model_runner_contracts import (
        _WORKER_EXECUTE_PATCH_MARKER,
        _WORKER_SAMPLE_PATCH_MARKER,
        _WORKER_SAMPLE_TOKENS_PATCH_MARKER,
        _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER,
    )

    runner_cls = type(runner)
    v1_sample = getattr(runner_cls, "_sample", None)
    if v1_sample is not None and getattr(v1_sample, _WORKER_SAMPLE_PATCH_MARKER, False):
        return
    sample_tokens = getattr(runner_cls, "sample_tokens", None)
    if getattr(
        getattr(runner_cls, "execute_model", None),
        _WORKER_EXECUTE_PATCH_MARKER,
        False,
    ) and (
        getattr(sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False)
        or getattr(sample_tokens, _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False)
    ):
        return
    raise RuntimeError(
        "constraint tables require either the vllm-gr V1 raw-logits hook or "
        "the V1/V2 persistent-stage execute/sample_tokens hooks on active runner "
        f"{type(runner).__name__}. Constraint Tables cannot start on this "
        "worker; refusing to degrade to unconstrained sampling."
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
