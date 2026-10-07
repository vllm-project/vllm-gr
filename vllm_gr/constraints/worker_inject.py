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

import torch

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


def _publish_canonical_backend_from_resource(runner: Any, resource: Any) -> bool:
    """Build a ready CUDA backend from a cached resource and publish it.

    Used when constraint tables arrive on a CUDA worker before the canonical
    backend exists (e.g. an injection that precedes ``load_model`` promotion
    or standalone runners). Publishes ``runner.constraint_backend`` and binds
    the LogitsProcessor catalog window so both the engine-driven hook and the
    P2D gate/capture see one canonical owner. Returns ``False`` for non-CUDA
    devices (the platform adapter path stays in charge there).
    """
    device = getattr(runner, "device", None)
    if device is None or device.type != "cuda":
        return False
    from vllm_gr.config import GRConfig
    from vllm_gr.constraints.artifact_compiler import constraint_table_from_resource
    from vllm_gr.constraints.loader import load_cuda_backend_from_table
    from vllm_gr.v1.worker.model_runner_patch import (
        MAX_CONSTRAINT_TOP_K,
        _install_constraint_vocab_window,
        _runner_constraint_max_rows,
    )

    gr_config = GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))
    max_rows = _runner_constraint_max_rows(gr_config)
    dtype = getattr(getattr(runner, "model_config", None), "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"canonical constraint backend requires model_config.dtype, got {dtype!r}"
        )
    backend = load_cuda_backend_from_table(
        constraint_table_from_resource(resource),
        device=device,
        dtype=dtype,
        max_top_k=MAX_CONSTRAINT_TOP_K,
        max_rows=max_rows,
    )
    runner.constraint_backend = backend
    _install_constraint_vocab_window(runner, backend)
    logger.info(
        "vllm-gr: worker published canonical constraint backend "
        "digest=%s from the injected resource",
        resource.digest,
    )
    return True


def load_constraint_tables_on_worker(worker: Any) -> bool:
    """Preload the injected constraint tables onto the runner device.

    Idempotent: a runner that already owns a canonical backend
    (``constraint_backend``, published by the startup catalog promotion or a
    previous injection) or a loaded adapter (``_gr_constraint_execution``) is
    left untouched — the tables are never uploaded twice.  When the worker has
    injected tables (the driver broadcast them via
    ``EngineCore.collective_rpc``) and a runner is available, the platform
    adapter is created and ``load_tables`` moves the resource to the device —
    at weight-load time (``init_device``) or as soon as the injection arrives
    (``set_gr_constraint_resources``), so the runtime never pays a lazy build
    / H2D on the first constrained step.  On CUDA workers the canonical
    backend is preferred over the legacy adapter.

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
    if getattr(runner, "constraint_backend", None) is not None:
        return True
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

    resource = manager.require(digest)
    if getattr(
        getattr(runner, "device", None), "type", ""
    ) == "cuda" and _publish_canonical_backend_from_resource(runner, resource):
        return True

    from vllm_gr.constraints.startup import create_constraint_execution

    adapter = create_constraint_execution(device=getattr(runner, "device", None))
    adapter.load_tables(resource)
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
