# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Two-layer adapter registry for GR model families.

**Layer 1 — family metadata** (pure Python, safe at arg_utils time):
Worker class paths, defaults function references.  No heavy imports.

**Layer 2 — adapter classes** (heavy, engine/worker instantiation time):
Actual ``GRModelAdapter`` subclasses that import torch/torchrec.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

logger = logging.getLogger("vllm_gr.adapters.registry")

if TYPE_CHECKING:
    from vllm_gr.adapters.base import GRModelAdapter

# ── Layer 1: lightweight (no heavy imports) ──────────────────────

_FAMILY_ENTRIES: dict[str, dict[str, Any]] = {}


def register_family(
    model_type: str,
    gpu_worker: str,
    npu_worker: str,
    defaults_fn: str | None = None,
    payload_type: str | None = None,
) -> None:
    """Register a model family's metadata.

    Parameters
    ----------
    model_type:
        Unique family identifier (e.g. ``"hstu"``).
    gpu_worker:
        Dotted path to the GPU worker class (e.g.
        ``"vllm_gr.worker.gr_worker.GRWorker"``).
    npu_worker:
        Dotted path to the NPU worker class.
    defaults_fn:
        Dotted path to a callable that applies family-specific defaults
        to engine args.  Stored as a string and imported lazily so that
        Layer 1 registration never triggers heavy imports.
    payload_type:
        Dotted path to the payload class (e.g.
        ``"vllm_gr.engine:GRFeaturesPayload"``).  Stored as a string
        and resolved lazily by ``get_payload_type()``.
    """
    _FAMILY_ENTRIES[model_type] = {
        "gpu_worker": gpu_worker,
        "npu_worker": npu_worker,
        "defaults_fn": defaults_fn,
        "payload_type": payload_type,
    }


def get_worker_cls_path(model_type: str, is_npu: bool) -> str | None:
    """Return the dotted worker class path for the given family and platform."""
    entry = _FAMILY_ENTRIES.get(model_type)
    if entry is None:
        return None
    return entry.get("npu_worker" if is_npu else "gpu_worker")


def get_family_info(model_type: str) -> dict[str, Any] | None:
    """Return the full family metadata dict, or ``None``."""
    return _FAMILY_ENTRIES.get(model_type)


def get_registered_families() -> set[str]:
    """Return the set of currently registered model family names."""
    return set(_FAMILY_ENTRIES.keys())


def apply_family_defaults(model_type: str, engine_args: Any) -> None:
    """Import and call the family's ``defaults_fn`` on *engine_args*.

    No-op when the family has no ``defaults_fn`` or is not registered.
    """
    entry = _FAMILY_ENTRIES.get(model_type)
    if entry is None:
        return
    fn_path: str | None = entry.get("defaults_fn")
    if fn_path is None:
        return
    fn = _import_dotted_path(fn_path)
    fn(engine_args)


def get_payload_type(model_type: str) -> type:
    """Return the payload class for *model_type*.

    Resolves the ``payload_type`` dotted path registered with
    ``register_family()``.  Raises ``KeyError`` when the family is
    unknown or has no ``payload_type`` registered.
    """
    entry = _FAMILY_ENTRIES.get(model_type)
    if entry is None:
        registered = sorted(_FAMILY_ENTRIES.keys())
        raise KeyError(
            f"No family registered for model_type={model_type!r}. "
            f"Registered families: {registered}"
        )
    pt_path: str | None = entry.get("payload_type")
    if pt_path is None:
        raise KeyError(
            f"Family {model_type!r} has no payload_type registered"
        )
    return _import_dotted_path(pt_path)


# ── Layer 2: adapter class (heavy, engine/worker time) ──────────

_ADAPTER_CLASSES: dict[str, type["GRModelAdapter"]] = {}


def register_adapter_class(
    model_type: str,
    cls: type["GRModelAdapter"],
) -> None:
    """Register an adapter class for *model_type*.

    Called at module-import time in the adapter subpackage (e.g.
    ``hstu/adapter.py``) so that ``create_adapter`` can look it up later.
    """
    _ADAPTER_CLASSES[model_type] = cls


def get_adapter_class(model_type: str) -> type["GRModelAdapter"]:
    """Return the adapter class for *model_type*.

    Raises ``KeyError`` with a helpful message when the family is unknown.
    """
    cls = _ADAPTER_CLASSES.get(model_type)
    if cls is not None:
        return cls
    registered = sorted(_ADAPTER_CLASSES.keys())
    raise KeyError(
        f"No adapter class registered for model_type={model_type!r}. "
        f"Registered families: {registered}"
    )


def create_adapter(
    model_type: str,
    vllm_config: Any,
    gr_config: Any,
) -> "GRModelAdapter":
    """Instantiate the adapter for *model_type*.

    Thin wrapper around ``ensure_adapter_class_registered`` +
    ``get_adapter_class`` + ``cls(vllm_config, gr_config)``.
    """
    ensure_adapter_class_registered(model_type)
    cls = get_adapter_class(model_type)
    return cls(vllm_config, gr_config)


def ensure_adapter_class_registered(model_type: str) -> None:
    """Ensure the adapter class for *model_type* is in the Layer-2 registry.

    If the class is already registered, this is a no-op.  Otherwise it
    dynamically imports ``vllm_gr.adapters.<model_type>.adapter`` which
    should call ``register_adapter_class()`` at module level.

    Raises the original exception if the adapter module exists but
    fails to import (syntax error, missing dependency, etc.).
    Only ``ModuleNotFoundError`` where ``exc.name`` matches the target
    adapter module itself is silently ignored.  Any other import failure
    (missing transitive dependency, syntax error, etc.) is re-raised.
    """
    try:
        get_adapter_class(model_type)
        return  # already registered
    except KeyError:
        pass

    target_module = f"vllm_gr.adapters.{model_type}.adapter"
    try:
        importlib.import_module(target_module)
    except ModuleNotFoundError as exc:
        # Only suppress if the *target* module itself is missing.
        # If a transitive dependency (e.g. torchrec) is missing,
        # exc.name will point to that dependency — re-raise it.
        if exc.name == target_module:
            pass
        else:
            raise


# ── Internal helpers ─────────────────────────────────────────────

def resolve_model_type(vllm_config: Any) -> str:
    """Resolve the GR model type from a vllm_config.

    Checks ``hf_config.model_type`` first, then falls back to
    ``gr_config.model_family``, then to ``"hstu"`` as a last resort.
    """
    model_type = getattr(
        getattr(getattr(vllm_config, "model_config", None), "hf_config", None),
        "model_type", None,
    )
    if model_type:
        return model_type
    return getattr(
        getattr(vllm_config, "gr_config", None), "model_family", None,
    ) or "hstu"


def _import_dotted_path(dotted_path: str) -> Callable:
    """Import and return the object at *dotted_path*.

    ``"vllm_gr.adapters.hstu._defaults:apply_hstu_defaults"`` resolves to
    the ``apply_hstu_defaults`` function.
    """
    if ":" in dotted_path:
        module_path, attr = dotted_path.rsplit(":", 1)
    else:
        module_path, attr = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


# ── Reset (for testing) ─────────────────────────────────────────

def _reset() -> None:
    """Clear all registrations.  Only for use in tests."""
    _FAMILY_ENTRIES.clear()
    _ADAPTER_CLASSES.clear()


# ── Adapter discovery ───────────────────────────────────────────

def register_adapter_layer1_only() -> None:
    """Import adapter ``__init__.py`` files (Layer 1) without ``adapter.py``.

    Scans ``vllm_gr/adapters/`` for sub-packages.  Each sub-package
    ``__init__.py`` calls :func:`register_family` (pure Python, no heavy
    imports).  The ``adapter.py`` (Layer 2, heavy) is *not* imported.
    """
    registered = get_registered_families()
    adapters_dir = Path(__file__).resolve().parent

    for entry in sorted(adapters_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name == "__pycache__":
            continue
        if not (entry / "__init__.py").exists():
            continue
        family = entry.name
        if family in registered:
            continue
        try:
            importlib.import_module(f"vllm_gr.adapters.{family}")
        except ModuleNotFoundError as exc:
            if exc.name == f"vllm_gr.adapters.{family}":
                logger.debug(
                    "Adapter package vllm_gr.adapters.%s not found, skipping",
                    family,
                )
            else:
                raise
