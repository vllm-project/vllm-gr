# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM general plugin entry point for vllm-GR.

Registered via ``pyproject.toml`` under ``[project.entry-points."vllm.general_plugins"]``.
vLLM calls ``load_general_plugins()`` in every process (main, EngineCore subprocess,
Worker subprocess), which discovers and executes this entry point.

Phase 2 — ``register()`` is metadata-only.  It registers Layer 1 adapter
strings (pure Python, no heavy imports) and, if the environment variable
``_VLLM_GR_RUNTIME_SELECTION`` is set (Phase 4 subprocess propagation),
calls ``activate()`` with the deserialized selection.

Runtime activation (patches, config, model registry, adapter Layer 2)
is performed exclusively by :func:`~vllm_gr.runtime.activation.activate`,
called either:

1. Explicitly by ``GREngineArgs.create_engine_config()`` in the main process.
2. Via ENV propagation in subprocesses (Phase 4).

``register()`` **never** performs activation in the absence of an explicit
ENV signal — the native vLLM path remains zero-pollution.
"""

import logging
import os
import warnings
from importlib import import_module
from pathlib import Path

from vllm_gr.adapters.registry import register_adapter_layer1_only

logger = logging.getLogger("vllm_gr.plugin")


# ── Adapter registration helpers ─────────────────────────────────


def register_gr_model_architectures() -> bool:
    """Register all GR model architectures with vLLM."""
    model_registry = import_module("vllm_gr.model_executor.models.registry")

    register_fn = getattr(
        model_registry,
        "register_gr_model_architectures",
        None,
    )
    if register_fn is None:
        register_fn = model_registry.register_gr_models_to_vllm
    return register_fn()


def _register_adapter_families() -> None:
    """Discover and register all adapter families (Layer 1 + Layer 2).

    Layer 1 delegates to
    :func:`~vllm_gr.adapters.registry.register_adapter_layer1_only`.
    Layer 2 scans for ``adapter.py`` in each adapter sub-package.
    """
    # Layer 1: import __init__.py files (pure Python metadata).
    register_adapter_layer1_only()

    # Layer 2: import adapter.py files (heavy, guarded by _ADAPTER_CLASSES).
    adapters_dir = Path(__file__).resolve().parent / "adapters"

    for entry in sorted(adapters_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name == "__pycache__":
            continue
        if not (entry / "__init__.py").exists():
            continue
        family = entry.name
        try:
            import_module(f"vllm_gr.adapters.{family}.adapter")
        except ModuleNotFoundError as exc:
            adapter_module = f"vllm_gr.adapters.{family}.adapter"
            if exc.name in (f"vllm_gr.adapters.{family}", adapter_module):
                logger.debug(
                    "Adapter module %s not found, skipping",
                    adapter_module,
                )
            else:
                raise


# ── Deprecated backward-compat shim ──────────────────────────────


def initialize_runtime() -> dict[str, int | bool]:
    """Deprecated: use ``activate(build_patch_plan(selection))`` instead.

    Kept for backward compatibility.  Internally constructs a
    ``gr_recommend`` PatchPlan and delegates to
    :func:`~vllm_gr.runtime.activation.activate`.
    """
    warnings.warn(
        "initialize_runtime() is deprecated; "
        "use activate() with an explicit PatchPlan",
        DeprecationWarning,
        stacklevel=2,
    )
    from vllm_gr.runtime.activation import activate
    from vllm_gr.runtime.patch_plan import PatchPlan

    plan = PatchPlan(
        family="gr_recommend",
        capabilities=frozenset(),
        model_arch="hstu",
        needs_core_patches=True,
        needs_hstu_config=True,
        needs_model_registry=True,
        needs_adapter_layer2=True,
    )
    return activate(plan)


# ── Plugin entry point ───────────────────────────────────────────


def register() -> None:
    """Entry point used by vLLM's general plugin loader.

    Metadata-only: registers Layer 1 adapter strings (pure Python).
    Does **not** apply patches, register configs, or import heavy
    dependencies.  The native vLLM path remains zero-pollution.

    If ``_VLLM_GR_RUNTIME_SELECTION`` is set in the environment
    (Phase 4 subprocess propagation), deserializes the selection and
    calls ``activate()``.  Otherwise does nothing beyond Layer 1.
    """
    register_adapter_layer1_only()

    # Subprocess propagation (Phase 4): if the main process has written
    # a serialized RuntimeSelection to the environment, deserialize and
    # activate accordingly.
    from vllm_gr._runtime_selection import (
        ENV_PROPAGATION_RUNTIME_SELECTION,
        RuntimeSelection,
    )

    serialized = os.environ.get(ENV_PROPAGATION_RUNTIME_SELECTION)
    if serialized:
        selection = RuntimeSelection.deserialize(serialized)
        from vllm_gr.runtime.activation import activate
        from vllm_gr.runtime.patch_plan import build_patch_plan

        plan = build_patch_plan(selection)
        state = activate(plan)
        logger.debug(
            "vllm-GR plugin activated from ENV: family=%s (%d patches)",
            selection.family,
            state.get("patch_count", 0),
        )
