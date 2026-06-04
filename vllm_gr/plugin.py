# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM general plugin entry point for vllm-GR.

Registered via ``pyproject.toml`` under ``[project.entry-points."vllm.general_plugins"]``.
vLLM calls ``load_general_plugins()`` in every process (main, EngineCore subprocess,
Worker subprocess), which discovers and executes this entry point.

This module is the formal runtime initialization boundary for vllm-GR.
It is responsible only for:

- applying GR monkey patches
- registering GR config hooks
- registering GR model architectures with vLLM
"""

import logging
from importlib import import_module
import os

logger = logging.getLogger("vllm_gr.plugin")


def register_gr_configs() -> bool:
    """Register all GR config integrations with the active runtime."""
    hstu_ok = True
    fuxi_ok = True

    try:
        from vllm_gr.config.hstu_vllm_registration import ensure_hstu_config_registered

        hstu_ok = ensure_hstu_config_registered()
    except ModuleNotFoundError:
        logger.debug("Skipping HSTU config registration in stubbed runtime")

    try:
        from vllm_gr.config.fuxi_vllm_registration import ensure_fuxi_config_registered

        fuxi_ok = ensure_fuxi_config_registered()
    except ModuleNotFoundError:
        logger.debug("Skipping FuXi config registration in stubbed runtime")

    return hstu_ok and fuxi_ok


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
    """Discover and register all adapter families.

    Scans ``vllm_gr.adapters`` for sub-packages.  Each sub-package
    (e.g. ``hstu/``) should have:

    - ``__init__.py`` calling ``register_family()`` (Layer 1, pure Python)
    - ``adapter.py`` calling ``register_adapter_class()`` (Layer 2, heavy)

    Adding a new model family requires only creating the adapter package
    directory under ``vllm_gr/adapters/`` — no changes to this function.
    """
    from pathlib import Path

    from vllm_gr.adapters.registry import get_registered_families

    adapters_dir = Path(__file__).resolve().parent / "adapters"
    registered = get_registered_families()

    for entry in sorted(adapters_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name in {
            "__pycache__",
            "pangu_llm",
        }:
            continue
        if not (entry / "__init__.py").exists():
            continue
        family = entry.name
        if family in registered:
            continue
        try:
            # Layer 1: register_family() in __init__.py (pure Python)
            import_module(f"vllm_gr.adapters.{family}")
            # Layer 2: register_adapter_class() in adapter.py (heavy)
            import_module(f"vllm_gr.adapters.{family}.adapter")
        except ModuleNotFoundError as exc:
            # Only suppress if the adapter package itself is missing.
            # If a transitive dependency is missing, exc.name will
            # point to that dependency — let it propagate.
            adapter_module = f"vllm_gr.adapters.{family}.adapter"
            if exc.name in (f"vllm_gr.adapters.{family}", adapter_module):
                logger.debug(
                    "Adapter package vllm_gr.adapters.%s not found, skipping",
                    family,
                )
            else:
                raise


def initialize_runtime(*, strict_gr: bool = True) -> dict[str, int | bool]:
    """Initialize the current process for vllm-GR runtime usage.

    Safe to call repeatedly. The underlying patch and registration helpers are
    idempotent, so this can be used both by the vLLM plugin loader and by
    explicit engine startup paths in the main process.

    When ``strict_gr`` is False (default for the vLLM plugin entry point), GR
    config/model registration failures are logged but do not abort startup so
    causal-LM checkpoints can use LLM passthrough mode.
    """
    from vllm_gr.patch import re_apply_patches

    patch_count = re_apply_patches()
    config_registered = register_gr_configs()
    models_registered = register_gr_model_architectures()

    _register_adapter_families()

    state = {
        "patch_count": patch_count,
        "config_registered": config_registered,
        "models_registered": models_registered,
        "strict_gr": strict_gr,
    }
    if strict_gr:
        if not config_registered:
            raise RuntimeError(
                "vllm-GR runtime initialization failed: "
                "HSTU config registration did not succeed"
            )
        if not models_registered:
            raise RuntimeError(
                "vllm-GR runtime initialization failed: "
                "GR model architecture registration did not succeed"
            )
    else:
        if not config_registered:
            logger.warning(
                "vllm-GR config registration incomplete (LLM passthrough mode)"
            )
        if not models_registered:
            logger.warning(
                "vllm-GR model architecture registration incomplete "
                "(LLM passthrough mode)"
            )
    logger.debug(
        "vllm-GR runtime initialized: patches=%d, config_registered=%s, "
        "models_registered=%s, strict_gr=%s",
        patch_count,
        config_registered,
        models_registered,
        strict_gr,
    )
    return state


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    strict_gr = os.getenv("VLLM_GR_STRICT", "0") == "1"
    state = initialize_runtime(strict_gr=strict_gr)
    logger.debug("vllm-GR plugin loaded (%d patches applied)", state["patch_count"])
