# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Shared lazy loader for the xLLM Ascend operator library.

The CANN vendor library ``libxllm_ops.so`` is linked through the
``custom_ops_lib`` Python binding.  ``cpp_extension.load`` is only a
best-effort pre-step (it can fail when the shared library is already loaded
or when ``torch.utils.cpp_extension`` is not directly reachable); the
authoritative availability probe is importing ``custom_ops_lib``.
"""

from __future__ import annotations

from types import ModuleType

import torch


def load_xllm_ops() -> None:
    """Load the shared xLLM extension before probing its Python bindings.

    The load is idempotent and best-effort: the extension may already be
    loaded by the runtime, by an LD_PRELOAD/setup script, or installed
    directly into site-packages.  The availability probes below remain the
    authoritative checks.
    """
    if hasattr(torch.ops, "xllm_ops"):
        return
    try:
        from torch.utils import cpp_extension

        cpp_extension.load(
            name="xllm_ops",
            sources=[],
            extra_ldflags=["-lxllm_ops"],
        )
    except Exception:
        # The extension may already be loaded by the runtime or by an
        # LD_PRELOAD/setup script.  The availability probe below is the
        # authoritative check.
        pass


def custom_ops() -> ModuleType | None:
    """Load the xLLM Python binding lazily, or ``None`` when unavailable.

    ``custom_xllm_math`` is the CANN OPP vendor name, not the Python module
    name.  Its environment script makes ``libcust_opapi.so`` discoverable;
    the callable Python API is exported by ``custom_ops_lib``.
    """
    load_xllm_ops()
    try:
        import custom_ops_lib as custom_ops  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        return None
    return custom_ops


def has_op(name: str) -> bool:
    """Return whether ``custom_ops_lib`` exports the given operator."""
    module = custom_ops()
    return module is not None and callable(getattr(module, name, None))


__all__ = ["custom_ops", "has_op", "load_xllm_ops"]
