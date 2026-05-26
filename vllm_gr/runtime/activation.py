# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime activation — the single entry-point that executes patches and registry.

:func:`activate` consumes a :class:`~vllm_gr.runtime.patch_plan.PatchPlan`
and conditionally applies monkey patches, config registrations, model
architecture registrations, and adapter imports.  It is idempotent.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm_gr._runtime_errors import GRInconsistencyError
from vllm_gr.runtime.patch_plan import PatchPlan

logger = logging.getLogger("vllm_gr.runtime.activation")


# ── Module-level activation guard ────────────────────────────────

_ACTIVATED: bool = False
_LAST_STATE: dict[str, Any] = {}
_LAST_PLAN: PatchPlan | None = None


def _reset_activation_state() -> None:
    """Clear activation guard.  Only for use in tests."""
    global _ACTIVATED, _LAST_STATE, _LAST_PLAN
    _ACTIVATED = False
    _LAST_STATE = {}
    _LAST_PLAN = None


def activate(plan: PatchPlan) -> dict[str, int | bool]:
    """Execute the runtime activation described by *plan*.

    This is the **only** function that calls into ``patch.py``, the config
    registration module, or the model-architecture registry.  It is fully
    idempotent — a second call with the same plan returns the cached state
    immediately.

    Raises :class:`GRInconsistencyError` if called again with a *different*
    plan (same process must not activate conflicting runtime selections).

    Returns a dict summarising what was done (patch count, registration
    status).
    """
    global _ACTIVATED, _LAST_STATE, _LAST_PLAN

    if _ACTIVATED:
        if plan != _LAST_PLAN:
            raise GRInconsistencyError(
                f"activate() called with conflicting plan: "
                f"previous family={_LAST_PLAN.family!r}, "
                f"new family={plan.family!r}.  "
                f"Each process must activate at most one RuntimeSelection."
            )
        return _LAST_STATE

    # ── HSTU config registration ────────────────────────────────
    if plan.needs_hstu_config:
        from vllm_gr.config.hstu_vllm_registration import (
            ensure_hstu_config_registered,
        )

        config_ok = ensure_hstu_config_registered()
    else:
        config_ok = True

    # ── Core monkey patches ─────────────────────────────────────
    if plan.needs_core_patches:
        from vllm_gr.patch import re_apply_patches

        patch_count = re_apply_patches()
    else:
        patch_count = 0

    # ── Model architecture registration ─────────────────────────
    if plan.needs_model_registry:
        from vllm_gr.plugin import register_gr_model_architectures

        models_ok = register_gr_model_architectures()
    else:
        models_ok = True

    # ── Adapter registration ────────────────────────────────────
    if plan.needs_adapter_layer2:
        from vllm_gr.plugin import _register_adapter_families

        _register_adapter_families()
    else:
        # Phase 2+: only register Layer 1 (pure Python, no heavy import)
        from vllm_gr.adapters.registry import register_adapter_layer1_only
        register_adapter_layer1_only()

    state: dict[str, int | bool] = {
        "patch_count": patch_count,
        "config_registered": config_ok,
        "models_registered": models_ok,
    }

    if plan.needs_hstu_config and not config_ok:
        raise RuntimeError(
            "vllm-GR runtime activation failed: "
            "HSTU config registration did not succeed"
        )
    if plan.needs_model_registry and not models_ok:
        raise RuntimeError(
            "vllm-GR runtime activation failed: "
            "GR model architecture registration did not succeed"
        )

    _ACTIVATED = True
    _LAST_STATE = state
    _LAST_PLAN = plan

    logger.debug(
        "vllm-GR runtime activated: family=%s patches=%d config=%s models=%s",
        plan.family,
        patch_count,
        config_ok,
        models_ok,
    )
    return state


def activate_gr_recommend_runtime(model_arch: str) -> dict[str, int | bool]:
    """Activate the Phase-2 recommendation runtime for *model_arch*.

    GR entry points call this before handing model paths to vLLM so any
    model-specific config mapping is present before ``ModelConfig`` loads.
    """
    from vllm_gr._runtime_selection import RuntimeSelection
    from vllm_gr._static_descriptors import FAMILY_GR_RECOMMEND
    from vllm_gr.runtime.patch_plan import build_patch_plan

    selection = RuntimeSelection(
        family=FAMILY_GR_RECOMMEND,
        model_arch=model_arch,
    )
    return activate(build_patch_plan(selection))
