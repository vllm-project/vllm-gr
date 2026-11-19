# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Platform adapter factory for canonical constraint tables."""

from __future__ import annotations

from typing import Any

from vllm_gr.platforms.services import constraint_execution_override, planning_provider


def _backend_label(provider: Any) -> str:
    """Describe the active provider for a startup diagnostic.

    The backend ID is the GR implementation; tensor placement stays available
    through the platform identity when the two differ.
    """
    backend_id = getattr(provider, "backend_id", None)
    return repr(backend_id) if isinstance(backend_id, str) else repr(provider)


def create_constraint_execution(
    *,
    device: str | Any | None = None,
    plan: Any = None,
) -> Any:
    """Create the platform constrained-top-k adapter, failing explicitly."""
    override = constraint_execution_override(device=device)
    if override is not None:
        return override

    from vllm_gr.platforms.loader import activate_current_backend

    provider = activate_current_backend()
    if plan is None:
        adapter = provider.create_constrained_topk_execution(device=device)
    else:
        adapter = planning_provider(provider).create_constrained_topk_execution(
            plan=plan, device=device
        )
    if adapter is None:
        raise RuntimeError(
            "constraint.mode='table' cannot start: the active backend "
            f"{_backend_label(provider)} does not implement Constraint Table execution."
        )
    if not adapter.is_available():
        raise RuntimeError(
            "constraint.mode='table' cannot start: the active backend "
            f"{_backend_label(provider)} provides a Constraint Table adapter but its custom "
            "operator is unavailable (is_available()=False)."
        )
    return adapter
