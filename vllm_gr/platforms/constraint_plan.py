# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Constraint identities shared by engine admission, the frontend and workers."""

from __future__ import annotations

from typing import Any

from vllm_gr.config import ConstraintTableConfig

from .plan import BackendPlan


def constraint_plan_choices(gr: dict[str, Any], *, legacy_id: str | None) -> dict[str, Any]:
    """Declare the provider's historical execution ID unless a table opts into another."""
    table = gr.get("constraint_table") or {}
    if table.get("enabled"):
        explicit = table.get("implementation_id")
        if (
            explicit is None
            and legacy_id is not None
            and gr.get("beam_execution_mode") == "v1"
            and table["backend"] != legacy_id
        ):
            raise ValueError(
                f"beam_search_v1 requires an enabled canonical {legacy_id.upper()} constraint table"
            )
        return {
            "constraint": "canonical",
            "constraint_implementation": explicit or table["backend"],
        }
    if gr.get("constraint_backend") == "constraint_table":
        return {"constraint": "legacy", "constraint_implementation": legacy_id}
    return {"constraint": "none", "constraint_implementation": None}


def admitted_constraint_id(plan: BackendPlan) -> str:
    """Verify that a provider admitted the enabled table requested by EngineArgs."""
    if plan.constraint_mode != "canonical":
        raise ValueError("An enabled canonical constraint table requires a canonical backend plan")
    request_table = plan.request.config.to_dict().get("constraint_table") or {}
    explicit = request_table.get("implementation_id")
    # An engine plan from before implementation IDs may carry only the historical enum.
    fallback = request_table.get("backend") if explicit is None else None
    admitted = plan.constraint_implementation
    if admitted is None and "constraint_implementation" not in plan.choices.to_dict():
        admitted = fallback
    if not isinstance(admitted, str) or not admitted.strip():
        raise ValueError("Backend plan requires a non-empty canonical constraint implementation")
    if explicit is not None and admitted != explicit:
        raise ValueError("Backend plan does not honor the requested constraint implementation")
    return admitted


def canonical_constraint_id(table: ConstraintTableConfig, plan: BackendPlan | None) -> str:
    """Read the admitted identity and reject an edited or incompatible table selection."""
    configured = table.implementation_id or table.backend.value
    if plan is None:
        return configured
    request_table = plan.request.config.to_dict().get("constraint_table") or {}
    requested = request_table.get("implementation_id") or request_table.get("backend")
    if not table.enabled or not request_table.get("enabled") or configured != requested:
        raise ValueError("Constraint table implementation differs from the admitted request")
    return admitted_constraint_id(plan)
