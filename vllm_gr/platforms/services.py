# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Explicit plan binding and the versioned legacy provider delegation path."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .plan import PLAN_API_VERSION, BackendPlan, BackendRequest, PlanData

PLAN_METHODS = (
    "resolve",
    "validate_worker",
    "create_model_runner_adapter",
    "create_attention_executor",
    "create_constrained_topk_execution",
)


def validate_planning_api(provider: Any) -> None:
    version = getattr(provider, "plan_api_version", None)
    if version is None:
        return  # API-v1/v2 did not declare a planning extension.
    if type(version) is not int or version != PLAN_API_VERSION:
        raise ValueError(f"Unsupported backend plan_api_version={version!r}")
    missing = [name for name in PLAN_METHODS if not callable(getattr(provider, name, None))]
    if missing:
        raise ValueError(f"Backend planning API is missing methods: {missing}")
    revision = getattr(provider, "adapter_revision", None)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Backend planning API requires an adapter_revision compatibility string")


@dataclass(frozen=True)
class BoundService:
    """Bind a stateless implementation without moving resources off its owner."""

    plan: BackendPlan
    implementation: Any
    native_runner: Any = None
    device: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.implementation, name)


def require_provider_plan(provider: Any, plan: BackendPlan) -> None:
    if (plan.backend_id, plan.provider_api_version) != (provider.backend_id, provider.api_version):
        raise ValueError("Factory received a plan for another backend/provider API")


class BuiltinPlanSupport:
    """Temporary module delegation while lifecycle consumers migrate."""

    plan_api_version = PLAN_API_VERSION
    adapter_revision = "plan-bindings-v1"

    def create_model_runner_adapter(self, *, plan: BackendPlan, native_runner: Any) -> BoundService:
        require_provider_plan(self, plan)
        return BoundService(
            plan, import_module(plan.choices.to_dict()["runner"]), native_runner=native_runner
        )

    def create_attention_executor(self, *, plan: BackendPlan, device: Any) -> BoundService:
        require_provider_plan(self, plan)
        implementation = import_module(plan.choices.to_dict()["attention"])
        if not implementation.is_available():
            raise RuntimeError(
                f"Backend {plan.backend_id!r} attention implementation is unavailable"
            )
        return BoundService(plan, implementation, device=device)

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None:
        require_provider_plan(self, plan)
        from vllm_gr.v1.worker.model_runner_patch import _active_runner_module_name

        if _active_runner_module_name() != plan.choices.to_dict()["runner"]:
            raise RuntimeError(
                "Worker native runner selection differs from the admitted backend plan"
            )
        # Keep the existing device and execution validators as the capability
        # authority until their owning adapter migrations. No second policy.
        if plan.request.config.to_dict()["beam_execution_mode"] == "v1":
            from vllm_gr.v1.engine.beam_search_v1 import (
                validate_gr_device_config,
                validate_gr_execution_runtime,
            )

            validate_gr_execution_runtime(native_runner.vllm_config)
            validate_gr_device_config(native_runner.vllm_config)


@dataclass(frozen=True)
class LegacyPlanningProvider:
    """Delegate only the shipped API-v2 surface; infer no new capabilities."""

    provider: Any

    def resolve(self, request: BackendRequest) -> BackendPlan:
        if request.options.to_dict():
            raise ValueError(
                "Legacy providers cannot admit typed backend options; implement planning API 1"
            )
        gr = request.config.to_dict()
        return BackendPlan(
            self.provider.backend_id,
            self.provider.api_version,
            request,
            PlanData(
                {
                    "runner": "legacy-provider-hooks",
                    "attention": "legacy-provider-execution",
                    "constraint": "canonical"
                    if (gr["constraint_table"] or {}).get("enabled")
                    else ("legacy" if gr["constraint_backend"] == "constraint_table" else "none"),
                    "requires_attention": gr["attention_backend"] == "CUSTOM",
                    "compatibility": "provider-api-v2-delegation",
                }
            ),
        )

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None:
        del native_runner
        require_provider_plan(self.provider, plan)
        self.provider.check_environment()

    def create_model_runner_adapter(self, *, plan: BackendPlan, native_runner: Any) -> BoundService:
        require_provider_plan(self.provider, plan)
        return BoundService(
            plan, self.provider.get_model_runner_hooks(), native_runner=native_runner
        )

    def create_attention_executor(self, *, plan: BackendPlan, device: Any) -> BoundService:
        require_provider_plan(self.provider, plan)
        module = self.provider.get_attention_execution()
        if not module.is_available():
            raise RuntimeError("Legacy provider attention is unavailable")
        return BoundService(plan, module, device=device)

    def create_constrained_topk_execution(self, *, plan: BackendPlan, device: Any) -> Any:
        require_provider_plan(self.provider, plan)
        return self.provider.create_constrained_topk_execution(device=device)


def planning_provider(provider: Any) -> Any:
    validate_planning_api(provider)
    # Selection is by an explicit version, never by incidental method presence.
    if getattr(provider, "plan_api_version", None) is None:
        return LegacyPlanningProvider(provider)
    return provider
