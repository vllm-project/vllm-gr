# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Explicit plan binding and the versioned legacy provider delegation path."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_gr.config.backend_plan import (
    PLAN_API_VERSION,
    BackendPlan,
    BackendRequest,
    PlanData,
)

if TYPE_CHECKING:
    from .interface import ModelRunnerAdapter

logger = logging.getLogger(__name__)

PLAN_METHODS = (
    "resolve",
    "validate_worker",
    "create_model_runner_adapter",
    "create_attention_executor",
    "create_constrained_topk_execution",
)


def constraint_execution_override(*, device: Any) -> Any:
    """Honor the shared CI/dev Torch oracle before calling any provider factory."""
    override = os.environ.get("VLLM_GR_CONSTRAINT_EXECUTION")
    if override is None:
        return None
    if override != "torch":
        raise ValueError(
            f"unsupported VLLM_GR_CONSTRAINT_EXECUTION={override!r}; expected 'torch' or unset"
        )
    from vllm_gr.constraints.execution import TorchConstrainedTopKExecution

    logger.info("vllm-gr: constraint execution overridden to Torch oracle for CI/dev")
    return TorchConstrainedTopKExecution(device=device)


def validate_required_services(
    plan: BackendPlan, required: dict[str, str | None], reported: dict[str, Any], worker: str
) -> None:
    for name, expected in required.items():
        actual = reported.get(name)
        if not isinstance(actual, str) or not actual.strip():
            raise RuntimeError(f"{worker} has no bound {name}")
        if expected is not None and actual != expected:
            label = (
                "canonical constraint"
                if name == "constraint" and plan.constraint_mode == "canonical"
                else name
            )
            raise RuntimeError(
                f"{worker} {label} implementation mismatch: expected {expected}, got {actual}"
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
    """Bind a native runner and its admitted attention implementation."""

    plan_api_version = PLAN_API_VERSION
    adapter_revision = "native-runner-v1"

    def create_model_runner_adapter(
        self, *, plan: BackendPlan, native_runner: Any
    ) -> ModelRunnerAdapter:
        from vllm_gr.v1.worker.runner_lifecycle import bind_native_runner

        require_provider_plan(self, plan)
        return bind_native_runner(plan, native_runner)

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None:
        require_provider_plan(self, plan)
        # Keep the existing device and execution validators as the capability
        # authority until their owning adapter migrations. No second policy.
        if plan.request.config.to_dict()["beam_execution_mode"] == "v1":
            from vllm_gr.v1.engine.beam_admission import (
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

    def create_model_runner_adapter(self, *, plan: BackendPlan, native_runner: Any) -> Any:
        require_provider_plan(self.provider, plan)
        hooks = self.provider.get_model_runner_hooks()
        installed = getattr(type(native_runner), "_gr_runner_implementation_id", None)
        if installed is not None:
            from vllm_gr.v1.worker.runner_lifecycle import bind_native_runner

            # Legacy plans delegate to an existing module, without admitting
            # a new implementation choice on the provider's behalf.
            verify = getattr(hooks, "is_patched", None)
            if getattr(hooks, "__name__", None) != installed and not (
                callable(verify) and verify(type(native_runner))
            ):
                raise RuntimeError("Legacy runner hooks differ from the installed native runner")
            return bind_native_runner(plan, native_runner, implementation_id=installed)
        return BoundService(plan, hooks, native_runner=native_runner)

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
