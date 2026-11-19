# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Build, adopt and release the services owned by one native runner."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from types import ModuleType
from typing import Any

from .plan import BackendPlan, read_backend_plan
from .planning import normalize_backend_request, resolve_backend_plan

# Compatibility imports for providers written against the original binding API.
from .readiness import validate_worker_readiness as validate_worker_readiness
from .readiness import (
    verify_engine_backend_readiness as verify_engine_backend_readiness,
)
from .readiness import worker_backend_readiness as worker_backend_readiness
from .services import BoundService, constraint_execution_override, planning_provider

logger = logging.getLogger(__name__)
_BINDING_ATTR = "_gr_backend_binding"


def _implementation_id(service: Any) -> str | None:
    if service is None:
        return None
    if isinstance(service, BoundService):
        service = service.implementation
    if isinstance(service, ModuleType):
        return service.__name__
    declared = getattr(service, "implementation_id", None)
    if isinstance(declared, str):
        return declared
    return f"{type(service).__module__}.{type(service).__qualname__}"


def _close_services(services: list[Any]) -> None:
    seen = set()
    for service in reversed(services):
        if service is None or isinstance(service, BoundService) or id(service) in seen:
            continue
        seen.add(id(service))
        close = getattr(service, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Continue releasing the remaining services, including when
                # cleanup follows a more useful startup exception.
                logger.exception("Failed to close backend service %s", _implementation_id(service))


@dataclass(frozen=True)
class BackendBinding:
    plan: BackendPlan
    runner_adapter: Any
    attention_executor: Any
    constraint_execution: Any

    def report(self) -> dict[str, Any]:
        services = {
            "runner": _implementation_id(self.runner_adapter),
            "attention": _implementation_id(self.attention_executor),
            "constraint": _implementation_id(self.constraint_execution),
        }
        if self.plan.choices.to_dict().get("operator_policy"):
            stage = self.runner_adapter.stage
            services.update(
                stage=stage.implementation_id, operator=stage.decision.implementation_id
            )
        return {"ready": True, "fingerprint": self.plan.fingerprint, "services": services}


def _clear_unplanned_stage(runner: Any, keep: Any = None) -> None:
    unplanned = getattr(runner, "_gr_unplanned_stage", None)
    if unplanned is not None:
        runner._gr_unplanned_stage = None
        if unplanned is not keep:
            _close_services([unplanned])


def close_backend_binding(runner: Any) -> None:
    binding = getattr(runner, _BINDING_ATTR, None)
    setattr(runner, _BINDING_ATTR, None)
    adapter = binding.runner_adapter if binding is not None else None
    _clear_unplanned_stage(runner, getattr(adapter, "stage", None))
    if binding is not None:
        if getattr(runner, "_gr_constraint_execution", None) is binding.constraint_execution:
            runner._gr_constraint_execution = None
            runner._gr_constraint_plan_fingerprint = None
        _close_services(
            [binding.runner_adapter, binding.attention_executor, binding.constraint_execution]
        )


def bind_backend_services(runner: Any) -> BackendBinding | None:
    """Validate the engine plan locally, then publish all services atomically."""
    from .loader import activate_current_backend

    config = getattr(runner, "vllm_config", None)
    plan = read_backend_plan(config)
    if plan is None:
        return None  # Unplanned native configs retain the pre-plan API path.
    existing = getattr(runner, _BINDING_ATTR, None)
    if existing is not None:
        if not isinstance(existing, BackendBinding):
            raise RuntimeError("Runner contains an invalid backend binding")
        if existing.plan != plan:
            raise RuntimeError("Runner backend plan changed; reload its owned resources first")
        # Native KV-cache profiling may auto-fit max_model_len after model load.
        # Readiness attests the admitted plan and the services already bound.
        return existing
    provider = activate_current_backend()
    local = resolve_backend_plan(provider, normalize_backend_request(config, provider))
    if local.fingerprint != plan.fingerprint:
        raise RuntimeError(
            f"Worker backend plan mismatch: engine={plan.fingerprint}, worker={local.fingerprint}"
        )
    selected = planning_provider(provider)
    selected.validate_worker(plan=plan, native_runner=runner)
    choices = plan.choices.to_dict()
    device = getattr(runner, "device", None)
    created: list[Any] = []
    try:
        adapter = selected.create_model_runner_adapter(plan=plan, native_runner=runner)
        created.append(adapter)
        if adapter is None:
            raise RuntimeError("Backend did not bind the required runner adapter")
        for phase in getattr(type(runner), "_gr_runner_methods", ()):
            if not callable(getattr(adapter, phase, None)):
                raise RuntimeError(f"Backend runner adapter does not implement {phase}()")
        if choices.get("operator_policy"):
            stage = getattr(adapter, "stage", None)
            if getattr(stage, "implementation_id", None) != choices.get("stage"):
                raise RuntimeError("Worker did not bind the admitted stage implementation")
            for method in (
                "execute",
                "retire",
                "capture_startup_graph",
                "admit_unconstrained_beam",
            ):
                if not callable(getattr(stage, method, None)):
                    raise RuntimeError(f"Backend stage does not implement {method}()")
            operators = getattr(stage, "decision", None)
            if (
                not getattr(operators, "implementation_id", None)
                or not callable(getattr(operators, "select", None))
                or not callable(getattr(operators, "reorder", None))
            ):
                raise RuntimeError("Worker did not bind a Beam decision operator")
        attention = None
        if choices.get("requires_attention", False):
            attention = selected.create_attention_executor(plan=plan, device=device)
            created.append(attention)
            if attention is None:
                raise RuntimeError("Backend did not bind the required attention executor")
        constraint = None
        if choices.get("constraint") == "legacy":
            constraint = getattr(runner, "_gr_constraint_execution", None)
            if constraint is not None:
                if getattr(runner, "_gr_constraint_plan_fingerprint", None) != plan.fingerprint:
                    raise RuntimeError(
                        "Preloaded constraint execution belongs to another backend plan"
                    )
            else:
                constraint = constraint_execution_override(device=device)
                if constraint is None:
                    constraint = selected.create_constrained_topk_execution(
                        plan=plan, device=device
                    )
            created.append(constraint)
            if constraint is None or not constraint.is_available():
                raise RuntimeError("Backend constraint execution is unavailable")
        elif choices.get("constraint") == "canonical":
            # Canonical artifact ownership stays with the existing loader until
            # #714. Its load/warmup must already have completed successfully.
            if getattr(runner, "constraint_backend", None) is None:
                raise RuntimeError("Required canonical constraint backend was not loaded")
        binding = BackendBinding(plan, adapter, attention, constraint)
        setattr(runner, _BINDING_ATTR, binding)
        _clear_unplanned_stage(runner, getattr(adapter, "stage", None))
        return binding
    except BaseException:
        preloaded = getattr(runner, "_gr_constraint_execution", None)
        if preloaded is not None and any(service is preloaded for service in created):
            runner._gr_constraint_execution = None
            runner._gr_constraint_plan_fingerprint = None
        _close_services(created)
        raise


def bind_after_model_load(
    original: Any, *, close_runtime: Callable[[Any], None] | None = None
) -> Any:
    if getattr(original, "_gr_binds_backend_after_load", False):
        return original

    @wraps(original)
    def load(self: Any, *args: Any, **kwargs: Any) -> Any:
        if close_runtime is not None:
            # Reject undrained native work before replacing the current generation.
            close_runtime(self)
        try:
            result = original(self, *args, **kwargs)
            bind_backend_services(self)
        except BaseException:
            from vllm_gr.v1.worker.model_runner_patch import close_constraint_backend

            if close_runtime is not None:
                try:
                    close_runtime(self)
                except Exception:
                    logger.exception("Failed to release native runtime after model-load failure")
            try:
                close_constraint_backend(self)
            except Exception:
                logger.exception("Failed to release backend resources after model-load failure")
            raise
        return result

    setattr(load, "_gr_binds_backend_after_load", True)
    return load
