# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Runner-owned services and the all-worker startup agreement gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps
from types import ModuleType
from typing import Any

from .plan import BackendPlan, PlanData, read_backend_plan
from .planning import normalize_backend_request, resolve_backend_plan
from .services import BoundService, planning_provider

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
        return {
            "ready": True,
            "fingerprint": self.plan.fingerprint,
            "services": {
                "runner": _implementation_id(self.runner_adapter),
                "attention": _implementation_id(self.attention_executor),
                "constraint": _implementation_id(self.constraint_execution),
            },
        }


def close_backend_binding(runner: Any) -> None:
    binding = getattr(runner, _BINDING_ATTR, None)
    setattr(runner, _BINDING_ATTR, None)
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
    provider = activate_current_backend()
    local = resolve_backend_plan(provider, normalize_backend_request(config, provider))
    if local.fingerprint != plan.fingerprint:
        raise RuntimeError(
            f"Worker backend plan mismatch: engine={plan.fingerprint}, worker={local.fingerprint}"
        )
    existing = getattr(runner, _BINDING_ATTR, None)
    if existing is not None:
        if not isinstance(existing, BackendBinding):
            raise RuntimeError("Runner contains an invalid backend binding")
        if existing.plan != plan:
            raise RuntimeError("Runner backend plan changed; reload its owned resources first")
        return existing
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
                constraint = selected.create_constrained_topk_execution(plan=plan, device=device)
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
        return binding
    except BaseException:
        _close_services(created)
        raise


def bind_after_model_load(original: Any) -> Any:
    @wraps(original)
    def load(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            bind_backend_services(self)
        except BaseException:
            from vllm_gr.v1.worker.model_runner_patch import close_constraint_backend

            close_constraint_backend(self)
            raise
        return result

    return load


def worker_backend_readiness(worker: Any) -> dict[str, Any]:
    """Callable RPC: native executors pass their Worker as the first argument."""
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("Worker has no model runner for backend plan validation")
    binding = bind_backend_services(runner)
    if binding is None:
        raise RuntimeError("Worker did not receive the engine backend plan")
    return binding.report()


def validate_worker_readiness(plan: BackendPlan, reports: Any, *, expected_workers: int) -> None:
    if type(expected_workers) is not int or expected_workers < 1:
        raise ValueError("Backend readiness requires a positive integer worker count")
    if not isinstance(reports, (list, tuple)) or len(reports) != expected_workers or not reports:
        raise RuntimeError(f"Backend readiness requires {expected_workers} worker reports")
    services = None
    for index, report in enumerate(reports):
        if not isinstance(report, dict) or report.get("ready") is not True:
            raise RuntimeError(f"Backend worker {index} is not ready")
        if report.get("fingerprint") != plan.fingerprint:
            raise RuntimeError(f"Backend worker {index} plan fingerprint mismatch")
        if not isinstance(report.get("services"), dict) or not report["services"].get("runner"):
            raise RuntimeError(f"Backend worker {index} has no bound runner")
        current = PlanData(report["services"])
        if services is not None and current != services:
            raise RuntimeError("Workers disagree on bound backend service implementations")
        services = current


def verify_engine_backend_readiness(engine: Any) -> None:
    plan = read_backend_plan(engine.vllm_config)
    if plan is None:
        return
    parallel = getattr(engine.vllm_config, "parallel_config", None)
    # Each DP EngineCore owns its local TP/PP executor; DP rank is not a worker
    # in this executor's collective. Native world_size is TP * PP.
    expected = getattr(parallel, "world_size", 1)
    try:
        reports = engine.model_executor.collective_rpc(worker_backend_readiness)
        validate_worker_readiness(plan, reports, expected_workers=expected)
    except BaseException:
        try:
            engine.model_executor.shutdown()
        except Exception:
            logger.exception("Failed to shut down executor after backend readiness failure")
        raise
    engine._gr_backend_fingerprint = plan.fingerprint
