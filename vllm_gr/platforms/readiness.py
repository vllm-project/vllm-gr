# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Worker readiness RPC and cross-worker startup agreement."""

from __future__ import annotations

import logging
from typing import Any

from .plan import BackendPlan, PlanData, read_backend_plan

logger = logging.getLogger(__name__)


def worker_backend_readiness(worker: Any) -> dict[str, Any]:
    """Callable RPC: native executors pass their Worker as the first argument."""
    from .binding import bind_backend_services

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("Worker has no model runner for backend plan validation")
    binding = bind_backend_services(runner)
    if binding is None:
        raise RuntimeError("Worker did not receive the engine backend plan")
    return binding.report()


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


def validate_worker_readiness(plan: BackendPlan, reports: Any, *, expected_workers: int) -> None:
    if type(expected_workers) is not int or expected_workers < 1:
        raise ValueError("Backend readiness requires a positive integer worker count")
    if not isinstance(reports, (list, tuple)) or len(reports) != expected_workers or not reports:
        raise RuntimeError(f"Backend readiness requires {expected_workers} worker reports")
    required = plan.required_services
    services = None
    for index, report in enumerate(reports):
        if not isinstance(report, dict):
            raise RuntimeError(f"Backend worker {index} is not ready")
        if report.get("ready") is not True:
            raise RuntimeError(
                f"Backend worker {index} is not ready: {report.get('error', 'unknown error')}"
            )
        if report.get("fingerprint") != plan.fingerprint:
            raise RuntimeError(f"Backend worker {index} plan fingerprint mismatch")
        reported = report.get("services")
        if not isinstance(reported, dict):
            raise RuntimeError(f"Backend worker {index} has no bound services")
        validate_required_services(plan, required, reported, f"Backend worker {index}")
        current = PlanData(reported)
        if services is not None and current != services:
            raise RuntimeError("Workers disagree on bound backend service implementations")
        services = current


def _external_launcher_reports(executor: Any, expected_workers: int) -> list[Any]:
    """Collect the single local worker report from every externally launched rank."""
    from torch import distributed as dist
    from vllm.distributed.parallel_state import get_world_group

    group = get_world_group().cpu_group
    actual = dist.get_world_size(group)
    try:
        if actual != expected_workers:
            raise RuntimeError(
                f"External launcher readiness expected {expected_workers} ranks, got {actual}"
            )
        local = executor.collective_rpc(worker_backend_readiness)
        if not isinstance(local, (list, tuple)) or len(local) != 1:
            raise RuntimeError(
                "External launcher executor must return its single local worker report"
            )
        report = local[0]
    except Exception as exc:
        # Enter the collective on failed ranks as well; every peer must see the failure.
        report = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    reports: list[Any] = [None] * actual
    dist.all_gather_object(reports, report, group=group)
    return reports


def verify_engine_backend_readiness(engine: Any) -> None:
    plan = read_backend_plan(engine.vllm_config)
    if plan is None:
        return
    parallel = getattr(engine.vllm_config, "parallel_config", None)
    # For native multiprocessing each DP EngineCore owns a TP/PP/PCP executor.
    expected = getattr(parallel, "world_size", 1)
    try:
        if (
            expected > 1
            and getattr(parallel, "distributed_executor_backend", None) == "external_launcher"
        ):
            reports = _external_launcher_reports(engine.model_executor, expected)
        else:
            reports = engine.model_executor.collective_rpc(worker_backend_readiness)
        validate_worker_readiness(plan, reports, expected_workers=expected)
    except BaseException:
        try:
            engine.model_executor.shutdown()
        except Exception:
            logger.exception("Failed to shut down executor after backend readiness failure")
        raise
    engine._gr_backend_fingerprint = plan.fingerprint
    # The Core admits request-specific geometry after every worker has reported
    # the same services. Keep this local to the native Core configuration.
    engine.vllm_config._gr_ready_services = dict(reports[0]["services"])
