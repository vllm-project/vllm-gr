# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Check actual GR worker initialization before accepting requests."""

from __future__ import annotations

import logging
from typing import Any

from vllm_gr.config import GRAttentionBackend, GRConfig
from vllm_gr.config.models import runtime_constraint_table
from vllm_gr.config.parser import gr_config_to_mapping
from vllm_gr.platforms.identity import qualified_class_name

logger = logging.getLogger(__name__)


def _attention_by_kind(runner: Any) -> dict[str, list[str]]:
    groups = getattr(runner, "attn_groups", ())
    if not groups:
        return {}
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    selected: dict[str, set[str]] = {}
    for group in groups:
        for attention in group:
            kind = get_kv_cache_spec_kind(attention.kv_cache_spec).value
            selected.setdefault(kind, set()).add(attention.backend.get_name())
    return {kind: sorted(names) for kind, names in selected.items()}


def _runtime_versions() -> dict[str, str | None]:
    from vllm_gr.platforms import current_gr_platform
    from vllm_gr.platforms.compat import installed_version

    distributions = {"vllm", "vllm-gr"}
    distributions.update(
        name
        for name in (
            current_gr_platform.vendor_distribution,
            current_gr_platform.plugin_distribution,
        )
        if name is not None
    )
    return {name: installed_version(name) for name in sorted(distributions)}


def worker_backend_readiness(worker: Any) -> dict[str, Any]:
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    from vllm_gr.platforms import current_gr_platform

    # Callable executor RPCs receive the native wrapper, not its loaded Worker.
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    runner = worker.model_runner
    if not getattr(runner, "_gr_ready", False):
        raise RuntimeError("GR Worker must initialize its GR ModelRunner before serving")
    config = runner.vllm_config
    gr = GRConfig.from_vllm_config(config)
    if getattr(runner, "_gr_loaded_config", gr) != gr:
        raise RuntimeError("GR configuration changed after model initialization")
    backend = getattr(runner, "constraint_backend", None)
    metadata = getattr(getattr(backend, "info", None), "table_metadata", None)
    execution = getattr(runner, "_gr_attention_execution", None)
    return {
        "ready": True,
        "platform": qualified_class_name(type(current_gr_platform)),
        "runner": qualified_class_name(type(runner)),
        "versions": _runtime_versions(),
        "worker_classes": [qualified_class_name(cls) for cls in type(worker).__mro__],
        "gr_config": gr_config_to_mapping(GRConfig.from_vllm_config(config)),
        "max_model_len": config.model_config.max_model_len,
        "constraint_digest": getattr(metadata, "artifact_digest", None),
        "attention_by_kind": _attention_by_kind(runner),
        "attention_execution": getattr(
            execution, "__name__", qualified_class_name(type(execution))
        ),
        "decode_mode": getattr(execution, "DECODE_MODE", None),
        "graph": runner.gr_graph_status(),
    }


def validate_worker_readiness(config: Any, reports: Any, *, expected_workers: int) -> None:
    from vllm_gr.platforms import current_gr_platform

    if not isinstance(reports, (list, tuple)) or len(reports) != expected_workers or not reports:
        raise RuntimeError(f"GR startup requires {expected_workers} worker reports")
    gr = GRConfig.from_vllm_config(config)
    expected_config = gr_config_to_mapping(gr)
    table = runtime_constraint_table(gr)
    expected_versions = _runtime_versions()
    expected_worker = config.parallel_config.worker_cls
    if isinstance(expected_worker, type):
        expected_worker = qualified_class_name(expected_worker)
    first = None
    for index, report in enumerate(reports):
        if not isinstance(report, dict) or report.get("ready") is not True:
            raise RuntimeError(f"GR worker {index} is not ready: {report}")
        if report.get("platform") != qualified_class_name(type(current_gr_platform)):
            raise RuntimeError(f"GR worker {index} selected a different platform")
        if report.get("versions") != expected_versions:
            raise RuntimeError(f"GR worker {index} has different runtime or plugin versions")
        if expected_worker not in report.get("worker_classes", ()):
            raise RuntimeError(f"GR worker {index} did not use the configured Worker class")
        if not isinstance(report.get("runner"), str) or not report["runner"]:
            raise RuntimeError(f"GR worker {index} did not report an initialized ModelRunner")
        if report.get("gr_config") != expected_config:
            raise RuntimeError(f"GR worker {index} received a different GR configuration")
        length = report.get("max_model_len")
        if type(length) is not int or not 0 < length <= config.model_config.max_model_len:
            raise RuntimeError(f"GR worker {index} has an invalid max_model_len")
        if (
            table is not None
            and table.enabled
            and report.get("constraint_digest") != table.artifact_digest
        ):
            raise RuntimeError(f"GR worker {index} loaded a different constraint artifact")
        if gr.attention_backend is GRAttentionBackend.CUSTOM:
            if (
                not report.get("attention_execution")
                or report["attention_execution"] == "builtins.NoneType"
            ):
                raise RuntimeError(f"GR worker {index} has no Beam attention implementation")
            mode = report.get("decode_mode")
            if gr.npu_decode_execution not in (None, "auto") and mode != gr.npu_decode_execution:
                raise RuntimeError(f"GR worker {index} did not select the requested decode mode")
        overrides = getattr(config.attention_config, "backend_per_kind", None) or {}
        for kind, names in report.get("attention_by_kind", {}).items():
            expected = overrides.get(kind)
            if expected is not None and names != [getattr(expected, "name", expected)]:
                raise RuntimeError(
                    f"GR worker {index} ignored the explicit {kind} attention backend"
                )
        graph = report.get("graph")
        if not isinstance(graph, dict) or not {"captured", "fallback"} <= graph.keys():
            raise RuntimeError(f"GR worker {index} did not report graph execution status")
        if not gr.beam_graph_enabled and graph["captured"]:
            raise RuntimeError(f"GR worker {index} captured graphs for an eager configuration")
        # These are actual initialized outcomes, not a second configuration plan.
        if first is not None and report != first:
            raise RuntimeError(
                "GR workers disagree on implementations, versions, constraints or graph execution"
            )
        first = report


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
    config = engine.vllm_config
    gr = GRConfig.from_vllm_config(config)
    if gr.beam_max_width is None and gr.attention_backend is not GRAttentionBackend.CUSTOM:
        return
    parallel = config.parallel_config
    expected = parallel.world_size
    try:
        if expected > 1 and parallel.distributed_executor_backend == "external_launcher":
            reports = _external_launcher_reports(engine.model_executor, expected)
        else:
            reports = engine.model_executor.collective_rpc(worker_backend_readiness)
        validate_worker_readiness(config, reports, expected_workers=expected)
        from vllm_gr.platforms.beam_requests import engine_request_validator

        engine._gr_validate_platform_request = engine_request_validator(config)
    except BaseException:
        try:
            engine.model_executor.shutdown()
        except Exception:
            logger.exception("Failed to shut down executor after GR initialization failure")
        raise
