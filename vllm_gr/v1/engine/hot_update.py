# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Individual Core control operations; Sidecar owns the update sequence.

Reload is a synchronous utility call on the Core thread and returns after all
Workers finish. Pause uses the upstream drain Future. There is no update manager,
background orchestration, progress polling, or custom Core keepalive loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm_gr.config import GRConfig
from vllm_gr.config.models import runtime_constraint_table
from vllm_gr.constraints.versioned_table import INITIAL_CONSTRAINT_VERSION
from vllm_gr.update_status import ServiceState, ServiceStatus


def _resources(core: Any) -> dict[str, tuple[str | None, str | None]]:
    """Remember only committed versions/paths, not update-process progress."""
    resources = getattr(core, "_gr_serving_resources", None)
    if resources is None:
        model = str(core.vllm_config.model_config.model)
        table = runtime_constraint_table(GRConfig.from_vllm_config(core.vllm_config))
        path = table.path if table and table.enabled else None
        resources = core._gr_serving_resources = {
            "checkpoint": (Path(model).name, model),
            "trie": ((Path(path).parent.name or Path(path).stem), path) if path else (None, None),
        }
    return resources


def _guard(call: Any, *args: Any) -> Any:
    try:
        return call(*args)
    except ValueError as error:
        return {"ok": False, "error": str(error), "error_type": "invalid"}
    except Exception as error:
        return {"ok": False, "error": str(error), "error_type": "reload_failed"}


def _validate_target(path: str, version: str) -> None:
    if not path.strip() or not version.strip() or version == INITIAL_CONSTRAINT_VERSION:
        raise ValueError("Invalid resource path or version")


def service_status(core: Any) -> ServiceStatus:
    resources = _resources(core)
    checkpoint_version, checkpoint_path = resources["checkpoint"]
    trie_version, trie_path = resources["trie"]
    return ServiceStatus(
        service_state=ServiceState.PAUSED if core.is_scheduler_paused() else ServiceState.RUNNING,
        checkpoint_version=checkpoint_version,
        trie_version=trie_version,
        checkpoint_path=checkpoint_path,
        trie_path=trie_path,
        checkpoint_valid=True,  # The control listener owns the native reload latch.
    )


def _pause(core: Any, resource: str) -> Any:
    if resource not in {"checkpoint", "trie"}:
        raise ValueError("Unknown resource_type")
    # Upstream completes this Future after the in-flight requests drain. Queued
    # requests stay queued; upstream clears caches after the device drains.
    result = core.pause_scheduler(mode="wait", clear_cache=True)
    return result if result is not None else {"ok": True}


def pause(core: Any, resource: str = "checkpoint") -> Any:
    return _guard(_pause, core, resource)


def _resume(core: Any) -> dict[str, Any]:
    core.resume_scheduler()
    return {"ok": True}


def resume(core: Any) -> Any:
    return _guard(_resume, core)


def _reload_trie(core: Any, path: str, version: str) -> dict[str, Any]:
    _validate_target(path, version)
    config = GRConfig.from_vllm_config(core.vllm_config)
    if config.beam_max_width is None:
        raise ValueError("Online Trie reload requires beam_search execution")
    table = runtime_constraint_table(config)
    if table is None or not table.enabled:
        raise ValueError("Online Trie reload requires an enabled canonical constraint table")
    if table.backend.value != "cuda":
        raise ValueError("Online Trie reload currently requires the CUDA GR execution path")
    if not core.vllm_config.scheduler_config.async_scheduling:
        raise ValueError("Online Trie reload requires asynchronous GR scheduling")
    resources = _resources(core)
    if resources["trie"][0] == version:
        return {"ok": True, "version": version}
    core.model_executor.collective_rpc("reload_trie", args=(path, version))
    resources["trie"] = (version, path)
    return {"ok": True, "version": version}


def reload_trie(core: Any, path: str, version: str) -> Any:
    return _guard(_reload_trie, core, path, version)


def install_core_hot_update_hooks(core_cls: type) -> None:
    """Expose individual utility methods without changing the Core execution loop."""
    for name, method in {
        "service_status": service_status,
        "pause": pause,
        "resume": resume,
        "reload_trie": reload_trie,
    }.items():
        setattr(core_cls, name, method)
