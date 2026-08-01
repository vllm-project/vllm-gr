# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Active-platform ModelRunner Beam patch dispatcher."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from vllm.logger import init_logger

from vllm_gr.v1.worker.model_runner_common import (
    is_worker_runner_patched,
    make_patched_execute_model,
)

logger = init_logger(__name__)

_GPU_PATCH_MODULE = "vllm_gr.v1.worker.gpu_model_runner_patch"
_NPU_PATCH_MODULE = "vllm_gr.v1.worker.npu_model_runner_patch"


def _active_platform_module_name() -> str:
    """Return the patch module for the platform selected by vLLM."""
    from vllm.platforms import current_platform

    if str(current_platform.device_type).lower() == "npu":
        return _NPU_PATCH_MODULE
    return _GPU_PATCH_MODULE


def _get_active_platform_module() -> ModuleType:
    return import_module(_active_platform_module_name())


def discover_active_worker_runners() -> tuple[tuple[str, type], ...]:
    """Discover ModelRunner targets for only the active platform."""
    return _get_active_platform_module().discover_worker_runners()


def apply_worker_patches() -> None:
    """Install grouped-Beam hooks on active-platform ModelRunner targets."""
    platform_module = _get_active_platform_module()
    runners_to_patch = [
        (runner_name, runner_class)
        for runner_name, runner_class in platform_module.discover_worker_runners()
        if not is_worker_runner_patched(runner_class)
    ]

    for runner_name, runner_class in runners_to_patch:
        runner_class.execute_model = make_patched_execute_model(runner_class.execute_model)
        runner_class._prepare_inputs = platform_module.make_patched_prepare_inputs(
            runner_class._prepare_inputs
        )
        runner_class._bookkeeping_sync = platform_module.make_patched_bookkeeping_sync(
            runner_class._bookkeeping_sync
        )
        runner_class._patched_for_beam_pos_ids = True
        logger.info("%s hooks patched successfully.", runner_name)
