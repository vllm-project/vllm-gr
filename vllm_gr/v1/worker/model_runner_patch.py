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


def _make_patched_capture_model(_orig_capture_model, beam_capture):
    """Wrap capture_model to chain the startup beam-decode graph capture.

    Runs the native capture_model first, then the beam-decode dummy capture;
    any failure falls back to eager beam decode (the capture function already
    logs + returns False on failure).
    """
    from functools import wraps

    from vllm_gr.v1.worker.model_runner_common import _WORKER_CAPTURE_PATCH_MARKER

    @wraps(_orig_capture_model)
    def patched_capture_model(self, *args, **kwargs):
        result = _orig_capture_model(self, *args, **kwargs)
        try:
            beam_capture(self)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vllm-gr: beam decode graph capture failed (%r); beam decode will run eager",
                exc,
            )
        return result

    setattr(patched_capture_model, _WORKER_CAPTURE_PATCH_MARKER, True)
    return patched_capture_model


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
        if hasattr(runner_class, "_capture_cudagraphs") and hasattr(
            platform_module, "make_patched_capture_cudagraphs"
        ):
            runner_class._capture_cudagraphs = platform_module.make_patched_capture_cudagraphs(
                runner_class._capture_cudagraphs
            )
        if hasattr(runner_class, "_model_forward") and hasattr(
            platform_module, "make_patched_model_forward"
        ):
            runner_class._model_forward = platform_module.make_patched_model_forward(
                runner_class._model_forward
            )
        if hasattr(runner_class, "_determine_batch_execution_and_padding") and hasattr(
            platform_module, "make_patched_determine_batch_execution_and_padding"
        ):
            runner_class._determine_batch_execution_and_padding = (
                platform_module.make_patched_determine_batch_execution_and_padding(
                    runner_class._determine_batch_execution_and_padding
                )
            )
        beam_capture = getattr(platform_module, "capture_beam_decode_graph", None)
        if (
            beam_capture is not None
            and hasattr(runner_class, "capture_model")
            and not getattr(runner_class.capture_model, "_vllm_gr_beam_capture_patch", False)
        ):
            runner_class.capture_model = _make_patched_capture_model(
                runner_class.capture_model, beam_capture
            )
        runner_class._patched_for_beam_pos_ids = True
        logger.info("%s hooks patched successfully.", runner_name)
