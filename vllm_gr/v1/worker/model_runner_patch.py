# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Active-platform ModelRunner Beam patch dispatcher."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from types import ModuleType

from vllm.logger import init_logger

from vllm_gr.v1.worker.model_runner_common import (
    _WORKER_CAPTURE_PATCH_MARKER,
    _WORKER_DETERMINE_PATCH_MARKER,
    is_worker_runner_patched,
    make_patched_execute_model,
)

logger = init_logger(__name__)

_GPU_PATCH_MODULE = "vllm_gr.v1.worker.gpu_model_runner_patch"
_NPU_PATCH_MODULE = "vllm_gr.v1.worker.npu_model_runner_patch"
_CONSTRAINT_LOAD_PATCH_MARKER = "_vllm_gr_constraint_load_patched"
_CONSTRAINT_SHUTDOWN_PATCH_MARKER = "_vllm_gr_constraint_shutdown_patched"


def close_constraint_backend(runner: object) -> None:
    """Detach and idempotently close the runner-owned Backend."""

    backend = getattr(runner, "constraint_backend", None)
    setattr(runner, "constraint_backend", None)
    if backend is not None:
        backend.close()


def _runner_vocab_size(runner: object) -> int:
    model_config = getattr(runner, "model_config", None)
    vocab_size = getattr(model_config, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise RuntimeError(
            f"constraint backend requires a positive model vocab_size, got {vocab_size!r}"
        )
    return vocab_size


def _runner_max_rows(runner: object, gr_config: object) -> int:
    scheduler_config = getattr(getattr(runner, "vllm_config", None), "scheduler_config", None)
    max_rows = getattr(scheduler_config, "max_num_seqs", None)
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
        max_rows = getattr(gr_config, "beam_max_width", None)
    return max_rows if isinstance(max_rows, int) and max_rows > 0 else 1


def make_load_model_with_constraint_backend(original):
    """Load weights first, then atomically publish a fully checked Backend."""

    @wraps(original)
    def load_model_with_constraint_backend(self, *args, **kwargs):
        from vllm_gr.config import GRConfig
        from vllm_gr.constraints.loader import load_constraint_backend

        close_constraint_backend(self)
        result = original(self, *args, **kwargs)
        backend = None
        try:
            gr_config = GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
            model_config = getattr(self, "model_config", None)
            dtype = getattr(model_config, "dtype", None)
            if gr_config.constraint_table.enabled and dtype is None:
                raise RuntimeError("constraint backend requires model_config.dtype")
            if gr_config.constraint_table.enabled:
                backend = load_constraint_backend(
                    config=gr_config.constraint_table,
                    device=getattr(self, "device", None),
                    vocab_size=_runner_vocab_size(self),
                    dtype=dtype,
                    max_rows=_runner_max_rows(self, gr_config),
                )
            if backend is not None:
                model_memory_usage = int(getattr(self, "model_memory_usage", 0))
                self.model_memory_usage = (
                    model_memory_usage + backend.info.device_memory_bytes
                )
            # Publication is the atomic boundary: no request can observe a
            # partially loaded or readiness-untested Backend.
            self.constraint_backend = backend
            return result
        except Exception:
            if backend is not None:
                backend.close()
            self.constraint_backend = None
            raise

    setattr(load_model_with_constraint_backend, _CONSTRAINT_LOAD_PATCH_MARKER, True)
    return load_model_with_constraint_backend


def make_shutdown_with_constraint_backend(original):
    @wraps(original)
    def shutdown_with_constraint_backend(self, *args, **kwargs):
        close_constraint_backend(self)
        return original(self, *args, **kwargs)

    setattr(shutdown_with_constraint_backend, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, True)
    return shutdown_with_constraint_backend


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
        load_model = getattr(runner_class, "load_model", None)
        if callable(load_model) and not getattr(load_model, _CONSTRAINT_LOAD_PATCH_MARKER, False):
            runner_class.load_model = make_load_model_with_constraint_backend(load_model)
        shutdown = getattr(runner_class, "shutdown", None)
        if callable(shutdown) and not getattr(
            shutdown,
            _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
            False,
        ):
            runner_class.shutdown = make_shutdown_with_constraint_backend(shutdown)
        runner_class.execute_model = make_patched_execute_model(runner_class.execute_model)
        runner_class._prepare_inputs = platform_module.make_patched_prepare_inputs(
            runner_class._prepare_inputs
        )
        runner_class._bookkeeping_sync = platform_module.make_patched_bookkeeping_sync(
            runner_class._bookkeeping_sync
        )
        # The sampler hook consumes only the already-loaded runner Backend.
        if hasattr(runner_class, "_sample"):
            runner_class._sample = platform_module.make_patched_sample(runner_class._sample)

        # After native capture_model() finishes, run the startup dummy capture
        # of the beam-decode FULL graph (only when GRConfig enables it —
        # beam_max_width configured and not disabled; no-op otherwise).
        beam_capture = getattr(platform_module, "capture_beam_decode_graph", None)
        if (
            beam_capture is not None
            and hasattr(runner_class, "capture_model")
            and not getattr(runner_class.capture_model, _WORKER_CAPTURE_PATCH_MARKER, False)
        ):
            _orig_capture_model = runner_class.capture_model

            def make_patched_capture_model(_orig, _beam_capture):
                @wraps(_orig)
                def patched_capture_model(self, *args, **kwargs):
                    res = _orig(self, *args, **kwargs)
                    try:
                        _beam_capture(self)
                    except Exception:
                        logger.exception(
                            "vllm-gr: beam decode graph dummy capture failed; "
                            "beam decode will fall back to eager"
                        )
                    return res

                return patched_capture_model

            patched = make_patched_capture_model(_orig_capture_model, beam_capture)
            setattr(patched, _WORKER_CAPTURE_PATCH_MARKER, True)
            runner_class.capture_model = patched

        # Patch _determine_batch_execution_and_padding to force FULL CUDA Graph
        # for beam decode steps (if the platform module provides it).
        determine_patcher = getattr(
            platform_module, "make_patched_determine_batch_execution_and_padding", None
        )
        if determine_patcher is not None and hasattr(
            runner_class, "_determine_batch_execution_and_padding"
        ):
            if not getattr(
                runner_class._determine_batch_execution_and_padding,
                _WORKER_DETERMINE_PATCH_MARKER,
                False,
            ):
                runner_class._determine_batch_execution_and_padding = determine_patcher(
                    runner_class._determine_batch_execution_and_padding
                )

        runner_class._patched_for_beam_pos_ids = True
        logger.info("%s hooks patched successfully.", runner_name)
