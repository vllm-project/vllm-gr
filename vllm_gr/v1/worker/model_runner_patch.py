# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Active-platform ModelRunner Beam patch dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from vllm.logger import init_logger

from vllm_gr.platforms.protocol import ModelRunnerPatchHooks
from vllm_gr.v1.worker.model_runner_common import (
    is_worker_runner_patched as _is_worker_runner_patched,
)
from vllm_gr.v1.worker.model_runner_common import (
    make_patched_execute_model,
)

logger = init_logger(__name__)


def is_worker_runner_patched(runner_class: type[Any]) -> bool:
    """Return whether all shared and platform wrappers are installed."""
    checker = cast(Callable[[type[Any]], bool], _is_worker_runner_patched)
    return checker(runner_class)


def _get_active_model_runner_hooks() -> ModelRunnerPatchHooks:
    """Return ModelRunner hooks from the process-local active provider."""
    from vllm_gr.platforms import get_active_backend

    return get_active_backend().get_model_runner_hooks()


def discover_active_worker_runners() -> tuple[tuple[str, type[Any]], ...]:
    """Discover ModelRunner targets for only the active platform."""
    return _get_active_model_runner_hooks().discover_worker_runners()


def apply_worker_patches() -> None:
    """Install grouped-Beam hooks on active-platform ModelRunner targets."""
    from vllm_gr.platforms import get_active_backend

    backend = get_active_backend()
    hooks = backend.get_model_runner_hooks()
    runners_to_patch = [
        (runner_name, runner_class)
        for runner_name, runner_class in hooks.discover_worker_runners()
        if not is_worker_runner_patched(runner_class)
    ]

    for runner_name, runner_class in runners_to_patch:
        execute_wrapper_factory = cast(
            Callable[[Callable[..., Any]], Callable[..., Any]],
            make_patched_execute_model,
        )
        runner_class.execute_model = execute_wrapper_factory(runner_class.execute_model)
        runner_class._prepare_inputs = hooks.make_patched_prepare_inputs(
            runner_class._prepare_inputs
        )
        runner_class._bookkeeping_sync = hooks.make_patched_bookkeeping_sync(
            runner_class._bookkeeping_sync
        )
        runner_class._patched_for_beam_pos_ids = True
        logger.info(
            "%s hooks patched successfully for backend %s.",
            runner_name,
            backend.device_type,
        )
