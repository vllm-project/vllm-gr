# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device routing for the native execute/sample/retirement contract."""

from collections.abc import Callable
from copy import copy
from typing import Any


def worker_input(scheduled: Any) -> Any:
    """Detach mutable native state that a Worker retains across dispatches.

    Inproc executors share Python objects with Scheduler. In particular,
    native CachedRequestState retains and extends the Prefill block lists.
    Immutable GR stage metadata and its scalar geometry need no copying.
    """
    result = copy(scheduled)
    result.num_scheduled_tokens = dict(scheduled.num_scheduled_tokens)
    if hasattr(scheduled, "finished_req_ids"):
        result.finished_req_ids = set(scheduled.finished_req_ids)
    if hasattr(scheduled, "scheduled_new_reqs"):
        result.scheduled_new_reqs = []
        for native in scheduled.scheduled_new_reqs:
            local = copy(native)
            local.block_ids = tuple(list(group) for group in native.block_ids)
            local.prompt_token_ids = list(native.prompt_token_ids)
            local.sampling_params = copy(native.sampling_params)
            result.scheduled_new_reqs.append(local)
    return result


def get_gr_stage_runner(worker: Any, scheduled: Any) -> Any:
    """Return the installed device adapter; unsupported devices fail explicitly."""
    if str(worker.device).split(":")[0] != "cuda":
        raise ValueError("beam_search_v1 has no execution adapter for this device")
    from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner

    return get_stage_runner(worker, scheduled)


def execute_gr_stage(
    worker: Any, scheduled: Any, original: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Route one dispatch through the device adapter's validation and execution."""
    if str(worker.device).split(":")[0] != "cuda":
        raise ValueError("beam_search_v1 has no execution adapter for this device")
    from vllm_gr.v1.worker.gpu_beam_stage_runner import execute_gr_stage as execute

    return execute(worker, scheduled, original, *args, **kwargs)


def execute_gr_retirement(worker: Any, scheduled: Any) -> Any:
    if str(worker.device).split(":")[0] != "cuda":
        raise ValueError("beam_search_v1 has no retirement adapter for this device")
    from vllm_gr.v1.worker.gpu_beam_stage_runner import execute_gr_retirement as retire

    return retire(worker, worker_input(scheduled))
