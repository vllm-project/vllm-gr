# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Device routing and input ownership for native GR dispatches."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy, deepcopy
from typing import Any

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

from vllm_gr.v1.worker.gpu_stage_hooks import CudaStageHooksType


def _stage_execution_service() -> Any:
    """Return the selected backend's GR stage execution service.

    Resolution goes through the provider registry, never through the worker's
    device name. An unselected backend fails here with its own diagnostic, which
    keeps this module free of hardware-specific branches.
    """
    from vllm_gr.platforms.loader import activate_current_backend

    return activate_current_backend().create_stage_execution_service()


def _copy_record(value: Any) -> Any:
    # These native dataclasses store their fields (including GR metadata) in
    # __dict__. Preserve the complete shallow state without copy's reducer.
    if type(value) in (SchedulerOutput, NewRequestData, CachedRequestData):
        local = object.__new__(type(value))
        local.__dict__ = value.__dict__.copy()
        return local
    return copy(value)


def worker_input(scheduled: Any) -> Any:
    """Detach mutable fields retained by native Worker request state.

    Tensors, immutable GR metadata and ordinary cached-step data stay shared.
    Native resumed requests retain their replacement block lists as well.
    """
    local = _copy_record(scheduled)
    local.num_scheduled_tokens = dict(scheduled.num_scheduled_tokens)
    local.finished_req_ids = set(scheduled.finished_req_ids)
    local.scheduled_new_reqs = []
    for request in scheduled.scheduled_new_reqs:
        owned = _copy_record(request)
        owned.block_ids = tuple(list(group) for group in request.block_ids)
        owned.prompt_token_ids = request.prompt_token_ids.copy()
        prefill = getattr(request, "prefill_token_ids", None)
        if prefill is not None:
            owned.prefill_token_ids = prefill.copy()
        # Native SamplingParams retains flat scalar containers and optional
        # nested parameters. Detach both; remaining fields are immutable scalars.
        # clone() is shallow when skip_clone=True. A shared memo preserves
        # aliases between fields and any values in its cached-property dict.
        params = request.sampling_params
        owned.sampling_params = copy(params)
        if params is not None:
            memo = {id(params): owned.sampling_params}
            for name in (
                "stop",
                "stop_token_ids",
                "logprob_token_ids",
                "_all_stop_token_ids",
                "logit_bias",
                "allowed_token_ids",
                "bad_words",
            ):
                value = getattr(params, name)
                if value is not None:
                    key = id(value)
                    if key not in memo:
                        memo[key] = copy(value)
                    setattr(owned.sampling_params, name, memo[key])
            for name in (
                "structured_outputs",
                "extra_args",
                "_bad_words_token_ids",
                "repetition_detection",
            ):
                value = getattr(params, name)
                if value is not None:
                    setattr(owned.sampling_params, name, deepcopy(value, memo))
            owned.sampling_params.__dict__ = deepcopy(params.__dict__, memo)
        local.scheduled_new_reqs.append(owned)
    cached = scheduled.scheduled_cached_reqs
    if cached.resumed_req_ids:
        local.scheduled_cached_reqs = _copy_record(cached)
        local.scheduled_cached_reqs.new_block_ids = [
            tuple(list(group) for group in blocks)
            if req_id in cached.resumed_req_ids and blocks is not None
            else blocks
            for req_id, blocks in zip(cached.req_ids, cached.new_block_ids)
        ]
    return local


def execute_gr_stage(
    worker: Any,
    scheduled: Any,
    original: Callable[..., Any],
    *args: Any,
    hooks_type: CudaStageHooksType | None = None,
    **kwargs: Any,
) -> Any:
    local = worker_input(scheduled)
    # The selected backend owns executor selection. Shared dispatch must not
    # compare the worker device against a device name to pick one.
    service = _stage_execution_service()
    return service.execute(
        worker,
        local,
        original,
        *args,
        hooks_type=hooks_type,
        **kwargs,
    )


def execute_gr_retirement(worker: Any, scheduled: Any) -> Any:
    local = worker_input(scheduled)
    return _stage_execution_service().retire(worker, local)
