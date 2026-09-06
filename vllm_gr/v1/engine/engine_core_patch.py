# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore lifecycle hooks for persistent GR requests."""

from __future__ import annotations

import time
from functools import wraps

from vllm.logger import init_logger

logger = init_logger(__name__)

_GR_STATE_INIT_MARKER = "_vllm_gr_initializes_persistent_state"
_ENGINE_STEP_PATCH_MARKER = "_vllm_gr_persistent_engine_step"
_GR_ABORT_PATCH_MARKER = "_vllm_gr_releases_runner_state_on_abort"


def make_patched_engine_core_init(original_init):
    """Validate the BatchQueue contract after EngineCore construction."""
    if getattr(original_init, _GR_STATE_INIT_MARKER, False):
        return original_init

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from vllm_gr.v1.core.sched.async_scheduler import GRAsyncScheduler

        if (
            isinstance(getattr(self, "scheduler", None), GRAsyncScheduler)
            and int(getattr(self, "batch_queue_size", 1)) != 2
        ):
            raise ValueError(
                "persistent GR requires EngineCore BatchQueue depth == 2; "
                "single-depth and pipeline-parallel queues are not validated"
            )

    setattr(patched_init, _GR_STATE_INIT_MARKER, True)
    return patched_init


def _retire_engine_after_gr_release_failure(engine, error):
    """Fail closed when Worker state release leaves the executor uncertain."""
    engine._gr_lifecycle_error = error

    def reject(*args, **kwargs):
        raise RuntimeError(
            "EngineCore retired after persistent GR release failure; shutdown required"
        ) from error

    for name in ("add_request", "step", "step_with_batch_queue", "step_fn"):
        setattr(engine, name, reject)
    engine.model_executor.collective_rpc = reject


def make_patched_abort_requests(original_abort):
    """Release device-owned GR state after Scheduler abort succeeds."""
    if getattr(original_abort, _GR_ABORT_PATCH_MARKER, False) is True:
        return original_abort

    @wraps(original_abort)
    def patched_abort_requests(self, request_ids):
        lifecycle_error = getattr(self, "_gr_lifecycle_error", None)
        if isinstance(lifecycle_error, BaseException):
            raise RuntimeError(
                "EngineCore retired after persistent GR release failure; shutdown required"
            ) from lifecycle_error
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        requests = getattr(getattr(self, "scheduler", None), "requests", {})
        gr_ids = [
            request_id
            for request_id in ids
            if bool(
                getattr(requests.get(request_id), "gr_persistent_runner_state", False)
            )
        ]
        result = original_abort(self, ids)
        if gr_ids:
            try:
                release_future = self.model_executor.collective_rpc(
                    "release_gr_model_runner_states",
                    args=(gr_ids,),
                    non_block=True,
                )
                release_future.result()
            except BaseException as error:
                _retire_engine_after_gr_release_failure(self, error)
                error.add_note(f"Persistent GR abort release failed for {gr_ids!r}")
                raise
        return result

    setattr(patched_abort_requests, _GR_ABORT_PATCH_MARKER, True)
    return patched_abort_requests


def make_patched_process_engine_step(original_step):
    """Preserve progress accounting for final-only persistent control steps."""
    if getattr(original_step, _ENGINE_STEP_PATCH_MARKER, False):
        return original_step

    def has_persistent_gr_work(engine) -> bool:
        batch_queue = getattr(engine, "batch_queue", None)
        if batch_queue:
            return bool(getattr(batch_queue[-1][1], "gr_outputs", None))
        requests = getattr(getattr(engine, "scheduler", None), "requests", {})
        return any(
            bool(getattr(request, "is_gr_request", False))
            and bool(getattr(request, "gr_persistent_runner_state", False))
            for request in requests.values()
        )

    @wraps(original_step)
    def patched_process_engine_step(self):
        persistent_gr = has_persistent_gr_work(self)
        outputs, model_executed = self.step_fn()
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        self.post_step(model_executed)

        made_progress = model_executed or (persistent_gr and outputs is not None)
        if not made_progress and self.scheduler.has_requests():
            time.sleep(0.001)
        return model_executed

    setattr(patched_process_engine_step, _ENGINE_STEP_PATCH_MARKER, True)
    return patched_process_engine_step


def apply_engine_core_request_patches() -> None:
    """Install only the lifecycle hooks used by persistent GR."""
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)
    EngineCoreProc.__init__ = make_patched_engine_core_init(EngineCoreProc.__init__)
    EngineCore.abort_requests = make_patched_abort_requests(EngineCore.abort_requests)
    EngineCoreProc.abort_requests = make_patched_abort_requests(
        EngineCoreProc.abort_requests
    )
    EngineCoreProc._process_engine_step = make_patched_process_engine_step(
        EngineCoreProc._process_engine_step
    )


def apply_engine_core_child_patches() -> None:
    """Install persistent EngineCore and Worker hooks in this process."""
    from vllm_gr.v1.worker.model_runner_patch import apply_worker_patches

    apply_engine_core_request_patches()
    apply_worker_patches()


_run_engine_core_active = False


def run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    """Apply persistent hooks before starting a spawned EngineCore."""
    global _run_engine_core_active
    if _run_engine_core_active:
        raise RuntimeError("run_engine_core re-entry detected")

    from vllm.v1.engine.core import EngineCoreProc

    original_run = getattr(run_engine_core, "_original_run_engine_core", None)
    if original_run is None or original_run is run_engine_core:
        original_run = EngineCoreProc.run_engine_core
    if original_run is run_engine_core or not callable(original_run):
        raise RuntimeError("run_engine_core recursion detected")

    _run_engine_core_active = True
    try:
        from vllm_gr.patch import initialize_engine_core_beam_patches

        initialize_engine_core_beam_patches()
        original_run(
            *args,
            dp_rank=dp_rank,
            local_dp_rank=local_dp_rank,
            **kwargs,
        )
    finally:
        _run_engine_core_active = False
