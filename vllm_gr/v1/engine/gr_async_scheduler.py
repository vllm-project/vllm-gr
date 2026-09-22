# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam Search V1 state layered on vLLM's native asynchronous scheduler."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchRequest,
    BeamBatchResult,
)
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRExecutionCapacity,
    GRResourceStatus,
    GRRetireMetadata,
    GRRetireResult,
    GRSessionState,
    GRSessionStatus,
    GRStageCompletion,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
    validate_gr_execution_config,
)

GR_STAGE_OUTPUT_ATTR = "gr_stage_metadata"
GR_WORKER_RESULT_ATTR = "gr_worker_results"
GR_BATCH_RESULT_ATTR = "gr_batch_results"
GR_LOOKAHEAD_DEPTH = 2
GR_RETIRE_OUTPUT_ATTR = "gr_retire_metadata"
GR_RETIRE_RESULT_ATTR = "gr_retire_result"
_DRAINING_STATUSES = frozenset(
    {
        GRSessionStatus.FINISHING,
        GRSessionStatus.ABORTED,
        GRSessionStatus.FAILED,
    }
)


class _GRWorkerProtocolError(ValueError):
    """A malformed per-request Worker result; never escapes EngineCore."""


class GRAdmissionError(ValueError):
    """An expected request rejection before Scheduler/Worker registration."""


def _sessions(scheduler: Any) -> dict[str, GRSessionState]:
    sessions = getattr(scheduler, "_gr_v1_sessions", None)
    if sessions is None:
        sessions = {}
        scheduler._gr_v1_sessions = sessions
    return sessions


def has_gr_cleanup_work(scheduler: Any) -> bool:
    """Keep terminal delivery and Worker retirement live under native pause."""
    # Retained sessions are not necessarily runnable: native pause policy
    # decides that, and EngineCore's batch queue drains ordinary in-flight
    # dispatches. Only GR cleanup adds work beyond those native checks.
    return any(
        session.terminal_result_pending
        or session.resource_status in (GRResourceStatus.DRAINING, GRResourceStatus.RETIRING)
        for session in getattr(scheduler, "_gr_v1_sessions", {}).values()
    )


def _progress(scheduler: Any) -> None:
    scheduler._gr_progress = getattr(scheduler, "_gr_progress", 0) + 1


def _admit_owner(scheduler: Any) -> GRSessionState | None:
    sessions = _sessions(scheduler)
    owner_id = getattr(scheduler, "_gr_owner", None)
    if owner_id is not None:
        return sessions[owner_id]
    # Dict insertion order is the session FIFO, independent of native priority.
    for session in sessions.values():
        if session.resource_status is GRResourceStatus.WAITING:
            generation = getattr(scheduler, "_gr_owner_generation", 0) + 1
            scheduler._gr_owner_generation = generation
            scheduler._gr_owner = session.session_id
            session.owner_generation = generation
            session.resource_status = GRResourceStatus.RUNNING
            _progress(scheduler)
            return session
    return None


class _AdmissionQueue:
    """Filtered native queue: hidden requests retain their objects and FIFO positions."""

    def __init__(self, queue: Any, owner_id: str):
        self.queue, self.owner_id = queue, owner_id

    def __iter__(self):
        return (r for r in self.queue if r.request_id == self.owner_id)

    def __len__(self):
        return sum(1 for _ in self)

    def __bool__(self):
        return any(True for _ in self)

    def peek_request(self):
        return next(iter(self))

    def pop_request(self):
        request = self.peek_request()
        self.queue.remove_request(request)
        return request

    def __getattr__(self, name):
        return getattr(self.queue, name)


@contextmanager
def gr_admission(scheduler: Any):
    """Gate native selection and reserve the full logical KV extent on first allocation."""
    owner = _admit_owner(scheduler)
    if owner is None:
        yield
        return
    waiting, skipped = scheduler.waiting, scheduler.skipped_waiting
    budget = scheduler.max_num_scheduled_tokens
    manager = scheduler.kv_cache_manager
    allocate = manager.allocate_slots

    def reserve(request, num_new_tokens, *args, **kwargs):
        if request.request_id != owner.session_id:
            raise RuntimeError("Asynchronous scheduling allocation escaped the execution owner")
        computed = request.num_computed_tokens + kwargs.get("num_new_computed_tokens", 0)
        extent = request.num_prompt_tokens + owner.max_decode_steps
        kwargs["num_lookahead_tokens"] = max(0, extent - computed - num_new_tokens)
        result = allocate(request, num_new_tokens, *args, **kwargs)
        if result is None and owner.issued_stages:
            raise RuntimeError(
                "Asynchronous scheduling lost reserved KV capacity after Beam generation"
            )
        return result

    scheduler.waiting = _AdmissionQueue(waiting, owner.session_id)
    scheduler.skipped_waiting = _AdmissionQueue(skipped, owner.session_id)
    manager.allocate_slots = reserve
    if not can_issue_gr_stage(scheduler, owner):
        scheduler.max_num_scheduled_tokens = 0
    try:
        yield
    finally:
        scheduler.waiting, scheduler.skipped_waiting = waiting, skipped
        scheduler.max_num_scheduled_tokens = budget
        manager.allocate_slots = allocate


def _drain_resources(scheduler: Any, session: GRSessionState) -> None:
    session.resource_status = GRResourceStatus.DRAINING
    if not session.worker_resources_allocated:
        _release_owner(scheduler, session)


def _release_owner(scheduler: Any, session: GRSessionState) -> None:
    session.resource_status = GRResourceStatus.CLOSED
    _sessions(scheduler).pop(session.session_id, None)
    if getattr(scheduler, "_gr_owner", None) == session.session_id:
        scheduler._gr_owner = None
    _progress(scheduler)


def attach_gr_retirement(scheduler: Any, output: Any) -> None:
    owner = _sessions(scheduler).get(getattr(scheduler, "_gr_owner", None))
    if (
        owner is None
        or owner.resource_status is not GRResourceStatus.DRAINING
        or owner._in_flight_dispatches
        or owner.terminal_result_pending
    ):
        return
    if output.total_num_scheduled_tokens:
        raise RuntimeError("Asynchronous scheduling retirement must be a zero-token control frame")
    retire_id = getattr(scheduler, "_gr_retire_id", 0) + 1
    scheduler._gr_retire_id = retire_id
    owner.retirement = GRRetireMetadata(owner.session_id, owner.owner_generation, retire_id)
    owner.resource_status = GRResourceStatus.RETIRING
    setattr(output, GR_RETIRE_OUTPUT_ATTR, owner.retirement)
    output.finished_req_ids.add(owner.session_id)
    _progress(scheduler)


def consume_gr_retirement(scheduler: Any, output: Any, model_output: Any) -> None:
    metadata = getattr(output, GR_RETIRE_OUTPUT_ATTR, None)
    if metadata is None:
        return
    session = _sessions(scheduler).get(metadata.session_id)
    if session is None or session.retirement != metadata:
        return  # A duplicate/stale receipt cannot release the current owner.
    result = getattr(model_output, GR_RETIRE_RESULT_ATTR, None)
    expected = GRRetireResult(
        metadata.session_id, metadata.owner_generation, metadata.retire_id, True
    )
    if (
        not isinstance(result, GRRetireResult)
        or type(result.owner_generation) is not int
        or type(result.retire_id) is not int
        or result.released is not True
        or result != expected
    ):
        # A missing/mismatched release proof is a Worker protocol failure.
        # Never continue service by guessing that the shared buffers are free.
        raise RuntimeError(
            "Asynchronous scheduling Worker did not prove release of the matching execution owner"
        )
    _release_owner(scheduler, session)


def _install_gr_prompt_cache_guard(scheduler: Any) -> None:
    """Cap native async output caching as well as schedule-time caching."""
    manager = getattr(scheduler, "kv_cache_manager", None)
    original = getattr(manager, "cache_blocks", None)
    if (
        manager is None
        or not callable(original)
        or getattr(manager, "_gr_prompt_cache_guard", False)
    ):
        return

    def cache_prompt(request: Any, num_computed_tokens: int) -> Any:
        if getattr(request, "gr_batch_request", None) is not None:
            num_computed_tokens = min(num_computed_tokens, request.num_prompt_tokens)
        return original(request, num_computed_tokens)

    manager.cache_blocks = cache_prompt
    manager._gr_prompt_cache_guard = True


def _prepare_gr_session(scheduler: Any, request: BeamBatchRequest) -> GRSessionState:
    """Validate admission without mutating session, owner or capacity state."""
    if not scheduler.scheduler_config.async_scheduling:
        raise RuntimeError("beam_search_v1 requires vLLM async_scheduling")
    if len(request.items) != 1:
        raise GRAdmissionError("Beam Search V1 currently requires batch size B=1")
    session_id = request.batch_id
    if not session_id:
        raise GRAdmissionError("session_id must not be empty")
    engine_request = request.items[0].request
    if engine_request.request_id != session_id:
        raise GRAdmissionError("batch_id and the EngineCore request_id must be identical")
    sessions = getattr(scheduler, "_gr_v1_sessions", {})
    if session_id in sessions:
        raise GRAdmissionError(f"GR session already exists: {session_id}")
    config = getattr(scheduler, "vllm_config", None)
    if config is not None:
        validate_gr_execution_config(config)
    capacity = getattr(scheduler, "_gr_execution_capacity", None)
    if capacity is None:
        from vllm_gr.config import GRConfig

        width = GRConfig.from_vllm_config(config).beam_max_width if config is not None else None
        if config is not None and width is None:
            raise ValueError(
                "Asynchronous scheduling requires beam_max_width to be fixed at startup"
            )
        capacity = GRExecutionCapacity(width or request.beam_params.beam_width)
    try:
        capacity.validate(request.beam_params)
    except ValueError as error:
        raise GRAdmissionError(str(error)) from error
    manager = getattr(scheduler, "kv_cache_manager", None)
    if manager is not None:
        pool = manager.block_pool
        block_size = scheduler.cache_config.block_size
        extent = len(engine_request.prompt_token_ids) + request.beam_params.max_tokens - 1
        if (
            extent >= manager.max_model_len
            or (extent + block_size - 1) // block_size > pool.num_gpu_blocks - 1
        ):
            raise GRAdmissionError(
                "Asynchronous scheduling request exceeds the total native KV capacity"
            )
    return GRSessionState(
        session_id=session_id,
        request=request,
        # The final Prefill invocation produces output token 0; the remaining
        # max_tokens - 1 invocations are Decode stages.
        max_decode_steps=max(0, request.beam_params.max_tokens - 1),
    )


def _register_gr_session(scheduler: Any, session: GRSessionState) -> None:
    """Publish validated state after the native ADD succeeds on the core thread."""
    _install_gr_prompt_cache_guard(scheduler)
    if getattr(scheduler, "_gr_execution_capacity", None) is None:
        scheduler._gr_execution_capacity = GRExecutionCapacity(
            session.request.beam_params.beam_width
        )
    _sessions(scheduler)[session.session_id] = session
    _admit_owner(scheduler)


def add_gr_session(scheduler: Any, request: BeamBatchRequest) -> GRSessionState:
    """Register one validated B=1 session for direct Scheduler callers."""
    session = _prepare_gr_session(scheduler, request)
    _register_gr_session(scheduler, session)
    return session


def can_issue_gr_stage(scheduler: Any, session: GRSessionState) -> bool:
    return (
        getattr(scheduler, "_gr_owner", None) == session.session_id
        and session.resource_status is GRResourceStatus.RUNNING
        and session.status in (GRSessionStatus.WAITING, GRSessionStatus.RUNNING)
        and session.next_stage_kind is not None
        and len(session._in_flight_dispatches) < GR_LOOKAHEAD_DEPTH
    )


def reserve_gr_output_placeholder(scheduler: Any, session: GRSessionState, num_tokens: int) -> None:
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    session.output_placeholders += num_tokens
    session.pending_device_token_count += num_tokens


def attach_gr_stage_metadata(
    scheduler: Any, scheduler_output: Any, session: GRSessionState
) -> GRStageMetadata | None:
    req_id = session.session_id
    if req_id not in scheduler_output.num_scheduled_tokens:
        return None
    native_request = scheduler.requests.get(req_id)
    if native_request is None:
        return None
    if not can_issue_gr_stage(scheduler, session):
        raise RuntimeError("native Scheduler selected a blocked Asynchronous scheduling dispatch")

    # Initial chunked Prefill and post-preemption recomputation do not produce
    # a token. Physical dispatch accounting still tracks them, but logical
    # stage, placeholder, and device-state ownership must not advance.
    prefill_chunk = bool(native_request.is_prefill_chunk)
    if prefill_chunk and session.issued_stages:
        raise RuntimeError(
            "Asynchronous scheduling cannot recompute Prefill after Beam initialization"
        )
    dispatch_id = session._next_dispatch_id
    stage_index = session.issued_stages
    kind = GRStageKind.PREFILL if prefill_chunk or stage_index == 0 else GRStageKind.DECODE
    output_state = GRDeviceStateRef(req_id, stage_index)
    manager = getattr(scheduler, "kv_cache_manager", None)
    kv_blocks: tuple[tuple[int, ...], ...] | None = None
    if manager is not None:
        # Scheduler owns the native block refcounts. Python references in a
        # Worker cannot prevent block reuse after abort or preemption.
        kv = manager.get_blocks(req_id)
        groups = tuple(tuple(group) for group in kv.blocks)
        blocks = tuple(block for group in groups for block in group)
        manager.block_pool.touch(blocks)
        # The last reference determines eviction order. Match native free():
        # preserve group order and release each group's prompt tail first.
        session._kv_dispatch_leases[dispatch_id] = tuple(
            block for group in groups for block in reversed(group)
        )
        # Publish the per-group block ids so a runner without host-side
        # request state (V2) can build its beam attention block table without
        # reaching back into scheduler-owned structures.  ``blocks`` holds
        # KVCacheBlock objects; only the integer ids cross the boundary.
        # Tuple-of-tuples so the frozen metadata stays hashable.  The
        # attribute fallback covers stand-in KVCacheBlocks in tests.
        block_ids = getattr(kv, "get_block_ids", None)
        if callable(block_ids):
            kv_blocks = tuple(tuple(group) for group in block_ids())
        else:
            kv_blocks = tuple(
                tuple(
                    block if isinstance(block, int) else getattr(block, "block_id", block)
                    for block in group
                )
                for group in groups
            )
    metadata = GRStageMetadata(
        session_id=req_id,
        dispatch_id=dispatch_id,
        stage_index=stage_index,
        kind=kind,
        decode_step=None if kind == GRStageKind.PREFILL else stage_index - 1,
        is_last_stage=not prefill_chunk and stage_index == session.total_stages - 1,
        input_device_state=session.latest_device_state,
        output_device_state=output_state,
        beam_params=session.request.beam_params,
        output_options=session.request.output_options,
        num_prompt_tokens=int(native_request.num_prompt_tokens),
        produces_output=not prefill_chunk,
        owner_generation=session.owner_generation,
        execution_capacity=scheduler._gr_execution_capacity,
        kv_blocks=kv_blocks,
    )
    session._next_dispatch_id += 1
    session._in_flight_dispatches.add(dispatch_id)
    session.worker_resources_allocated = True
    _progress(scheduler)
    if not prefill_chunk:
        session.issued_stages += 1
        session.status = GRSessionStatus.RUNNING
        session.latest_device_state = output_state
        native_placeholders = int(getattr(native_request, "num_output_placeholders", 0))
        session.output_placeholders = native_placeholders
        session.pending_device_token_count = native_placeholders
    return metadata


def _stage_completion(
    session: GRSessionState,
    stage_index: int,
    model_output: Any,
    *,
    finished: bool = False,
    device_state: GRDeviceStateRef | None = None,
    error: str | None = None,
) -> GRStageCompletion:
    return GRStageCompletion(
        session_id=session.session_id,
        stage_index=stage_index,
        finished=finished,
        confirmed_token_count=session.confirmed_token_count,
        device_state=device_state,
        model_output=model_output,
        error=error,
    )


def _result_mapping(model_output: Any, attribute: str) -> Mapping[Any, Any] | None:
    value = getattr(model_output, attribute, None)
    if value is not None and not isinstance(value, Mapping):
        raise _GRWorkerProtocolError(f"Worker returned an invalid {attribute} mapping")
    return value


def _read_worker_payload(
    session: GRSessionState,
    metadata: GRStageMetadata,
    model_output: Any,
) -> tuple[GRWorkerResult, BeamBatchResult | None, int, bool]:
    """Require a Worker receipt with matching identity for every active dispatch."""
    worker_results = _result_mapping(model_output, GR_WORKER_RESULT_ATTR)
    worker_result = worker_results.get(session.session_id) if worker_results is not None else None
    if worker_result is None:
        raise _GRWorkerProtocolError("Worker omitted GRWorkerResult for a dispatched GR stage")
    if not isinstance(worker_result, GRWorkerResult):
        raise _GRWorkerProtocolError("Worker returned an invalid GRWorkerResult")
    if (
        worker_result.session_id != session.session_id
        or worker_result.dispatch_id != metadata.dispatch_id
        or worker_result.stage_index != metadata.stage_index
    ):
        raise _GRWorkerProtocolError("Worker returned a GR result for a different dispatch")
    if type(worker_result.resources_released) is not bool:
        raise _GRWorkerProtocolError("Worker returned an invalid resource-release proof")

    terminal_result = worker_result.terminal_result
    if terminal_result is None:
        terminal_results = _result_mapping(model_output, GR_BATCH_RESULT_ATTR)
        if terminal_results is not None:
            terminal_result = terminal_results.get(session.session_id)
    if terminal_result is not None:
        if not isinstance(terminal_result, BeamBatchResult):
            raise _GRWorkerProtocolError("Worker returned an invalid BeamBatchResult")
        if terminal_result.batch_id != session.session_id:
            raise _GRWorkerProtocolError("Worker returned a terminal result for another session")

    produced = worker_result.produced_token_count
    worker_finished = bool(worker_result.finished)
    if not isinstance(produced, int) or isinstance(produced, bool) or produced not in (0, 1):
        raise _GRWorkerProtocolError("Worker returned an invalid produced_token_count")
    if (
        metadata.produces_output
        and produced == 0
        and not worker_finished
        and terminal_result is None
    ):
        raise _GRWorkerProtocolError("Worker returned no token for a non-terminal logical stage")
    if not metadata.produces_output and produced != 0:
        raise _GRWorkerProtocolError("Worker produced a token for a non-output Prefill chunk")
    if produced == 1 and worker_result.device_state != metadata.output_device_state:
        raise _GRWorkerProtocolError("Worker returned a device state for a different logical stage")
    return worker_result, terminal_result, produced, worker_finished


def _sync_output_placeholders(scheduler: Any, session: GRSessionState) -> None:
    native_request = scheduler.requests.get(session.session_id)
    session.output_placeholders = (
        int(native_request.num_output_placeholders)
        if native_request is not None
        else session.pending_device_token_count
    )


def update_gr_from_output(
    scheduler: Any,
    session: GRSessionState,
    scheduler_output: Any,
    model_output: Any,
) -> GRStageCompletion:
    metadata = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {}).get(session.session_id)
    if metadata is None:
        return _stage_completion(session, session.completed_stages, model_output)

    # Completion is keyed by the physical dispatch rather than logical stage.
    # Initial Prefill chunks can share a logical stage, while each dispatched
    # SchedulerOutput must still be consumed exactly once.
    if (
        metadata.owner_generation != session.owner_generation
        or metadata.dispatch_id not in session._in_flight_dispatches
    ):
        return _stage_completion(session, metadata.stage_index, model_output)
    session._in_flight_dispatches.remove(metadata.dispatch_id)
    _progress(scheduler)
    blocks = session._kv_dispatch_leases.pop(metadata.dispatch_id, None)
    if blocks is not None:
        scheduler.kv_cache_manager.block_pool.free_blocks(blocks)

    # Once termination starts, successors that were already queued are only
    # drained. They must not advance logical tokens or replace the result that
    # selected the terminal path.
    if session.status in _DRAINING_STATUSES:
        try:
            worker_results = _result_mapping(model_output, GR_WORKER_RESULT_ATTR)
            worker_result = (
                worker_results.get(session.session_id) if worker_results is not None else None
            )
            if worker_result is not None:
                if not isinstance(worker_result, GRWorkerResult):
                    raise _GRWorkerProtocolError("Worker returned an invalid GRWorkerResult")
                if (
                    worker_result.session_id != session.session_id
                    or worker_result.dispatch_id != metadata.dispatch_id
                    or worker_result.stage_index != metadata.stage_index
                ):
                    raise _GRWorkerProtocolError(
                        "Worker returned a GR result for a different dispatch"
                    )
                if type(worker_result.resources_released) is not bool:
                    raise _GRWorkerProtocolError(
                        "Worker returned an invalid resource-release proof"
                    )
                if worker_result.resources_released:
                    session.worker_resources_allocated = False
        except _GRWorkerProtocolError as error:
            return _fail_gr_dispatch(scheduler, session, metadata, model_output, str(error))
        if metadata.produces_output:
            # The logical stage was issued before termination and is now
            # retired, but its speculative output must not be confirmed.
            session.completed_stages += 1
        if not session._in_flight_dispatches and (
            session.status in (GRSessionStatus.ABORTED, GRSessionStatus.FAILED)
            or not session.terminal_result_pending
        ):
            _drain_resources(scheduler, session)
        return _stage_completion(session, metadata.stage_index, model_output)

    try:
        worker_result, terminal_result, produced, worker_finished = _read_worker_payload(
            session, metadata, model_output
        )
    except _GRWorkerProtocolError as error:
        return _fail_gr_dispatch(scheduler, session, metadata, model_output, str(error))

    if worker_result.resources_released:
        if not worker_finished:
            return _fail_gr_dispatch(
                scheduler,
                session,
                metadata,
                model_output,
                "Worker released GR resources before a terminal receipt",
            )
        session.worker_resources_allocated = False

    if not metadata.produces_output:
        if worker_finished or terminal_result is not None:
            session.status = GRSessionStatus.FINISHING
            session.terminal_result_pending = True
            session._pending_terminal_result = terminal_result
            session.resource_status = GRResourceStatus.DRAINING
        return _stage_completion(
            session, metadata.stage_index, model_output, finished=worker_finished
        )

    session.completed_stages += 1
    session.confirmed_token_count += produced
    session.pending_device_token_count = max(0, session.pending_device_token_count - produced)
    _sync_output_placeholders(scheduler, session)
    finished = worker_finished or terminal_result is not None
    if finished or session.completed_stages >= session.total_stages:
        session.status = GRSessionStatus.FINISHING
        session.terminal_result_pending = True
        session._pending_terminal_result = terminal_result
        session.resource_status = GRResourceStatus.DRAINING
    return _stage_completion(
        session,
        metadata.stage_index,
        model_output,
        finished=finished,
        device_state=worker_result.device_state,
    )


def _make_failure_result(session: GRSessionState, code: str, message: str) -> BeamBatchResult:
    return BeamBatchResult(
        protocol_version=session.request.protocol_version,
        batch_id=session.session_id,
        output=None,
        failure=BeamBatchFailure(code=code, message=message, item_index=None),
    )


def _fail_gr_dispatch(
    scheduler: Any,
    session: GRSessionState,
    metadata: GRStageMetadata,
    model_output: Any,
    message: str,
) -> GRStageCompletion:
    """Turn a malformed Worker response into one request-scoped failure."""
    if metadata.produces_output:
        session.completed_stages += 1
    session.status = GRSessionStatus.FINISHING
    session.resource_status = GRResourceStatus.DRAINING
    session.terminal_result_pending = True
    if session._pending_terminal_result is None:
        session._pending_terminal_result = _make_failure_result(
            session, "worker_protocol_error", message
        )
    _sync_output_placeholders(scheduler, session)
    return _stage_completion(
        session,
        metadata.stage_index,
        model_output,
        finished=True,
        error=message,
    )


def _gr_session_ready_to_finalize(session: GRSessionState) -> bool:
    return (
        session.status == GRSessionStatus.FINISHING
        and session.terminal_result_pending
        and not session._terminal_emitted
        and not session._in_flight_dispatches
    )


def _gr_terminal_result(session: GRSessionState) -> BeamBatchResult:
    result = session._pending_terminal_result
    if result is not None:
        return result
    return _make_failure_result(
        session,
        "worker_result_unavailable",
        "Worker did not return a terminal BeamBatchResult",
    )


def _close_gr_session(scheduler: Any, session: GRSessionState) -> None:
    if session._terminal_emitted:
        return
    session._terminal_emitted = True
    session.terminal_result_pending = False
    session.output_placeholders = 0
    session.pending_device_token_count = 0
    session.status = GRSessionStatus.FINISHED
    _drain_resources(scheduler, session)


def finish_gr_session(
    scheduler: Any, session: GRSessionState, status: GRSessionStatus
) -> BeamBatchResult | None:
    session.status = status
    _drain_resources(scheduler, session)
    session.terminal_result_pending = False
    if status == GRSessionStatus.FAILED:
        return _make_failure_result(session, "engine_failure", "GR session failed in EngineCore")
    return None


def abort_gr_session(scheduler: Any, session_id: str) -> None:
    from vllm.v1.request import RequestStatus

    session = _sessions(scheduler).get(session_id)
    if session is not None:
        session.status = GRSessionStatus.ABORTED
        session.terminal_result_pending = False
    scheduler.finish_requests(session_id, RequestStatus.FINISHED_ABORTED)
    if session is not None and not session._in_flight_dispatches:
        _drain_resources(scheduler, session)


def release_gr_session(scheduler: Any, session_id: str) -> None:
    session = _sessions(scheduler).get(session_id)
    if session is None:
        return
    if session._in_flight_dispatches:
        raise RuntimeError("cannot release a GR session with in-flight stages")
    if session.worker_resources_allocated:
        raise RuntimeError(
            "Asynchronous scheduling release requires a matching Worker retirement proof"
        )
    _release_owner(scheduler, session)


def install_gr_scheduler_methods(scheduler_cls: type[Any]) -> None:
    scheduler_cls.add_gr_session = add_gr_session
    scheduler_cls.can_issue_gr_stage = can_issue_gr_stage
    scheduler_cls.reserve_gr_output_placeholder = reserve_gr_output_placeholder
    scheduler_cls.attach_gr_stage_metadata = attach_gr_stage_metadata
    scheduler_cls.update_gr_from_output = update_gr_from_output
    scheduler_cls.finish_gr_session = finish_gr_session
    scheduler_cls.abort_gr_session = abort_gr_session
    scheduler_cls.release_gr_session = release_gr_session
