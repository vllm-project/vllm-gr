# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Persistent Beam lifecycle and vLLM private-API compatibility boundary."""

from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PersistentBeamStage(str, Enum):
    PREFILLING = "prefilling"
    PARKED = "parked"
    BEAM_DECODING = "beam_decoding"
    FINALIZING = "finalizing"
    DRAINING = "draining"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(slots=True)
class PersistentBeamState:
    session_id: str
    stage: PersistentBeamStage
    prefix_len: int
    shared_prefix_tokens: tuple[int, ...]
    prefix_block_ids: tuple[tuple[int, ...], ...] = ()
    logical_step: int = -1
    next_epoch: int = 0
    in_flight_epoch: int | None = None
    completed_epoch: int | None = None
    scheduled_width: int = 1
    active_width: int = 1
    finalization_started: bool = False
    scheduler_freed: bool = False
    worker_teardown_sent: bool = False
    final_result_sent: bool = False
    original_arrival_time: float = 0.0
    request_identity: int = 0
    prefix_block_hashes: tuple[Any, ...] = ()
    execution_block_counts: tuple[int, ...] = ()


def persistent_beam_state(request) -> PersistentBeamState | None:
    state = getattr(request, "gr_persistent_beam_state", None)
    return state if isinstance(state, PersistentBeamState) else None


def initialize_persistent_engine_state(engine) -> None:
    if not hasattr(engine, "persistent_beam_states"):
        engine.persistent_beam_states = {}
    if not hasattr(engine, "persistent_active_session_id"):
        engine.persistent_active_session_id = None
    if not hasattr(engine, "persistent_pending_starts"):
        engine.persistent_pending_starts = deque()
    if not hasattr(engine, "persistent_pending_session_ids"):
        engine.persistent_pending_session_ids = set()


def defer_persistent_start(engine, start, request) -> bool:
    """Queue a Start while the single Worker Beam slot is active/draining."""
    initialize_persistent_engine_state(engine)
    session_id = start.controller.beam_session_id
    if session_id in engine.persistent_pending_session_ids:
        raise ValueError(f"duplicate pending beam_session_id {session_id!r}")
    if engine.persistent_active_session_id is None:
        return False
    engine.persistent_pending_starts.append((start, request))
    engine.persistent_pending_session_ids.add(session_id)
    return True


def admit_persistent_request(engine, request, controller) -> PersistentBeamState:
    """Admit exactly one resumable Request and establish its sidecar state."""
    from vllm_gr.config import BeamControlMode, GRConfig

    initialize_persistent_engine_state(engine)
    session_id = controller.beam_session_id
    if engine.persistent_active_session_id is not None:
        raise RuntimeError("persistent Beam admission requires an idle Worker slot")
    if GRConfig.from_vllm_config(engine.vllm_config).beam_control_mode is not (
        BeamControlMode.ENGINE_PERSISTENT_REQUEST
    ):
        raise RuntimeError("persistent Beam admission requires engine_persistent_request mode")
    scheduler_config = getattr(engine.vllm_config, "scheduler_config", None)
    if bool(getattr(scheduler_config, "async_scheduling", False)):
        raise NotImplementedError(
            "engine_persistent_request does not support async_scheduling=True"
        )
    if request.max_tokens != 1:
        raise ValueError("persistent Beam admission requires max_tokens=1")
    prefix_tokens = tuple(request.prompt_token_ids or ())
    prefix_len = len(controller.initial_tokens)
    if len(prefix_tokens) != prefix_len:
        raise RuntimeError("persistent Beam Request view does not match controller prefix")

    state = PersistentBeamState(
        session_id=session_id,
        stage=PersistentBeamStage.PREFILLING,
        prefix_len=prefix_len,
        shared_prefix_tokens=prefix_tokens,
        scheduled_width=controller.beam_width,
        active_width=controller.beam_width,
        original_arrival_time=request.arrival_time,
        request_identity=id(request),
        prefix_block_hashes=tuple(request.block_hashes),
    )
    request.resumable = True
    request.gr_persistent_beam_state = state
    engine.persistent_beam_states[session_id] = state
    engine.persistent_active_session_id = session_id
    return state


def rollback_persistent_admission(engine, session_id: str) -> None:
    """Undo a partially admitted Request without relaxing the seen-ID guard."""
    initialize_persistent_engine_state(engine)
    state = engine.persistent_beam_states.pop(session_id, None)
    request = engine.scheduler.requests.get(session_id)
    if request is not None and persistent_beam_state(request) is state:
        request.resumable = False
        request.gr_persistent_beam_state = None
    if engine.persistent_active_session_id == session_id:
        engine.persistent_active_session_id = None


def _current_block_ids(scheduler, request_id: str) -> tuple[tuple[int, ...], ...] | None:
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    if kv_cache_manager is None:
        return None
    block_ids = kv_cache_manager.get_blocks(request_id).get_block_ids()
    return tuple(tuple(group) for group in block_ids)


def _verify_prefix_blocks(scheduler, state: PersistentBeamState) -> None:
    all_ids = _current_block_ids(scheduler, state.session_id)
    cache_config = getattr(scheduler, "cache_config", None)
    block_size = getattr(cache_config, "block_size", None)
    if all_ids is None or not isinstance(block_size, int) or block_size <= 0:
        return
    prefix_blocks = (state.prefix_len + block_size - 1) // block_size
    current = tuple(tuple(group[:prefix_blocks]) for group in all_ids)
    if not state.prefix_block_ids:
        state.prefix_block_ids = current
    elif current != state.prefix_block_ids:
        raise RuntimeError("persistent Beam shared prefix KV block IDs changed")


def _verify_execution_blocks(scheduler, state: PersistentBeamState) -> None:
    all_ids = _current_block_ids(scheduler, state.session_id)
    if all_ids is None or state.stage is PersistentBeamStage.PREFILLING:
        return
    counts = tuple(len(group) for group in all_ids)
    if not state.execution_block_counts:
        state.execution_block_counts = counts
    elif counts != state.execution_block_counts:
        raise RuntimeError("persistent Beam paged execution block count changed across steps")


def _prepare_execution_view(request, state: PersistentBeamState) -> None:
    """Rewind the Scheduler view before upstream appends the current lanes.

    Grouped Beam lane tokens are parallel execution rows, not serial history.
    Retaining an older lane row would shift positions by ``width`` on every
    continuation and produce incorrect logits from the second decode step.
    """
    if request.num_computed_tokens < state.prefix_len:
        raise RuntimeError("persistent Beam shared prefix is not fully computed")
    view = list(state.shared_prefix_tokens)
    request._all_token_ids[:] = view
    request.prompt_token_ids[:] = view
    request._output_token_ids.clear()
    request.num_prompt_tokens = state.prefix_len
    request.num_computed_tokens = state.prefix_len
    if hasattr(request, "num_output_placeholders"):
        request.num_output_placeholders = 0
    spec_token_ids = getattr(request, "spec_token_ids", None)
    if spec_token_ids is not None:
        spec_token_ids.clear()
    request.block_hashes = list(state.prefix_block_hashes)


def continue_persistent_request(
    engine,
    session_id: str,
    decision,
    controller,
    sampling_params,
) -> None:
    """Continue the same parked Request with one parallel Beam token row."""
    from vllm.v1.request import RequestStatus, StreamingUpdate

    initialize_persistent_engine_state(engine)
    state = engine.persistent_beam_states[session_id]
    request = engine.scheduler.requests[session_id]
    if id(request) != state.request_identity:
        raise RuntimeError("persistent Beam Request identity changed")
    if persistent_beam_state(request) is not state:
        raise RuntimeError("persistent Beam Request lost its sidecar state")
    if state.stage is not PersistentBeamStage.PARKED:
        raise RuntimeError(f"persistent Beam continuation requires PARKED, got {state.stage.value}")
    if request.status is not RequestStatus.WAITING_FOR_STREAMING_REQ:
        raise RuntimeError(
            f"persistent Beam continuation requires WAITING_FOR_STREAMING_REQ, got {request.status}"
        )
    if state.in_flight_epoch is not None:
        raise RuntimeError("persistent Beam session already has an epoch in flight")
    if state.completed_epoch != state.next_epoch - 1:
        raise RuntimeError("persistent Beam continuation arrived before its epoch completed")
    lane_tokens = list(decision.next_tokens)
    if len(lane_tokens) != decision.active_width:
        raise RuntimeError("persistent Beam continuation width does not match lane tokens")

    _verify_prefix_blocks(engine.scheduler, state)
    _prepare_execution_view(request, state)
    request.is_beam = True
    request.is_beam_decode = True
    request.prefix_len = state.prefix_len
    request.beam_width = decision.active_width
    request.beam_decode_step = decision.completed_step
    request.beam_decode_steps = decision.completed_step + 1
    request.parent_beam_indices = list(decision.parent_beam_indices)
    request.max_decode_steps = controller.total_execution_steps

    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=lane_tokens,
        max_tokens=1,
        arrival_time=state.original_arrival_time,
        sampling_params=copy(sampling_params),
    )
    engine.scheduler._update_request_as_session(request, update)
    # Lane hashes are transient execution state. Prefix hashes remain the only
    # reusable Scheduler cache identity across persistent Beam steps.
    request.block_hashes = list(state.prefix_block_hashes)
    state.stage = PersistentBeamStage.BEAM_DECODING
    state.logical_step = decision.completed_step
    state.scheduled_width = len(lane_tokens)
    state.active_width = decision.active_width


def mark_persistent_step_scheduled(scheduler, scheduler_output) -> None:
    """Open one epoch after Scheduler.schedule selects persistent work."""
    from vllm.v1.request import RequestStatus

    for request in scheduler.requests.values():
        state = persistent_beam_state(request)
        if state is not None and request.status is RequestStatus.PREEMPTED:
            state.stage = PersistentBeamStage.FAILED
            raise RuntimeError("persistent Beam Request preemption is not supported")

    for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests.get(request_id)
        state = persistent_beam_state(request) if request is not None else None
        if state is None:
            continue
        if state.stage not in (
            PersistentBeamStage.PREFILLING,
            PersistentBeamStage.BEAM_DECODING,
        ):
            raise RuntimeError(f"persistent Beam cannot schedule from {state.stage.value}")
        if state.in_flight_epoch is not None:
            raise RuntimeError("persistent Beam epoch was dispatched more than once")
        if id(request) != state.request_identity:
            raise RuntimeError("persistent Beam Request identity changed before dispatch")
        expected = state.scheduled_width
        if state.stage is PersistentBeamStage.PREFILLING:
            valid_width = 0 < num_tokens <= state.prefix_len
        else:
            valid_width = num_tokens == expected
        if not valid_width:
            raise RuntimeError(
                f"persistent Beam scheduled {num_tokens} tokens; expected current execution view"
            )
        # Chunked prefill may allocate the shared prefix over several epochs.
        # Snapshot it only after prefill parks, or on a decode dispatch.
        if state.stage is PersistentBeamStage.BEAM_DECODING:
            _verify_prefix_blocks(scheduler, state)
        _verify_execution_blocks(scheduler, state)
        state.in_flight_epoch = state.next_epoch
        state.next_epoch += 1


def complete_persistent_step(scheduler, scheduler_output) -> None:
    """Close one epoch after upstream update parks the resumable Request."""
    from vllm.v1.request import RequestStatus

    for request_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(request_id)
        state = persistent_beam_state(request) if request is not None else None
        if state is None:
            continue
        if state.in_flight_epoch is None:
            raise RuntimeError("persistent Beam output has no matching in-flight epoch")
        if request.status is RequestStatus.PREEMPTED:
            state.stage = PersistentBeamStage.FAILED
            raise RuntimeError("persistent Beam Request preemption is not supported")
        state.completed_epoch = state.in_flight_epoch
        state.in_flight_epoch = None
        if request.status is RequestStatus.WAITING_FOR_STREAMING_REQ:
            _verify_prefix_blocks(scheduler, state)
            state.stage = PersistentBeamStage.PARKED
        elif state.stage is not PersistentBeamStage.PREFILLING:
            raise RuntimeError("persistent Beam decode output did not park its resumable Request")


def begin_persistent_finalization(engine, session_id: str) -> bool:
    """CAS finalization and queue one Scheduler free plus Worker teardown."""
    initialize_persistent_engine_state(engine)
    state = engine.persistent_beam_states[session_id]
    if state.finalization_started:
        return False
    if state.in_flight_epoch is not None:
        raise RuntimeError("cannot finalize a persistent Beam epoch still in flight")
    state.finalization_started = True
    state.stage = PersistentBeamStage.FINALIZING
    request = engine.scheduler.requests.get(session_id)
    if request is None or persistent_beam_state(request) is not state:
        raise RuntimeError("persistent Beam Request disappeared before finalization")
    request.resumable = False
    scheduler = engine.scheduler
    if not hasattr(scheduler, "_beam_finished_session_ids"):
        scheduler._beam_finished_session_ids = []
    if session_id not in scheduler._beam_finished_session_ids:
        scheduler._beam_finished_session_ids.append(session_id)
    engine.aborts_queue.put_nowait(session_id)
    state.stage = PersistentBeamStage.DRAINING
    return True


def mark_persistent_final_result_sent(engine, session_id: str) -> None:
    state = engine.persistent_beam_states[session_id]
    if state.final_result_sent:
        raise RuntimeError("persistent Beam final result was sent more than once")
    state.final_result_sent = True


def mark_persistent_teardown_dispatched(scheduler, session_ids: list[str]) -> None:
    dispatched = getattr(scheduler, "_gr_persistent_drains_dispatched", None)
    if dispatched is None:
        dispatched = []
        scheduler._gr_persistent_drains_dispatched = dispatched
    for session_id in session_ids:
        request = scheduler.requests.get(session_id)
        state = persistent_beam_state(request) if request is not None else None
        if state is None:
            continue
        if state.stage is not PersistentBeamStage.DRAINING:
            raise RuntimeError("persistent Beam teardown dispatched outside DRAINING")
        if state.worker_teardown_sent:
            raise RuntimeError("persistent Beam Worker teardown was sent more than once")
        state.worker_teardown_sent = True
        dispatched.append(session_id)


def mark_persistent_request_free(request) -> None:
    state = persistent_beam_state(request)
    if state is None:
        return
    if state.stage is not PersistentBeamStage.DRAINING:
        raise RuntimeError("persistent Beam Request was freed before draining")
    if state.scheduler_freed:
        raise RuntimeError("persistent Beam Request was freed more than once")
    state.scheduler_freed = True


def finish_persistent_request(engine, session_id: str) -> None:
    """Complete a Worker drain and release the EngineCore admission fence."""
    state = engine.persistent_beam_states[session_id]
    if not state.worker_teardown_sent or not state.scheduler_freed:
        raise RuntimeError("persistent Beam drain completed before teardown/free")
    state.stage = PersistentBeamStage.FINISHED
    engine.persistent_beam_states.pop(session_id, None)
    if engine.persistent_active_session_id != session_id:
        raise RuntimeError("persistent Beam drain does not own the active Worker slot")
    engine.persistent_active_session_id = None


def finish_dispatched_persistent_drains(engine) -> None:
    """Called only after the Worker execute carrying teardown has returned."""
    initialize_persistent_engine_state(engine)
    scheduler = engine.scheduler
    session_ids = list(getattr(scheduler, "_gr_persistent_drains_dispatched", []))
    scheduler._gr_persistent_drains_dispatched = []
    for session_id in session_ids:
        finish_persistent_request(engine, session_id)
    if engine.persistent_active_session_id is None and engine.persistent_pending_starts:
        start, request = engine.persistent_pending_starts.popleft()
        session_id = start.controller.beam_session_id
        engine.persistent_pending_session_ids.remove(session_id)
        engine.create_controller(start, request)


def abort_persistent_request(engine, session_id: str) -> bool:
    return begin_persistent_finalization(engine, session_id)
