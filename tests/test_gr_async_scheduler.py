# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# pytest decorators are untyped in the mypy-only pre-commit environments.
# mypy: disable-error-code="misc"
import asyncio
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)

from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
)
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRResourceStatus,
    GRSessionStatus,
    GRStageKind,
    GRWorkerResult,
)
from vllm_gr.v1.engine.core import _handle_submit_gr_session
from vllm_gr.v1.engine.core_client import (
    submit_gr_session_async as submit_gr_session_to_core_async,
)
from vllm_gr.v1.engine.core_client_patch import sync_mp_submit_gr_session
from vllm_gr.v1.engine.engine_core_patch import (
    _GR_FINALIZING_IDS_ATTR,
    _finalize_ready_gr_session,
    _patch_scheduler_reset_prefix_cache,
    _patch_scheduler_update_from_output,
    _sync_gr_sessions_after_native_finish,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    _gr_session_ready_to_finalize,
    add_gr_session,
    attach_gr_stage_metadata,
    can_issue_gr_stage,
    update_gr_from_output,
)
from vllm_gr.v1.engine.gr_session_client import (
    SyncGRSessionHandle,
    _terminal_result,
    abort_gr_session_async,
    abort_gr_session_sync,
    wait_gr_session_sync,
)
from vllm_gr.v1.engine.wire import register_engine_core_request_types


def _worker_result(*args: Any, **kwargs: Any) -> GRWorkerResult:
    return GRWorkerResult(*args, owner_generation=1, **kwargs)


def _batch_request(max_tokens: int = 3) -> BeamBatchRequest:
    request = EngineCoreRequest(
        request_id="session-0",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id="session-0",
        items=(BeamBatchItem(item_index=0, request=request),),
        beam_params=BeamBatchParams(
            beam_width=4,
            max_tokens=max_tokens,
            temperature=0.0,
            stop_token_ids=(2,),
            ignore_stop_tokens=True,
        ),
        output_options=BeamOutputOptions(
            num_return_sequences=4,
            include_stop_token=True,
            include_cumulative_logprob=True,
            include_parent_beam_indices=True,
        ),
    )


def _scheduler() -> SimpleNamespace:
    native_request = SimpleNamespace(
        request_id="session-0",
        is_prefill_chunk=False,
        num_output_placeholders=0,
        num_prompt_tokens=3,
        num_computed_tokens=2,
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        requests={"session-0": native_request},
    )


def _scheduler_output(scheduler: Any) -> SimpleNamespace:
    scheduler.requests["session-0"].num_computed_tokens += 1
    return SimpleNamespace(num_scheduled_tokens={"session-0": 1})


def _failure_result(batch_id: str = "session-0") -> BeamBatchResult:
    return BeamBatchResult(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=batch_id,
        output=None,
        failure=BeamBatchFailure("test_failure", "expected", None),
    )


def test_prefill_decode_lookahead_uses_device_dependency_without_cpu_update() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    native_request = scheduler.requests[session.session_id]

    native_request.num_output_placeholders = 1
    prefill_output = _scheduler_output(scheduler)
    prefill = attach_gr_stage_metadata(scheduler, prefill_output, session)
    setattr(prefill_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: prefill})

    assert prefill is not None
    assert prefill.kind == GRStageKind.PREFILL
    assert prefill.input_device_state is None
    assert can_issue_gr_stage(scheduler, session)

    # Decode1 is issued before Prefill's Future is consumed by the CPU.
    native_request.num_output_placeholders = 2
    decode_output = _scheduler_output(scheduler)
    decode = attach_gr_stage_metadata(scheduler, decode_output, session)
    setattr(decode_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: decode})

    assert decode is not None
    assert decode.kind == GRStageKind.DECODE
    assert decode.input_device_state == prefill.output_device_state
    assert session.in_flight_stages == 2
    assert not can_issue_gr_stage(scheduler, session)

    # Reaping old Prefill output opens one queue slot while Decode1 remains live.
    native_request.num_output_placeholders = 1
    worker_result = _worker_result(
        session_id=session.session_id,
        dispatch_id=prefill.dispatch_id,
        stage_index=0,
        produced_token_count=1,
        device_state=GRDeviceStateRef(session.session_id, 0),
        finished=False,
    )
    model_output = SimpleNamespace(**{GR_WORKER_RESULT_ATTR: {session.session_id: worker_result}})
    update_gr_from_output(scheduler, session, prefill_output, model_output)

    assert session.completed_stages == 1
    assert session.in_flight_stages == 1
    assert can_issue_gr_stage(scheduler, session)


def test_early_finish_stops_issue_and_drains_already_queued_stage() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    native_request = scheduler.requests[session.session_id]
    outputs = []
    for placeholders in (1, 2):
        native_request.num_output_placeholders = placeholders
        output = _scheduler_output(scheduler)
        metadata = attach_gr_stage_metadata(scheduler, output, session)
        assert metadata is not None
        setattr(output, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
        outputs.append((output, metadata))

    terminal_result = _failure_result()
    first_result = _worker_result(
        session_id=session.session_id,
        dispatch_id=outputs[0][1].dispatch_id,
        stage_index=0,
        produced_token_count=1,
        device_state=GRDeviceStateRef(session.session_id, 0),
        finished=True,
        terminal_result=terminal_result,
    )
    native_request.num_output_placeholders = 1
    update_gr_from_output(
        scheduler,
        session,
        outputs[0][0],
        SimpleNamespace(**{GR_WORKER_RESULT_ATTR: {session.session_id: first_result}}),
    )

    assert session.status == GRSessionStatus.FINISHING
    assert not can_issue_gr_stage(scheduler, session)

    premature_output = EngineCoreOutput(
        request_id=session.session_id,
        new_token_ids=[],
        finish_reason=FinishReason.STOP,
    )
    premature_result = {
        0: EngineCoreOutputs(
            outputs=[premature_output],
            finished_requests={session.session_id},
        )
    }
    _finalize_ready_gr_session(scheduler, premature_result, session)

    assert not premature_output.finished
    assert premature_result[0].finished_requests is not None
    assert session.session_id not in premature_result[0].finished_requests

    queued_result = _worker_result(
        session_id=session.session_id,
        dispatch_id=outputs[1][1].dispatch_id,
        stage_index=1,
        produced_token_count=0,
        device_state=GRDeviceStateRef(session.session_id, 1),
        finished=True,
    )
    update_gr_from_output(
        scheduler,
        session,
        outputs[1][0],
        SimpleNamespace(**{GR_WORKER_RESULT_ATTR: {session.session_id: queued_result}}),
    )

    assert session.status == GRSessionStatus.FINISHING
    assert session.completed_stages == 2
    assert session._pending_terminal_result == terminal_result
    assert not _gr_session_ready_to_finalize(session)
    from tests.test_async_scheduling_scheduler import release_receipt
    from vllm_gr.v1.engine.gr_async_scheduler import (
        attach_gr_retirement,
        consume_gr_retirement,
    )

    retirement = SimpleNamespace(total_num_scheduled_tokens=0, finished_req_ids=set())
    attach_gr_retirement(scheduler, retirement)
    consume_gr_retirement(scheduler, retirement, release_receipt(retirement))
    assert _gr_session_ready_to_finalize(session)
    assert session.session_id in scheduler._gr_v1_sessions

    native_request = scheduler.requests[session.session_id]
    native_request.client_index = 0
    native_request.is_finished = lambda: False
    finished: list[tuple[str, object]] = []

    def finish_requests(request_id: str, status: object) -> list[tuple[str, int]]:
        finished.append((request_id, status))
        return [(request_id, 0)]

    scheduler.finish_requests = finish_requests
    result: dict[int, Any] = {}
    _finalize_ready_gr_session(scheduler, result, session)

    assert finished
    assert native_request.num_output_placeholders == 0
    assert session.status == GRSessionStatus.FINISHED
    assert session.resource_status is GRResourceStatus.CLOSED
    assert session.session_id not in scheduler._gr_v1_sessions
    output = result[0].outputs[0]
    assert output.finished
    assert output.kv_transfer_params is not None
    assert output.kv_transfer_params["gr_batch_result"] == terminal_result


def test_native_eos_enters_gr_drain_after_worker_receipt() -> None:
    class NativeScheduler:
        def __init__(self) -> None:
            native_request = _scheduler().requests["session-0"]
            native_request.client_index = 0
            self.scheduler_config = SimpleNamespace(async_scheduling=True)
            self.requests = {"session-0": native_request}
            self._gr_v1_sessions: dict[str, Any] = {}

        def update_gr_from_output(
            self, session: Any, scheduler_output: Any, model_output: Any
        ) -> Any:
            return update_gr_from_output(self, session, scheduler_output, model_output)

        def update_from_output(
            self,
            scheduler_output: Any,
            model_output: Any,
            *args: Any,
            **kwargs: Any,
        ) -> dict[int, EngineCoreOutputs]:
            metadata = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {}).get("session-0")
            if metadata is None:
                return {0: EngineCoreOutputs(outputs=[])}
            if metadata.stage_index != 0:
                return {0: EngineCoreOutputs(outputs=[])}

            self.requests.pop("session-0")
            _sync_gr_sessions_after_native_finish(
                self,
                [("session-0", 0)],
                SimpleNamespace(name="FINISHED_STOPPED"),
            )
            return {
                0: EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id="session-0",
                            new_token_ids=[2],
                            finish_reason=FinishReason.STOP,
                        )
                    ],
                    finished_requests={"session-0"},
                )
            }

    _patch_scheduler_update_from_output(NativeScheduler)
    scheduler = NativeScheduler()
    request = _batch_request()
    request.items[0].request.client_index = 0
    session = add_gr_session(scheduler, request)
    outputs = []
    for placeholders in (1, 2):
        scheduler.requests[session.session_id].num_output_placeholders = placeholders
        scheduler_output = _scheduler_output(scheduler)
        metadata = attach_gr_stage_metadata(scheduler, scheduler_output, session)
        assert metadata is not None
        setattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
        outputs.append(scheduler_output)

    prefill = getattr(outputs[0], GR_STAGE_OUTPUT_ATTR)[session.session_id]
    worker_result = _worker_result(
        session.session_id,
        prefill.dispatch_id,
        prefill.stage_index,
        1,
        prefill.output_device_state,
        False,
    )
    premature_result = scheduler.update_from_output(
        outputs[0],
        SimpleNamespace(**{GR_WORKER_RESULT_ATTR: {session.session_id: worker_result}}),
    )

    assert session.status == GRSessionStatus.FINISHING
    assert session.terminal_result_pending
    assert session.session_id in scheduler._gr_v1_sessions
    assert len(session._in_flight_dispatches) == 1
    assert not premature_result[0].outputs[0].finished
    assert premature_result[0].finished_requests is not None
    assert session.session_id not in premature_result[0].finished_requests

    terminal_result = scheduler.update_from_output(outputs[1], SimpleNamespace())

    assert session.resource_status is GRResourceStatus.DRAINING
    assert scheduler._gr_v1_sessions[session.session_id] is session
    assert not terminal_result[0].outputs
    from tests.test_async_scheduling_scheduler import release_receipt
    from vllm_gr.v1.engine.gr_async_scheduler import attach_gr_retirement

    retirement = SimpleNamespace(total_num_scheduled_tokens=0, finished_req_ids=set())
    attach_gr_retirement(scheduler, retirement)
    terminal_result = scheduler.update_from_output(retirement, release_receipt(retirement))
    assert session.resource_status is GRResourceStatus.CLOSED
    assert session.session_id not in scheduler._gr_v1_sessions
    final_output = terminal_result[0].outputs[0]
    assert final_output.finished
    failure = final_output.kv_transfer_params["gr_batch_result"].failure
    assert failure is not None
    assert failure.code == "native_termination_without_gr_result"


def test_partial_prefill_rejected_without_dispatch_or_lease() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    scheduler.requests[session.session_id].is_prefill_chunk = True
    with pytest.raises(RuntimeError, match="partial Prefill"):
        attach_gr_stage_metadata(scheduler, _scheduler_output(scheduler), session)
    assert not session._dispatch_records
    assert not session._in_flight_dispatches
    assert not session._kv_dispatch_leases
    assert not session.worker_resources_allocated


def test_recompute_is_rejected_after_beam_initialization() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    native_request = scheduler.requests[session.session_id]

    native_request.num_output_placeholders = 1
    prefill_output = _scheduler_output(scheduler)
    prefill = attach_gr_stage_metadata(scheduler, prefill_output, session)
    assert prefill is not None
    setattr(prefill_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: prefill})
    native_request.num_output_placeholders = 0
    update_gr_from_output(
        scheduler,
        session,
        prefill_output,
        SimpleNamespace(
            **{
                GR_WORKER_RESULT_ATTR: {
                    session.session_id: _worker_result(
                        session_id=session.session_id,
                        dispatch_id=prefill.dispatch_id,
                        stage_index=prefill.stage_index,
                        produced_token_count=1,
                        device_state=prefill.output_device_state,
                        finished=False,
                    )
                }
            }
        ),
    )

    latest_device_state = session.latest_device_state
    native_request.is_prefill_chunk = True
    recompute_output = _scheduler_output(scheduler)
    with pytest.raises(RuntimeError, match="cannot recompute"):
        attach_gr_stage_metadata(scheduler, recompute_output, session)
    assert session.issued_stages == 1
    assert session.latest_device_state == latest_device_state
    assert session.completed_stages == 1
    assert not session._in_flight_dispatches


def test_backend_configuration_error_uses_request_error_channel() -> None:
    request = _batch_request()
    errors: list[object] = []
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"vllm_gr_beam_execution_mode": "v1"},
            scheduler_config=SimpleNamespace(async_scheduling=False),
        ),
        batch_queue=object(),
        batch_queue_size=2,
        _handle_request_preproc_error=lambda native_request: errors.append(native_request),
    )

    _handle_submit_gr_session(engine, request)

    assert errors == [request.items[0].request]


def test_gr_submission_disables_native_stop_ownership() -> None:
    request = _batch_request()
    native_request = SimpleNamespace(
        sampling_params=SimpleNamespace(ignore_eos=True, stop_token_ids=[2])
    )
    added: list[Any] = []
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"vllm_gr_beam_execution_mode": "v1"},
            scheduler_config=SimpleNamespace(async_scheduling=True),
        ),
        batch_queue=object(),
        batch_queue_size=2,
        preprocess_add_request=lambda engine_request: (native_request,),
        add_request=lambda processed: added.append(processed),
    )

    _handle_submit_gr_session(engine, request)

    assert native_request.sampling_params.ignore_eos
    assert native_request.sampling_params.stop_token_ids == []
    assert added == [native_request]


def test_forced_prefix_reset_is_rejected_while_gr_session_is_active() -> None:
    class ResetScheduler:
        def __init__(self) -> None:
            self._gr_v1_sessions = {"session-0": object()}
            self.calls: list[bool] = []

        def reset_prefix_cache(self, reset_running_requests: bool = False) -> bool:
            self.calls.append(reset_running_requests)
            return True

    _patch_scheduler_reset_prefix_cache(ResetScheduler)
    scheduler = ResetScheduler()

    assert not scheduler.reset_prefix_cache(reset_running_requests=True)
    assert scheduler.calls == []
    assert scheduler.reset_prefix_cache(reset_running_requests=False)
    assert scheduler.calls == [False]


def test_native_abort_all_result_cleans_unscheduled_gr_session() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())

    _sync_gr_sessions_after_native_finish(
        scheduler,
        [(session.session_id, 0)],
        SimpleNamespace(name="FINISHED_ABORTED"),
    )

    assert session.status == GRSessionStatus.ABORTED
    assert session.resource_status is GRResourceStatus.CLOSED
    assert session.session_id not in scheduler._gr_v1_sessions


def test_native_abort_overrides_a_pending_gr_terminal_result() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    session.status = GRSessionStatus.FINISHING
    session.terminal_result_pending = True
    session._pending_terminal_result = _failure_result()

    _sync_gr_sessions_after_native_finish(
        scheduler,
        [(session.session_id, 0)],
        SimpleNamespace(name="FINISHED_ABORTED"),
    )

    assert session.status == GRSessionStatus.ABORTED
    assert not session.terminal_result_pending
    assert session.resource_status is GRResourceStatus.CLOSED
    assert session.session_id not in scheduler._gr_v1_sessions


def test_internal_gr_finalization_preserves_the_pending_terminal_result() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    session.status = GRSessionStatus.FINISHING
    session.terminal_result_pending = True
    session._pending_terminal_result = _failure_result()
    setattr(scheduler, _GR_FINALIZING_IDS_ATTR, {session.session_id})

    _sync_gr_sessions_after_native_finish(
        scheduler,
        [(session.session_id, 0)],
        SimpleNamespace(name="FINISHED_ERROR"),
    )

    assert session.status == GRSessionStatus.FINISHING
    assert session.terminal_result_pending
    assert session.session_id in scheduler._gr_v1_sessions


@pytest.mark.parametrize("online", [False, True])
def test_abort_adapters_wait_for_retirement_with_internal_id(online: bool) -> None:
    from unittest.mock import AsyncMock, Mock

    internal_id = "internal-randomized-id"
    output_processor = SimpleNamespace(abort_requests=Mock(return_value=[]))
    if online:
        core = SimpleNamespace(
            abort_requests_async=AsyncMock(),
            call_utility_async=AsyncMock(side_effect=[False, False, True]),
        )
        engine = SimpleNamespace(output_processor=output_processor, engine_core=core)
        asyncio.run(abort_gr_session_async(engine, internal_id))
        core.abort_requests_async.assert_awaited_once_with([internal_id])
        assert core.call_utility_async.await_count == 3
        core.call_utility_async.assert_awaited_with("is_gr_session_retired", internal_id)
    else:
        core = SimpleNamespace(
            abort_requests=Mock(), call_utility=Mock(side_effect=[False, False, True])
        )
        engine = SimpleNamespace(output_processor=output_processor, engine_core=core)
        abort_gr_session_sync(engine, internal_id)
        core.abort_requests.assert_called_once_with([internal_id])
        assert core.call_utility.call_count == 3
        core.call_utility.assert_called_with("is_gr_session_retired", internal_id)
    output_processor.abort_requests.assert_called_once_with([internal_id], internal=True)


def test_terminal_result_restores_untyped_multiprocess_payload() -> None:
    request = _batch_request()
    expected = _failure_result()
    wire_value = msgspec.to_builtins(expected)
    assert isinstance(wire_value, list)

    output = SimpleNamespace(kv_transfer_params={"gr_batch_result": wire_value})

    assert _terminal_result(request, output) == expected


def test_sync_wait_uses_the_native_external_output_id() -> None:
    request = _batch_request()
    request.items[0].request.external_req_id = "external-request"
    expected = _failure_result()
    calls = 0

    def step() -> list[SimpleNamespace]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("wait did not accept the external request ID")
        return [
            SimpleNamespace(
                request_id="external-request",
                finished=True,
                kv_transfer_params={"gr_batch_result": expected},
            )
        ]

    result = wait_gr_session_sync(
        SimpleNamespace(step=step),
        request.batch_id,
        SyncGRSessionHandle(request),
    )

    assert result == expected
    assert calls == 1


def test_sync_mp_submit_defaults_the_missing_client_index() -> None:
    register_engine_core_request_types()
    request = _batch_request()
    sent = []
    client = SimpleNamespace(
        is_dp=False,
        _send_input=lambda request_type, payload: sent.append((request_type, payload)),
    )

    sync_mp_submit_gr_session(client, request)

    assert request.items[0].request.client_index == 0
    assert len(sent) == 1
    assert sent[0][1] is request


def test_async_dp_submit_starts_stats_task_before_first_request() -> None:
    register_engine_core_request_types()
    request = _batch_request()
    events: list[tuple[Any, ...]] = []

    async def send_input(request_type: Any, payload: Any, engine: Any) -> None:
        events.append(("send", request_type, payload, engine))

    async def send_first_request(payload: Any) -> None:
        events.append(("first", payload))

    client = SimpleNamespace(
        client_index=2,
        current_wave=7,
        engines_running=False,
        first_req_send_socket=SimpleNamespace(send=send_first_request),
        _ensure_stats_update_task=lambda: events.append(("stats",)),
        _ensure_output_queue_task=lambda: events.append(("output",)),
        get_core_engine_for_request=lambda native_request: b"engine-1",
        _send_input=send_input,
    )

    asyncio.run(submit_gr_session_to_core_async(client, request))

    assert events[0] == ("stats",)
    assert events[1][0] == "first"
    assert events[2][0] == "send"
    assert events[3] == ("output",)
    assert request.items[0].request.client_index == 2
    assert request.items[0].request.current_wave == 7


def test_prefill_validates_worker_receipt_identity() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    scheduled = _scheduler_output(scheduler)
    metadata = attach_gr_stage_metadata(scheduler, scheduled, session)
    assert metadata is not None
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
    result = _worker_result(
        session.session_id,
        metadata.dispatch_id + 1,
        metadata.stage_index,
        0,
        metadata.output_device_state,
        False,
    )
    output = SimpleNamespace()
    setattr(output, GR_WORKER_RESULT_ATTR, {session.session_id: result})
    completion = update_gr_from_output(scheduler, session, scheduled, output)
    assert completion.finished
    assert "different dispatch" in completion.error
    assert session.completed_stages == 1


@pytest.mark.parametrize("has_mapping", [False, True])
def test_missing_worker_receipt_fails_without_confirming_tokens(has_mapping: bool) -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    scheduled = _scheduler_output(scheduler)
    metadata = attach_gr_stage_metadata(scheduler, scheduled, session)
    assert metadata is not None
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
    output = SimpleNamespace(**({GR_WORKER_RESULT_ATTR: {}} if has_mapping else {}))

    completion = update_gr_from_output(scheduler, session, scheduled, output)

    assert completion.finished and "omitted GRWorkerResult" in (completion.error or "")
    assert not session._in_flight_dispatches
    assert session.resource_status is GRResourceStatus.DRAINING
    assert session._pending_terminal_result is not None
    assert session._pending_terminal_result.failure is not None
    assert session._pending_terminal_result.failure.code == "worker_protocol_error"


def test_prefill_rejects_zero_without_finishing() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    scheduled = _scheduler_output(scheduler)
    metadata = attach_gr_stage_metadata(scheduler, scheduled, session)
    assert metadata is not None
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
    output = SimpleNamespace()
    setattr(
        output,
        GR_WORKER_RESULT_ATTR,
        {
            session.session_id: _worker_result(
                session.session_id,
                metadata.dispatch_id,
                metadata.stage_index,
                0,
                metadata.output_device_state,
                False,
            )
        },
    )
    completion = update_gr_from_output(scheduler, session, scheduled, output)
    assert completion.finished and "no token" in completion.error
    assert session.completed_stages == 1
    assert not session._in_flight_dispatches


def test_native_kv_dispatch_lease_survives_abort_until_receipt() -> None:
    scheduler = _scheduler()
    block = SimpleNamespace(ref_cnt=1)

    def touch(blocks: Any) -> None:
        for value in blocks:
            value.ref_cnt += 1

    def free(blocks: Any) -> None:
        for value in blocks:
            value.ref_cnt -= 1

    scheduler.kv_cache_manager = SimpleNamespace(
        get_blocks=lambda req_id: SimpleNamespace(blocks=((block,),)),
        block_pool=SimpleNamespace(touch=touch, free_blocks=free, num_gpu_blocks=8),
        max_model_len=4096,
    )
    session = add_gr_session(scheduler, _batch_request())
    scheduled = _scheduler_output(scheduler)
    metadata = attach_gr_stage_metadata(scheduler, scheduled, session)
    assert block.ref_cnt == 2
    free((block,))  # Native abort/preemption drops its request reference.
    session.status = GRSessionStatus.ABORTED
    assert block.ref_cnt == 1
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
    update_gr_from_output(scheduler, session, scheduled, SimpleNamespace())
    assert block.ref_cnt == 0
    update_gr_from_output(scheduler, session, scheduled, SimpleNamespace())
    assert block.ref_cnt == 0


@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize("num_leases", [1, 2])
@pytest.mark.parametrize("native_first", [False, True])
@pytest.mark.parametrize("shared_prefix", [False, True])
@pytest.mark.parametrize("status", [GRSessionStatus.ABORTED, GRSessionStatus.FINISHING])
def test_dispatch_lease_preserves_native_eviction_order(
    num_groups: int,
    num_leases: int,
    native_first: bool,
    shared_prefix: bool,
    status: GRSessionStatus,
) -> None:
    def resources() -> tuple[BlockPool, list[Any], list[Any]]:
        pool = BlockPool(num_gpu_blocks=3 * num_groups + 2, enable_caching=True, hash_block_size=16)
        blocks = pool.get_new_blocks(3 * num_groups)
        # Physical block IDs deliberately differ from prompt order.
        groups = [
            [blocks[start + 2], blocks[start], blocks[start + 1]]
            for start in range(0, len(blocks), 3)
        ]
        if shared_prefix:
            pool.touch([groups[0][0]])
        managers = [
            SimpleNamespace(
                req_to_blocks={"session-0": group}, num_cached_block={}, block_pool=pool
            )
            for group in groups
        ]
        return pool, groups, managers

    def native_free(managers: list[Any]) -> None:
        for manager in managers:
            SingleTypeKVCacheManager.free(manager, "session-0")

    reference, _, reference_managers = resources()
    native_free(reference_managers)
    expected = [block.block_id for block in reference.free_block_queue.get_all_free_blocks()]
    pool, groups, managers = resources()
    scheduler = _scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(
        get_blocks=lambda _: SimpleNamespace(blocks=groups), block_pool=pool, max_model_len=4096
    )
    session = add_gr_session(scheduler, _batch_request())
    scheduled = []
    for _ in range(num_leases):
        output = _scheduler_output(scheduler)
        m = attach_gr_stage_metadata(scheduler, output, session)
        setattr(output, GR_STAGE_OUTPUT_ATTR, {session.session_id: m})
        scheduled.append(output)
    session.status = status
    if native_first:
        native_free(managers)
    for output in reversed(scheduled):
        update_gr_from_output(scheduler, session, output, SimpleNamespace())
    if not native_first:
        native_free(managers)
    assert [block.block_id for block in pool.free_block_queue.get_all_free_blocks()] == expected
    assert not session._kv_dispatch_leases
    before = [block.ref_cnt for block in pool.blocks]
    for output in scheduled:
        update_gr_from_output(scheduler, session, output, SimpleNamespace())
    assert [block.ref_cnt for block in pool.blocks] == before
    assert all(block.ref_cnt == 0 for block in pool.blocks if block.block_id in expected)
    if shared_prefix:
        assert groups[0][0].ref_cnt == 1
    assert [block.block_id for block in pool.get_new_blocks(len(expected))] == expected


def test_async_output_cache_never_hashes_generated_placeholders() -> None:
    from vllm_gr.v1.engine.gr_async_scheduler import _install_gr_prompt_cache_guard

    calls: list[int] = []
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(cache_blocks=lambda request, count: calls.append(count))
    )
    _install_gr_prompt_cache_guard(scheduler)
    _install_gr_prompt_cache_guard(scheduler)
    cache = scheduler.kv_cache_manager.cache_blocks
    gr = SimpleNamespace(gr_batch_request=object(), num_prompt_tokens=31)
    cache(gr, 30)
    cache(gr, 32)
    cache(SimpleNamespace(), 32)
    assert calls == [30, 31, 32]
