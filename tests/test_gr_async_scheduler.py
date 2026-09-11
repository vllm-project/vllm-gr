from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamOutputOptions,
)
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRSessionStatus,
    GRStageKind,
    GRWorkerResult,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    add_gr_session,
    attach_gr_stage_metadata,
    can_issue_gr_stage,
    update_gr_from_output,
)


def _batch_request(max_tokens: int = 3) -> BeamBatchRequest:
    request = EngineCoreRequest(
        request_id="session-0",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
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
            ignore_stop_tokens=False,
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
    )
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        requests={"session-0": native_request},
    )


def _scheduler_output() -> SimpleNamespace:
    return SimpleNamespace(num_scheduled_tokens={"session-0": 1})


def test_prefill_decode_lookahead_uses_device_dependency_without_cpu_update() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    native_request = scheduler.requests[session.session_id]

    native_request.num_output_placeholders = 1
    prefill_output = _scheduler_output()
    prefill = attach_gr_stage_metadata(scheduler, prefill_output, session)
    setattr(prefill_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: prefill})

    assert prefill is not None
    assert prefill.kind == GRStageKind.PREFILL
    assert prefill.input_device_state is None
    assert can_issue_gr_stage(scheduler, session)

    # Decode1 is issued before Prefill's Future is consumed by the CPU.
    native_request.num_output_placeholders = 2
    decode_output = _scheduler_output()
    decode = attach_gr_stage_metadata(scheduler, decode_output, session)
    setattr(decode_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: decode})

    assert decode is not None
    assert decode.kind == GRStageKind.DECODE
    assert decode.input_device_state == prefill.output_device_state
    assert session.in_flight_stages == 2
    assert not can_issue_gr_stage(scheduler, session)

    # Reaping old Prefill output opens one queue slot while Decode1 remains live.
    native_request.num_output_placeholders = 1
    worker_result = GRWorkerResult(
        session_id=session.session_id,
        stage_index=0,
        produced_token_count=1,
        device_state=GRDeviceStateRef(session.session_id, 0),
        finished=False,
    )
    model_output = SimpleNamespace(
        **{GR_WORKER_RESULT_ATTR: {session.session_id: worker_result}}
    )
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
        output = _scheduler_output()
        metadata = attach_gr_stage_metadata(scheduler, output, session)
        setattr(output, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
        outputs.append((output, metadata))

    first_result = GRWorkerResult(
        session_id=session.session_id,
        stage_index=0,
        produced_token_count=1,
        device_state=GRDeviceStateRef(session.session_id, 0),
        finished=True,
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

    queued_result = GRWorkerResult(
        session_id=session.session_id,
        stage_index=1,
        produced_token_count=0,
        device_state=GRDeviceStateRef(session.session_id, 1),
        finished=True,
    )
    scheduler.requests.pop(session.session_id)
    update_gr_from_output(
        scheduler,
        session,
        outputs[1][0],
        SimpleNamespace(**{GR_WORKER_RESULT_ATTR: {session.session_id: queued_result}}),
    )

    assert session.status == GRSessionStatus.FINISHED
    assert session.session_id not in scheduler._gr_v1_sessions


def test_chunked_prefill_does_not_consume_a_logical_stage() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, _batch_request())
    scheduler.requests[session.session_id].is_prefill_chunk = True
    output = _scheduler_output()

    metadata = attach_gr_stage_metadata(scheduler, output, session)

    assert metadata is not None
    assert metadata.kind == GRStageKind.PREFILL
    assert not metadata.produces_output
    assert session.issued_stages == 0
    assert session.output_placeholders == 0
