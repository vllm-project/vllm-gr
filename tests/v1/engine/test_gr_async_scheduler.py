# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# pytest decorators are untyped in the mypy-only pre-commit environments.
# mypy: disable-error-code="misc,no-untyped-def,var-annotated,no-untyped-call"
import asyncio
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from typing import Any

import msgspec
import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)

from tests.helpers.beam_requests import make_batch_request, make_multi_batch_request
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchResult,
    GRAdmissionError,
)
from vllm_gr.v1.beam.dispatch import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRDeviceStateRef,
    GRRetireResult,
    GRStageKind,
    GRWorkerResult,
)
from vllm_gr.v1.core.sched.beam_async_scheduler import (
    _drain_resources,
    _gr_session_ready_to_finalize,
    add_gr_session,
    attach_gr_retirement,
    attach_gr_stage_metadata,
    can_issue_gr_stage,
    consume_gr_retirement,
    gr_admission,
    update_gr_from_output,
)
from vllm_gr.v1.core.sched.beam_session import GRResourceStatus, GRSessionStatus
from vllm_gr.v1.core.sched.scheduler_patch import (
    _GR_FINALIZING_IDS_ATTR,
    _expand_gr_abort_request_ids,
    _finalize_ready_gr_session,
    _patch_scheduler_reset_prefix_cache,
    _patch_scheduler_update_from_output,
    _sync_gr_sessions_after_native_finish,
    apply_scheduler_patch,
)
from vllm_gr.v1.engine.beam_session_client import (
    SyncGRSessionHandle,
    _terminal_result,
    abort_gr_session_async,
    abort_gr_session_sync,
    wait_gr_session_sync,
)
from vllm_gr.v1.engine.core import _handle_submit_gr_session
from vllm_gr.v1.engine.core_client import (
    submit_gr_session_async as submit_gr_session_to_core_async,
)
from vllm_gr.v1.engine.core_client import (
    sync_mp_submit_gr_session,
)
from vllm_gr.v1.engine.wire import register_engine_core_request_types


def _scheduler() -> SimpleNamespace:
    native_request = SimpleNamespace(
        request_id="session-0",
        is_prefill_chunk=False,
        num_output_placeholders=0,
        num_prompt_tokens=3,
        num_computed_tokens=3,
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        requests={"session-0": native_request},
    )


def _scheduler_output() -> SimpleNamespace:
    return SimpleNamespace(num_scheduled_tokens={"session-0": 1})


@pytest.mark.parametrize(
    "call_style", ["default", "positional_false", "positional_true", "keyword_true"]
)
@pytest.mark.parametrize("raise_error", [False, True])
def test_scheduler_patch_processes_subclass_super_chain_once(
    monkeypatch, call_style, raise_error
) -> None:
    """A platform Scheduler subclass may delegate to an already-patched base."""

    from vllm.v1.core.sched import scheduler as scheduler_module

    from vllm_gr.v1.core.sched import beam_async_scheduler as gr_async_scheduler

    native_calls: list[tuple[str, bool]] = []
    retirement_calls: list[object] = []

    class BaseScheduler:
        def __init__(self) -> None:
            coordinator = SimpleNamespace(cache_blocks=lambda *_args, **_kwargs: None)
            self.kv_cache_manager = SimpleNamespace(
                coordinator=coordinator,
                get_computed_blocks=lambda _request: (SimpleNamespace(blocks=()), 0),
                create_kv_cache_blocks=lambda blocks: SimpleNamespace(blocks=blocks),
            )
            self.cache_config = SimpleNamespace(block_size=16)
            self.requests = {}

        def has_requests(self):
            return False

        def add_request(self, request):
            self.requests[request.request_id] = request

        def finish_requests(self, request_ids, finished_status):
            return None

        def reset_prefix_cache(self, *args, **kwargs):
            return True

        def update_from_output(self, *args, **kwargs):
            return {}

        def schedule(self, throttle_prefills=False):
            native_calls.append(("base", throttle_prefills))
            if raise_error:
                raise RuntimeError("native schedule failure")
            return SimpleNamespace(
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
                finished_req_ids=set(),
            )

    monkeypatch.setattr(
        gr_async_scheduler,
        "attach_gr_retirement",
        lambda _scheduler, output: retirement_calls.append(output),
    )
    monkeypatch.setattr(scheduler_module, "Scheduler", BaseScheduler)
    apply_scheduler_patch()

    class BalanceScheduler(BaseScheduler):
        def schedule(self, throttle_prefills=False):
            native_calls.append(("balance", throttle_prefills))
            return super().schedule(throttle_prefills)

    monkeypatch.setattr(scheduler_module, "Scheduler", BalanceScheduler)
    apply_scheduler_patch()

    scheduler = BalanceScheduler()
    original_get_computed_blocks = scheduler.kv_cache_manager.get_computed_blocks
    original_cache_blocks = scheduler.kv_cache_manager.coordinator.cache_blocks
    expected_throttle = call_style in ("positional_true", "keyword_true")
    with (
        pytest.raises(RuntimeError, match="native schedule failure")
        if raise_error
        else nullcontext()
    ):
        if call_style == "default":
            output = scheduler.schedule()
        elif call_style == "keyword_true":
            output = scheduler.schedule(throttle_prefills=True)
        else:
            output = scheduler.schedule(expected_throttle)

    assert native_calls == [("balance", expected_throttle), ("base", expected_throttle)]
    if raise_error:
        assert retirement_calls == []
    else:
        assert retirement_calls == [output]
        assert output.gr_stage_metadata == {}
    assert not hasattr(scheduler, "_vllm_gr_schedule_patch_depth")
    assert scheduler.kv_cache_manager.get_computed_blocks is original_get_computed_blocks
    assert scheduler.kv_cache_manager.coordinator.cache_blocks is original_cache_blocks


def test_caller_formed_batch_is_one_ordered_scheduler_session() -> None:
    request = make_multi_batch_request()
    scheduler = _scheduler()
    scheduler.scheduler_config.max_num_seqs = 2
    scheduler.scheduler_config.max_num_batched_tokens = 8
    scheduler.requests["session-0:1"] = SimpleNamespace(
        request_id="session-0:1",
        is_prefill_chunk=False,
        num_prompt_tokens=2,
        num_computed_tokens=2,
    )

    session = add_gr_session(scheduler, request)
    output = SimpleNamespace(num_scheduled_tokens={"session-0": 3, "session-0:1": 2})
    metadata = attach_gr_stage_metadata(scheduler, output, session)

    assert session.session_id == "session-0"
    assert session.request_ids == ("session-0", "session-0:1")
    assert session.batch_size == 2
    assert metadata is not None
    assert metadata.request_ids == session.request_ids
    assert metadata.num_prompt_tokens == 5
    assert metadata.num_prompt_tokens_by_request == (3, 2)


@pytest.mark.parametrize("native_block_ids", [False, True])
def test_caller_formed_batch_publishes_kv_blocks_in_member_order(
    native_block_ids: bool,
) -> None:
    scheduler = _scheduler()
    scheduler.scheduler_config.max_num_seqs = 2
    scheduler.scheduler_config.max_num_batched_tokens = 8
    scheduler.requests["session-0:1"] = SimpleNamespace(
        request_id="session-0:1",
        is_prefill_chunk=False,
        num_prompt_tokens=2,
        num_computed_tokens=2,
    )
    blocks = {
        "session-0": (SimpleNamespace(block_id=11), SimpleNamespace(block_id=12)),
        "session-0:1": (SimpleNamespace(block_id=21),),
    }
    touched: list[Any] = []

    def get_blocks(request_id: str) -> Any:
        groups = (blocks[request_id],)
        return KVCacheBlocks(groups) if native_block_ids else SimpleNamespace(blocks=groups)

    scheduler.kv_cache_manager = SimpleNamespace(
        get_blocks=get_blocks,
        block_pool=SimpleNamespace(touch=touched.extend, num_gpu_blocks=8),
        max_model_len=4096,
    )
    session = add_gr_session(scheduler, make_multi_batch_request())
    output = SimpleNamespace(num_scheduled_tokens={"session-0": 3, "session-0:1": 2})

    metadata = attach_gr_stage_metadata(scheduler, output, session)

    assert metadata is not None
    assert metadata.request_ids == ("session-0", "session-0:1")
    assert metadata.kv_blocks == ((11, 12), (21,))
    assert [block.block_id for block in touched] == [11, 12, 21]
    assert [[block.block_id for block in group] for group in session._kv_dispatch_leases[0]] == [
        [12, 11],
        [21],
    ]


def test_caller_formed_batch_rejects_effective_scheduling_budget() -> None:
    request = make_multi_batch_request()
    scheduler = _scheduler()
    scheduler.scheduler_config.max_num_seqs = 2
    scheduler.scheduler_config.max_num_batched_tokens = 64
    scheduler.max_num_scheduled_tokens = 4

    with pytest.raises(GRAdmissionError, match="effective native scheduling token budget"):
        add_gr_session(scheduler, request)

    assert not getattr(scheduler, "_gr_v1_sessions", {})


def test_caller_formed_batch_rejects_partial_native_schedule() -> None:
    request = make_multi_batch_request()
    scheduler = _scheduler()
    scheduler.scheduler_config.max_num_seqs = 2
    scheduler.scheduler_config.max_num_batched_tokens = 8
    scheduler.requests["session-0:1"] = SimpleNamespace(
        request_id="session-0:1",
        is_prefill_chunk=False,
        num_prompt_tokens=2,
        num_computed_tokens=2,
    )
    session = add_gr_session(scheduler, request)

    with pytest.raises(RuntimeError, match="split an atomic GR batch stage"):
        attach_gr_stage_metadata(scheduler, _scheduler_output(), session)


def test_session_rejects_native_chunked_prefill() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    scheduler.requests[session.session_id].is_prefill_chunk = True

    with pytest.raises(RuntimeError, match="cannot schedule partial Prefill"):
        attach_gr_stage_metadata(scheduler, _scheduler_output(), session)


def _failure_result(batch_id: str = "session-0") -> BeamBatchResult:
    return BeamBatchResult(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=batch_id,
        output=None,
        failure=BeamBatchFailure("test_failure", "expected", None),
    )


def test_prefill_decode_lookahead_uses_device_dependency_without_cpu_update() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    native_request = scheduler.requests[session.session_id]

    native_request.num_output_placeholders = 1
    prefill_output = _scheduler_output()
    prefill = attach_gr_stage_metadata(scheduler, prefill_output, session)
    setattr(prefill_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: prefill})

    assert prefill is not None
    assert prefill.kind == GRStageKind.PREFILL
    assert prefill.session_init is session.request
    assert prefill.input_device_state is None
    assert can_issue_gr_stage(scheduler, session)

    # Decode1 is issued before Prefill's Future is consumed by the CPU.
    native_request.num_output_placeholders = 2
    native_request.num_computed_tokens = 4
    decode_output = _scheduler_output()
    decode = attach_gr_stage_metadata(scheduler, decode_output, session)
    setattr(decode_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: decode})

    assert decode is not None
    assert decode.kind == GRStageKind.DECODE
    assert decode.session_init is None
    assert decode.input_device_state == prefill.output_device_state
    assert session.in_flight_stages == 2
    assert not can_issue_gr_stage(scheduler, session)

    # Reaping old Prefill output opens one queue slot while Decode1 remains live.
    native_request.num_output_placeholders = 1
    worker_result = GRWorkerResult(
        session_id=session.session_id,
        dispatch_id=prefill.dispatch_id,
        stage_index=0,
        produced_token_count=1,
        device_state=GRDeviceStateRef(session.session_id, 0),
        finished=False,
        owner_generation=session.owner_generation,
    )
    model_output = SimpleNamespace(**{GR_WORKER_RESULT_ATTR: {session.session_id: worker_result}})
    update_gr_from_output(scheduler, session, prefill_output, model_output)

    assert session.completed_stages == 1
    assert session.in_flight_stages == 1
    assert can_issue_gr_stage(scheduler, session)


def test_early_finish_stops_issue_and_drains_already_queued_stage() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    native_request = scheduler.requests[session.session_id]
    outputs = []
    for placeholders in (1, 2):
        native_request.num_output_placeholders = placeholders
        native_request.num_computed_tokens = native_request.num_prompt_tokens + placeholders - 1
        output = _scheduler_output()
        metadata = attach_gr_stage_metadata(scheduler, output, session)
        assert metadata is not None
        setattr(output, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
        outputs.append((output, metadata))

    terminal_result = _failure_result()
    first_result = GRWorkerResult(
        session_id=session.session_id,
        dispatch_id=outputs[0][1].dispatch_id,
        stage_index=0,
        produced_token_count=1,
        device_state=GRDeviceStateRef(session.session_id, 0),
        finished=True,
        terminal_result=terminal_result,
        owner_generation=session.owner_generation,
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

    queued_result = GRWorkerResult(
        session_id=session.session_id,
        dispatch_id=outputs[1][1].dispatch_id,
        stage_index=1,
        produced_token_count=0,
        device_state=GRDeviceStateRef(session.session_id, 1),
        finished=True,
        resources_released=True,
        owner_generation=session.owner_generation,
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
    assert _gr_session_ready_to_finalize(session)
    assert session.session_id in scheduler._gr_v1_sessions

    native_request = scheduler.requests[session.session_id]
    native_request.client_index = 0
    native_request.is_finished = lambda: False
    finished: list[tuple[str, object]] = []

    def finish_requests(request_id: str, status: object) -> list[Any]:
        finished.append((request_id, status))
        return [native_request]

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


def test_unexpected_native_termination_drains_before_delivering_failure() -> None:
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
            metadata = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR)["session-0"]
            if metadata.stage_index != 0:
                return {0: EngineCoreOutputs(outputs=[])}

            finished_request = self.requests.pop("session-0")
            _sync_gr_sessions_after_native_finish(
                self,
                [finished_request],
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
    request = make_batch_request()
    request.items[0].request.client_index = 0
    session = add_gr_session(scheduler, request)
    outputs = []
    for placeholders in (1, 2):
        scheduler.requests[session.session_id].num_output_placeholders = placeholders
        scheduler.requests[session.session_id].num_computed_tokens = 3 + placeholders - 1
        scheduler_output = _scheduler_output()
        metadata = attach_gr_stage_metadata(scheduler, scheduler_output, session)
        assert metadata is not None
        setattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
        outputs.append(scheduler_output)

    prefill = getattr(outputs[0], GR_STAGE_OUTPUT_ATTR)[session.session_id]
    worker_result = GRWorkerResult(
        session.session_id,
        prefill.dispatch_id,
        prefill.stage_index,
        1,
        prefill.output_device_state,
        False,
        owner_generation=session.owner_generation,
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

    successor = getattr(outputs[1], GR_STAGE_OUTPUT_ATTR)[session.session_id]
    drained = GRWorkerResult(
        session.session_id,
        successor.dispatch_id,
        successor.stage_index,
        0,
        successor.output_device_state,
        True,
        resources_released=True,
        owner_generation=session.owner_generation,
    )
    terminal_result = scheduler.update_from_output(
        outputs[1], SimpleNamespace(**{GR_WORKER_RESULT_ATTR: {session.session_id: drained}})
    )

    assert session.resource_status is GRResourceStatus.CLOSED
    assert session.session_id not in scheduler._gr_v1_sessions
    final_output = terminal_result[0].outputs[0]
    assert final_output.finished
    failure = final_output.kv_transfer_params["gr_batch_result"].failure
    assert failure is not None
    assert failure.code == "native_termination_without_gr_result"
    assert f"Native batch member {session.session_id} terminated" in failure.message


def test_partial_prefill_rejected_before_dispatch_or_worker_allocation() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    scheduler.requests[session.session_id].is_prefill_chunk = True
    with pytest.raises(RuntimeError, match="partial Prefill"):
        attach_gr_stage_metadata(scheduler, _scheduler_output(), session)
    assert not session._in_flight_dispatches
    assert not session.worker_resources_allocated
    assert session.issued_stages == 0


def test_recompute_is_rejected_after_beam_initialization() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    native_request = scheduler.requests[session.session_id]

    native_request.num_output_placeholders = 1
    prefill_output = _scheduler_output()
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
                    session.session_id: GRWorkerResult(
                        session_id=session.session_id,
                        dispatch_id=prefill.dispatch_id,
                        stage_index=prefill.stage_index,
                        produced_token_count=1,
                        device_state=prefill.output_device_state,
                        finished=False,
                        owner_generation=session.owner_generation,
                    )
                }
            }
        ),
    )

    latest_device_state = session.latest_device_state
    native_request.is_prefill_chunk = True
    recompute_output = _scheduler_output()
    with pytest.raises(RuntimeError, match="recompute"):
        attach_gr_stage_metadata(scheduler, recompute_output, session)
    assert session.issued_stages == 1
    assert session.latest_device_state == latest_device_state
    assert session.completed_stages == 1
    assert not session._in_flight_dispatches


def test_backend_configuration_error_uses_request_error_channel() -> None:
    request = make_batch_request()
    errors: list[object] = []
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={
                "vllm_gr": {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
            },
            scheduler_config=SimpleNamespace(async_scheduling=False),
        ),
        batch_queue=object(),
        batch_queue_size=2,
        _handle_request_preproc_error=lambda native_request: errors.append(native_request),
    )

    _handle_submit_gr_session(engine, request)

    assert errors == [request.items[0].request]


def test_gr_submission_disables_native_stop_ownership() -> None:
    from vllm.v1.request import Request

    request = make_batch_request()
    source = request.items[0].request
    assert source.sampling_params is not None
    source.sampling_params.stop_token_ids = [2]
    added: list[Any] = []

    def preprocess(owned: EngineCoreRequest) -> tuple[Request]:
        assert owned is not source
        assert owned.sampling_params is not source.sampling_params
        return (Request.from_engine_core_request(owned, block_hasher=None),)

    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={
                "vllm_gr": {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
            },
            scheduler_config=SimpleNamespace(async_scheduling=True),
        ),
        batch_queue=object(),
        batch_queue_size=2,
        preprocess_add_request=preprocess,
        add_request=lambda processed: added.append(processed),
    )

    _handle_submit_gr_session(engine, request)

    assert len(added) == 1
    assert added[0].sampling_params.ignore_eos
    assert added[0].sampling_params.stop_token_ids == []
    assert source.sampling_params.stop_token_ids == [2]


def test_forced_prefix_reset_is_rejected_while_gr_session_is_active() -> None:
    class ResetScheduler:
        def __init__(self) -> None:
            self._gr_v1_sessions = {
                "session-0": SimpleNamespace(
                    status=GRSessionStatus.RUNNING,
                    resource_status=GRResourceStatus.RUNNING,
                    in_flight_stages=1,
                )
            }
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


@pytest.mark.l1
def test_prefix_reset_after_drain_preserves_waiting_gr_sessions() -> None:
    class ResetScheduler:
        def __init__(self) -> None:
            self._gr_v1_sessions = {
                "queued": SimpleNamespace(
                    status=GRSessionStatus.WAITING,
                    resource_status=GRResourceStatus.WAITING,
                    in_flight_stages=0,
                )
            }
            self.calls: list[bool] = []

        def reset_prefix_cache(self, reset_running_requests: bool = False) -> bool:
            self.calls.append(reset_running_requests)
            return True

    _patch_scheduler_reset_prefix_cache(ResetScheduler)
    scheduler = ResetScheduler()
    queued = scheduler._gr_v1_sessions["queued"]
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert scheduler.calls == [True]
    assert scheduler._gr_v1_sessions["queued"] is queued


def test_native_abort_all_result_cleans_unscheduled_gr_session() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())

    _sync_gr_sessions_after_native_finish(
        scheduler,
        [scheduler.requests[session.session_id]],
        SimpleNamespace(name="FINISHED_ABORTED"),
    )

    assert session.status == GRSessionStatus.ABORTED
    assert session.resource_status is GRResourceStatus.CLOSED
    assert session.session_id not in scheduler._gr_v1_sessions


def test_native_shutdown_preserves_none_abort_selector() -> None:
    scheduler = _scheduler()

    request_ids = _expand_gr_abort_request_ids(
        scheduler,
        None,
        SimpleNamespace(name="FINISHED_ABORTED"),
    )

    assert request_ids is None


def test_native_abort_overrides_a_pending_gr_terminal_result() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    session.status = GRSessionStatus.FINISHING
    session.terminal_result_pending = True
    session._pending_terminal_result = _failure_result()

    _sync_gr_sessions_after_native_finish(
        scheduler,
        [scheduler.requests[session.session_id]],
        SimpleNamespace(name="FINISHED_ABORTED"),
    )

    assert session.status == GRSessionStatus.ABORTED
    assert not session.terminal_result_pending
    assert session.resource_status is GRResourceStatus.CLOSED
    assert session.session_id not in scheduler._gr_v1_sessions


def test_internal_gr_finalization_preserves_the_pending_terminal_result() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    session.status = GRSessionStatus.FINISHING
    session.terminal_result_pending = True
    session._pending_terminal_result = _failure_result()
    setattr(scheduler, _GR_FINALIZING_IDS_ATTR, {session.session_id})

    _sync_gr_sessions_after_native_finish(
        scheduler,
        [scheduler.requests[session.session_id]],
        SimpleNamespace(name="FINISHED_ERROR"),
    )

    assert session.status == GRSessionStatus.FINISHING
    assert session.terminal_result_pending
    assert session.session_id in scheduler._gr_v1_sessions


def test_abort_adapters_retry_internal_id_after_frontend_routing_disappears() -> None:
    internal_id = "internal-randomized-id"
    sync_calls = []
    async_calls = []
    frontend = SimpleNamespace(abort_requests=lambda ids, internal: [])
    core = SimpleNamespace(
        abort_requests=lambda ids: sync_calls.append(ids),
        call_utility=lambda method, sid: True,
    )

    async def abort(ids: list[str]) -> None:
        async_calls.append(ids)

    async def retired(method: str, sid: str) -> bool:
        return True

    async_core = SimpleNamespace(abort_requests_async=abort, call_utility_async=retired)
    for _ in range(2):
        abort_gr_session_sync(
            SimpleNamespace(output_processor=frontend, engine_core=core), internal_id
        )
        asyncio.run(
            abort_gr_session_async(
                SimpleNamespace(output_processor=frontend, engine_core=async_core), internal_id
            )
        )
    assert sync_calls == async_calls == [[internal_id], [internal_id]]


def test_terminal_result_restores_untyped_multiprocess_payload() -> None:
    request = make_batch_request()
    expected = _failure_result()
    wire_value = msgspec.to_builtins(expected)
    assert isinstance(wire_value, list)

    output = SimpleNamespace(kv_transfer_params={"gr_batch_result": wire_value})

    assert _terminal_result(request, output) == expected


def test_sync_wait_uses_the_native_external_output_id() -> None:
    request = make_batch_request()
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
    request = make_batch_request()
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
    request = make_batch_request()
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


def test_active_prefill_validates_worker_receipt_identity() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    scheduled = _scheduler_output()
    metadata = attach_gr_stage_metadata(scheduler, scheduled, session)
    assert metadata is not None
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
    result = GRWorkerResult(
        session.session_id,
        metadata.dispatch_id + 1,
        metadata.stage_index,
        0,
        metadata.output_device_state,
        False,
        owner_generation=session.owner_generation,
    )
    output = SimpleNamespace()
    setattr(output, GR_WORKER_RESULT_ATTR, {session.session_id: result})
    completion = update_gr_from_output(scheduler, session, scheduled, output)
    assert completion.finished
    assert "different dispatch" in completion.error
    assert session.completed_stages == 1
    assert session.worker_resources_allocated


@pytest.mark.parametrize("has_mapping", [False, True])
def test_missing_worker_receipt_fails_without_confirming_tokens(has_mapping: bool) -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    scheduled = _scheduler_output()
    metadata = attach_gr_stage_metadata(scheduler, scheduled, session)
    assert metadata is not None
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
    output = SimpleNamespace(**({GR_WORKER_RESULT_ATTR: {}} if has_mapping else {}))

    completion = update_gr_from_output(scheduler, session, scheduled, output)

    assert completion.finished and "omitted GRWorkerResult" in (completion.error or "")
    assert not _gr_session_ready_to_finalize(session)
    assert not session._in_flight_dispatches
    assert session.resource_status is GRResourceStatus.DRAINING
    assert session._pending_terminal_result is not None
    assert session._pending_terminal_result.failure is not None
    assert session._pending_terminal_result.failure.code == "worker_protocol_error"


def test_active_prefill_rejects_missing_token_without_terminal() -> None:
    scheduler = _scheduler()
    session = add_gr_session(scheduler, make_batch_request())
    scheduled = _scheduler_output()
    metadata = attach_gr_stage_metadata(scheduler, scheduled, session)
    assert metadata is not None
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {session.session_id: metadata})
    output = SimpleNamespace()
    setattr(
        output,
        GR_WORKER_RESULT_ATTR,
        {
            session.session_id: GRWorkerResult(
                session.session_id,
                metadata.dispatch_id,
                metadata.stage_index,
                0,
                metadata.output_device_state,
                False,
                owner_generation=session.owner_generation,
            )
        },
    )
    completion = update_gr_from_output(scheduler, session, scheduled, output)
    assert completion.finished and "no token" in completion.error
    assert session.completed_stages == 1
    assert session.worker_resources_allocated
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
    session = add_gr_session(scheduler, make_batch_request())
    scheduled = _scheduler_output()
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


@pytest.mark.l1
def test_hybrid_private_blocks_reserve_capacity_and_wait_for_retirement() -> None:
    scheduler = _scheduler()
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.max_num_scheduled_tokens = 16
    pool = BlockPool(num_gpu_blocks=9, enable_caching=False, hash_block_size=16)

    def native_allocate(request: Any, num_new_tokens: int, **kwargs: Any) -> Any:
        assert kwargs["reserved_blocks"] == 5
        assert kwargs["num_lookahead_tokens"] == 2
        assert pool.get_num_free_blocks() - kwargs["reserved_blocks"] >= 2
        return pool.get_new_blocks(2)

    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=pool, max_model_len=100, allocate_slots=native_allocate
    )
    session = add_gr_session(scheduler, make_batch_request(3))
    session.hybrid_beam_block_counts = ((3, 2),)
    request = scheduler.requests[session.session_id]
    request.num_computed_tokens = 0
    free = pool.get_num_free_blocks()
    with gr_admission(scheduler):
        native = scheduler.kv_cache_manager.allocate_slots(request, request.num_prompt_tokens)
    assert tuple(map(len, session._hybrid_beam_blocks[0])) == (3, 2)
    assert pool.get_num_free_blocks() == free - 7

    session.worker_resources_allocated = True
    session.status = GRSessionStatus.ABORTED
    _drain_resources(scheduler, session)
    pool.free_blocks(native)
    assert pool.get_num_free_blocks() == free - 5

    scheduled = SimpleNamespace(total_num_scheduled_tokens=0, finished_req_ids=set())
    attach_gr_retirement(scheduler, scheduled)
    metadata = getattr(scheduled, GR_RETIRE_OUTPUT_ATTR)
    output = SimpleNamespace()
    setattr(
        output,
        GR_RETIRE_RESULT_ATTR,
        GRRetireResult(metadata.session_id, metadata.owner_generation, metadata.retire_id, True),
    )
    assert consume_gr_retirement(scheduler, scheduled, output) is session
    assert consume_gr_retirement(scheduler, scheduled, output) is None
    assert pool.get_num_free_blocks() == free


@pytest.mark.l1
def test_hybrid_checkpoint_holds_native_blocks_until_cancel_drains() -> None:
    from vllm_gr.v1.core.sched.hybrid_prefill import (
        attach_hybrid_prefill_chunk,
        complete_hybrid_prefill_chunk,
        hybrid_prefill_scope,
    )

    scheduler = _scheduler()
    scheduler.need_mamba_block_aligned_split = True
    pool = BlockPool(num_gpu_blocks=10, enable_caching=False, hash_block_size=16)
    (block,) = pool.get_new_blocks(1)
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=pool, max_model_len=4096, get_blocks=lambda _: KVCacheBlocks(((block,),))
    )
    session = add_gr_session(scheduler, make_batch_request(3))
    session.hybrid_beam_block_counts = ((2,),)
    request = scheduler.requests[session.session_id]
    request.num_computed_tokens = 0
    request.is_prefill_chunk = True
    running = scheduler.running = [request]

    def native_split(request, num_new_tokens, *_):
        return min(num_new_tokens, 2)

    scheduler._mamba_block_aligned_split = native_split
    with hybrid_prefill_scope(scheduler, session) as member:
        assert member == session.session_id
        assert scheduler._mamba_block_aligned_split(request, 3) == 2
    assert scheduler.running is running
    assert scheduler._mamba_block_aligned_split is native_split

    scheduled = _scheduler_output()
    attach_hybrid_prefill_chunk(scheduler, scheduled, session.session_id)
    assert block.ref_cnt == 2
    pool.free_blocks((block,))
    session.status = GRSessionStatus.ABORTED
    _drain_resources(scheduler, session)
    assert block.ref_cnt == 1
    assert session.resource_status is GRResourceStatus.DRAINING
    complete_hybrid_prefill_chunk(scheduler, session)
    assert block.ref_cnt == 0
    assert session.resource_status is GRResourceStatus.CLOSED


@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize("num_leases", [1, 2])
@pytest.mark.parametrize("native_first", [False, True])
@pytest.mark.parametrize("shared_prefix", [False, True])
@pytest.mark.parametrize("status", [GRSessionStatus.ABORTED, GRSessionStatus.FINISHING])
@pytest.mark.parametrize("native_block_ids", [False, True])
@pytest.mark.parametrize("cache_pattern", ["uncached", "cached", "mixed"])
def test_dispatch_lease_preserves_native_eviction_order(
    num_groups: int,
    num_leases: int,
    native_first: bool,
    shared_prefix: bool,
    status: GRSessionStatus,
    native_block_ids: bool,
    cache_pattern: str,
) -> None:
    def resources() -> tuple[BlockPool, list[Any], list[Any]]:
        pool = BlockPool(num_gpu_blocks=3 * num_groups + 2, enable_caching=True, hash_block_size=16)
        blocks = pool.get_new_blocks(3 * num_groups)
        # Physical block IDs deliberately differ from prompt order.
        groups = [
            [blocks[start + 2], blocks[start], blocks[start + 1]]
            for start in range(0, len(blocks), 3)
        ]
        for group_id, group in enumerate(groups):
            for index, block in enumerate(group):
                if cache_pattern == "cached" or (cache_pattern == "mixed" and index == 0):
                    block_hash = make_block_hash_with_group_id(
                        BlockHash(block.block_id.to_bytes(32, "big")), group_id
                    )
                    block.set_block_hash(block_hash)
                    pool.cached_block_hash_to_block.insert(block_hash, block)
        if shared_prefix:
            pool.touch([groups[0][0]])
        managers = [
            SimpleNamespace(
                req_to_blocks={"session-0": group},
                num_cached_block={},
                _partial_hit_reqs={},
                block_pool=pool,
            )
            for group in groups
        ]
        return pool, groups, managers

    def native_free(managers: list[Any]) -> None:
        for manager in managers:
            manager.pop_blocks_for_free = MethodType(
                SingleTypeKVCacheManager.pop_blocks_for_free, manager
            )
            SingleTypeKVCacheManager.free(manager, "session-0")

    reference, _, reference_managers = resources()
    native_free(reference_managers)
    expected = [block.block_id for block in reference.free_block_queue.get_all_free_blocks()]
    pool, groups, managers = resources()
    scheduler = _scheduler()
    kv = KVCacheBlocks(tuple(groups)) if native_block_ids else SimpleNamespace(blocks=groups)
    scheduler.kv_cache_manager = SimpleNamespace(
        get_blocks=lambda _: kv, block_pool=pool, max_model_len=4096
    )
    session = add_gr_session(scheduler, make_batch_request())
    scheduled = []
    for _ in range(num_leases):
        output = _scheduler_output()
        refs_before = [block.ref_cnt for group in groups for block in group]
        m = attach_gr_stage_metadata(scheduler, output, session)
        assert m is not None
        expected_blocks = (
            tuple(tuple(block.block_id for block in group) for group in groups)
            if num_groups == 1
            else None
        )
        assert m.kv_blocks == expected_blocks
        assert [block.ref_cnt for group in groups for block in group] == [
            count + 1 for count in refs_before
        ]
        setattr(output, GR_STAGE_OUTPUT_ATTR, {session.session_id: m})
        scheduled.append(output)
        scheduler.requests[session.session_id].num_computed_tokens += 1
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
    from vllm_gr.v1.core.sched.beam_async_scheduler import (
        _install_gr_prompt_cache_guard,
    )

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


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "type",
        "identity",
        "score",
        "positive_infinity",
        "negative_infinity",
        "missing_score",
        "unexpected_score",
        "unrequested_parent",
        "finish_reason",
        "length",
        "both",
        "negative_token",
        "negative_parent",
        "parent_width",
        "parent_length",
        "stop",
        "protocol",
        "empty_failure",
        "neither",
    ],
)
def test_terminal_transport_validates_inproc_and_wire_equally(wire: bool, fault: str) -> None:
    from vllm_gr.v1.beam.batch_contracts import (
        BeamBatchOutput,
        BeamOutputSequence,
        BeamRequestOutput,
    )

    request = make_batch_request()
    seq = BeamOutputSequence((1, 2, 3), -1.0, (0, 0, 0), "length", None)
    valid_sequence = seq
    if fault == "type":
        seq = msgspec.structs.replace(seq, token_ids=(True, 2, 3))
    elif fault == "score":
        seq = msgspec.structs.replace(seq, cumulative_logprob=float("nan"))
    elif fault in ("positive_infinity", "negative_infinity"):
        score = float("inf") if fault == "positive_infinity" else -float("inf")
        seq = msgspec.structs.replace(seq, cumulative_logprob=score)
    elif fault == "missing_score":
        seq = msgspec.structs.replace(seq, cumulative_logprob=None)
    elif fault == "unexpected_score":
        request = msgspec.structs.replace(
            request,
            output_options=msgspec.structs.replace(
                request.output_options, include_cumulative_logprob=False
            ),
        )
        valid_sequence = msgspec.structs.replace(seq, cumulative_logprob=None)
    elif fault == "unrequested_parent":
        request = msgspec.structs.replace(
            request,
            output_options=msgspec.structs.replace(
                request.output_options, include_parent_beam_indices=False
            ),
        )
        valid_sequence = msgspec.structs.replace(seq, parent_beam_indices=())
    elif fault == "finish_reason":
        seq = msgspec.structs.replace(seq, finish_reason="stop")
    elif fault == "length":
        seq = msgspec.structs.replace(seq, token_ids=(1, 2))
    elif fault == "negative_token":
        seq = msgspec.structs.replace(seq, token_ids=(-1, 2, 3))
    elif fault in ("negative_parent", "parent_width", "parent_length"):
        parents = {"negative_parent": (-1, 0, 0), "parent_width": (4, 0, 0), "parent_length": (0,)}
        seq = msgspec.structs.replace(seq, parent_beam_indices=parents[fault])
    elif fault == "stop":
        seq = msgspec.structs.replace(seq, stop_token_id=2)
    item = BeamRequestOutput(
        0,
        "other" if fault == "identity" else request.batch_id,
        (valid_sequence, seq),
    )
    result = BeamBatchResult(
        BEAM_BATCH_PROTOCOL_VERSION,
        request.batch_id,
        BeamBatchOutput((item,)),
        _failure_result().failure if fault == "both" else None,
    )
    if fault == "protocol":
        result = msgspec.structs.replace(result, protocol_version=BEAM_BATCH_PROTOCOL_VERSION + 1)
    elif fault == "empty_failure":
        result = msgspec.structs.replace(
            result,
            output=None,
            failure=msgspec.structs.replace(_failure_result().failure, code=" "),
        )
    elif fault == "neither":
        result = msgspec.structs.replace(result, output=None)
    payload = msgspec.to_builtins(result) if wire else result
    with pytest.raises((ValueError, TypeError)):
        _terminal_result(request, SimpleNamespace(kv_transfer_params={"gr_batch_result": payload}))


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 3])
@pytest.mark.parametrize(
    "scores,parents", [(False, False), (False, True), (True, False), (True, True)]
)
def test_terminal_transport_accepts_requested_optional_fields(
    wire: bool, budget: int, scores: bool, parents: bool
) -> None:
    from vllm_gr.v1.beam.batch_contracts import (
        BeamBatchOutput,
        BeamOutputSequence,
        BeamRequestOutput,
    )

    request = make_batch_request(max_tokens=budget)
    request = msgspec.structs.replace(
        request,
        output_options=msgspec.structs.replace(
            request.output_options,
            include_cumulative_logprob=scores,
            include_parent_beam_indices=parents,
        ),
    )
    sequence = BeamOutputSequence(
        tuple(range(budget)),
        -1.0 if scores else None,
        (0,) * budget if parents else (),
        "length",
        None,
    )
    result = BeamBatchResult(
        request.protocol_version,
        request.batch_id,
        BeamBatchOutput((BeamRequestOutput(0, request.batch_id, (sequence,)),)),
        None,
    )
    payload = msgspec.to_builtins(result) if wire else result
    assert (
        _terminal_result(request, SimpleNamespace(kv_transfer_params={"gr_batch_result": payload}))
        == result
    )


pytestmark = pytest.mark.cpu_test
