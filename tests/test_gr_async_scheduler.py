# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from types import SimpleNamespace
from unittest import mock

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_gr.v1.core.sched.async_scheduler import GRAsyncScheduler
from vllm_gr.v1.engine.types import GR_REQUEST_METADATA_KEY, GRRequestMetadata
from vllm_gr.v1.metrics.stats import GRRequestCounters


def _metadata() -> GRRequestMetadata:
    return GRRequestMetadata(
        request_id="gr-0",
        beam_width=3,
        max_tokens=3,
        prefix_len=4,
        continue_token_id=7,
        finish_token_id=8,
        num_return_sequences=3,
    )


def _request(step: int = 0):
    return SimpleNamespace(
        request_id="gr-0",
        is_gr_request=True,
        gr_metadata=_metadata(),
        gr_step=step,
        gr_issued_steps=step,
        gr_completed_steps=step,
        gr_bundled_inflight=False,
        gr_parent_ids=[0, 1, 2],
        gr_beam_suffix=[[11], [12], [13]] if step else [],
        gr_counters=GRRequestCounters(),
        gr_persistent_runner_state=False,
    )


def test_add_request_reads_metadata_once() -> None:
    metadata = _metadata()
    request = SimpleNamespace(
        request_id="gr-0",
        num_prompt_tokens=4,
        sampling_params=SimpleNamespace(
            extra_args={
                GR_REQUEST_METADATA_KEY: {
                    field: getattr(metadata, field) for field in metadata.__struct_fields__
                }
            }
        ),
    )
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {}

    with mock.patch.object(AsyncScheduler, "add_request") as add_request:
        scheduler.add_request(request)

    add_request.assert_called_once_with(request)
    assert request.is_gr_request is True
    assert request.gr_step == 0
    assert request.gr_issued_steps == 0
    assert request.gr_completed_steps == 0
    assert request.gr_counters.native_request_created_per_session == 1
    assert request.gr_counters.scheduler_add_request_per_session == 1


def test_schedule_keeps_one_logical_token_and_dispatches_beam_rows() -> None:
    request = _request(step=1)
    output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        total_num_scheduled_tokens=1,
    )
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}

    with mock.patch.object(AsyncScheduler, "schedule", return_value=output):
        result = scheduler.schedule()

    assert result.num_scheduled_tokens == {"gr-0": 3}
    assert result.total_num_scheduled_tokens == 3
    assert result.beam_data["gr-0"]["next_token_ids"] == [11, 12, 13]
    assert result.beam_data["gr-0"]["decode_step"] == 0
    assert result.beam_data["gr-0"]["generation_index"] == 1
    assert request.gr_issued_steps == 2
    assert request.gr_counters.decode_scheduler_calls == 1


def test_schedule_marks_only_the_final_prefill_chunk() -> None:
    request = _request(step=0)
    request.num_computed_tokens = 2
    output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 2},
        total_num_scheduled_tokens=2,
    )
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}

    with mock.patch.object(AsyncScheduler, "schedule", return_value=output):
        first = scheduler.schedule()

    assert first.beam_data["gr-0"]["is_last_prefill"] is False
    assert first.beam_data["gr-0"]["generation_index"] is None
    assert request.gr_issued_steps == 0

    request.num_computed_tokens = request.gr_metadata.prefix_len
    output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 2},
        total_num_scheduled_tokens=2,
    )
    with mock.patch.object(AsyncScheduler, "schedule", return_value=output):
        final = scheduler.schedule()

    assert final.beam_data["gr-0"]["is_last_prefill"] is True
    assert final.beam_data["gr-0"]["generation_index"] == 0
    assert request.gr_issued_steps == 1


def test_update_from_output_appends_one_control_token() -> None:
    request = _request(step=1)
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 3},
        total_num_scheduled_tokens=3,
        beam_data={"gr-0": {"is_beam_decode": True, "generation_index": 1}},
    )
    payload = struct.pack("<6i3f", 21, 22, 23, 2, 0, 1, -1.0, -2.0, -3.0)
    decision = {
        "buffer_format": "beam_result_i32_f32_v1",
        "shape": [1, 3],
        "buffer": payload,
        "is_last_step": False,
    }
    model_output = SimpleNamespace(
        beam_search_result=decision,
        req_id_to_index={"gr-0": 0},
        sampled_token_ids=[[99]],
    )

    with mock.patch.object(Scheduler, "update_from_output", return_value={}):
        result = scheduler.update_from_output(scheduler_output, model_output)

    assert result == {}
    assert scheduler_output.num_scheduled_tokens == {"gr-0": 1}
    assert scheduler_output.total_num_scheduled_tokens == 1
    assert model_output.sampled_token_ids == [[7]]
    assert request.gr_step == 2
    assert request.gr_completed_steps == 2
    assert request.gr_parent_ids == [2, 0, 1]
    assert request.gr_beam_suffix == [[13, 21], [11, 22], [12, 23]]


def test_runner_state_intermediate_output_is_control_only() -> None:
    request = _request(step=1)
    request.gr_persistent_runner_state = True
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 3},
        total_num_scheduled_tokens=3,
        beam_data={"gr-0": {"is_beam_decode": True, "generation_index": 1}},
    )
    model_output = SimpleNamespace(
        beam_search_result=None,
        req_id_to_index={"gr-0": 0},
        sampled_token_ids=[[99]],
    )

    with mock.patch.object(Scheduler, "update_from_output", return_value={}):
        result = scheduler.update_from_output(scheduler_output, model_output)

    assert result == {}
    assert scheduler_output.num_scheduled_tokens == {"gr-0": 1}
    assert model_output.sampled_token_ids == [[7]]
    assert request.gr_step == 2
    assert request.gr_completed_steps == 2
    assert request.gr_counters.intermediate_beam_state_d2h_bytes == 0
    assert request.gr_beam_suffix == [[11], [12], [13]]


def test_persistent_gr_adds_placeholder_without_waiting_for_output() -> None:
    request = _request(step=0)
    request.gr_persistent_runner_state = True
    request.is_prefill_chunk = False
    request.num_output_placeholders = 0
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 4},
        scheduled_spec_decode_tokens={},
        pending_structured_output_tokens=False,
    )

    with mock.patch.object(Scheduler, "_update_after_schedule") as update:
        scheduler._update_after_schedule(output)

    update.assert_called_once_with(scheduler, output)
    assert request.num_output_placeholders == 1


def test_persistent_gr_does_not_add_placeholder_after_last_issue() -> None:
    request = _request(step=3)
    request.gr_persistent_runner_state = True
    request.is_prefill_chunk = False
    request.num_output_placeholders = 1
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        scheduled_spec_decode_tokens={},
        pending_structured_output_tokens=False,
    )

    with mock.patch.object(Scheduler, "_update_after_schedule"):
        scheduler._update_after_schedule(output)

    assert request.num_output_placeholders == 1


def test_persistent_gr_consumes_output_placeholder() -> None:
    request = _request(step=1)
    request.gr_persistent_runner_state = True
    request.async_tokens_to_discard = 0
    request.num_output_placeholders = 2
    request.num_computed_tokens = 5
    request.status = RequestStatus.RUNNING
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.kv_cache_manager = mock.Mock()

    with mock.patch.object(
        Scheduler,
        "_update_request_with_output",
        return_value=([7], False),
    ):
        token_ids, stopped = scheduler._update_request_with_output(request, [7])

    assert token_ids == [7]
    assert stopped is False
    assert request.num_output_placeholders == 1
    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 4)


def test_lookahead_schedule_uses_issue_cursor_not_completed_cursor() -> None:
    request = _request(step=0)
    request.gr_persistent_runner_state = True
    request.gr_issued_steps = 1
    output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        total_num_scheduled_tokens=1,
    )
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}

    with mock.patch.object(AsyncScheduler, "schedule", return_value=output):
        result = scheduler.schedule()

    assert result.beam_data["gr-0"]["is_beam_decode"] is True
    assert result.beam_data["gr-0"]["generation_index"] == 1
    assert result.beam_data["gr-0"]["decode_step"] == 0
    assert request.gr_step == 0
    assert request.gr_issued_steps == 2


def test_out_of_order_gr_output_is_rejected() -> None:
    request = _request(step=0)
    request.gr_persistent_runner_state = True
    request.gr_issued_steps = 2
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 3},
        total_num_scheduled_tokens=3,
        beam_data={"gr-0": {"is_beam_decode": True, "generation_index": 1}},
    )
    model_output = SimpleNamespace(
        beam_search_result=None,
        req_id_to_index={"gr-0": 0},
        sampled_token_ids=[[99]],
    )

    with (
        mock.patch.object(Scheduler, "update_from_output", return_value={}),
        pytest.raises(RuntimeError, match="output order mismatch"),
    ):
        scheduler.update_from_output(scheduler_output, model_output)


def test_one_token_gr_attaches_final_result_after_native_removal() -> None:
    request = _request(step=0)
    request.gr_metadata = GRRequestMetadata(
        request_id="gr-0",
        beam_width=3,
        max_tokens=1,
        prefix_len=4,
        continue_token_id=7,
        finish_token_id=8,
        num_return_sequences=3,
    )
    request.gr_persistent_runner_state = True
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        total_num_scheduled_tokens=1,
        beam_data={
            "gr-0": {
                "is_beam_decode": False,
                "generation_index": 0,
            }
        },
    )
    decision = {
        "buffer_format": "gr_final_i32_f32_v1",
        "buffer": b"final",
        "is_last_step": True,
        "gr_state_created": 1,
        "gr_state_freed": 1,
    }
    model_output = SimpleNamespace(
        beam_search_result=decision,
        req_id_to_index={"gr-0": 0},
        sampled_token_ids=[[99]],
    )
    request_output = SimpleNamespace(request_id="gr-0", kv_transfer_params=None)
    core_outputs = {0: SimpleNamespace(outputs=[request_output])}

    def remove_finished(*_args):
        scheduler.requests.clear()
        return core_outputs

    with mock.patch.object(
        Scheduler,
        "update_from_output",
        side_effect=remove_finished,
    ):
        result = scheduler.update_from_output(scheduler_output, model_output)

    assert result is core_outputs
    assert request_output.kv_transfer_params == {"beam_search_result": decision}
    assert model_output.sampled_token_ids == [[8]]
    assert request.gr_completed_steps == 1


def test_bundled_session_hides_inflight_request_from_batch_queue() -> None:
    request = _request(step=0)
    request.gr_persistent_runner_state = True
    request.gr_bundled_inflight = True
    request.gr_issued_steps = request.gr_metadata.max_tokens
    request.gr_completed_steps = 0
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {request.request_id: request}

    with mock.patch.object(AsyncScheduler, "has_requests", return_value=True):
        assert scheduler.has_requests() is False

    request.gr_completed_steps = request.gr_metadata.max_tokens
    with mock.patch.object(AsyncScheduler, "has_requests", return_value=True):
        assert scheduler.has_requests() is True


def test_only_final_prefill_of_isolated_runner_state_request_can_bundle() -> None:
    request = _request(step=0)
    request.gr_persistent_runner_state = True
    request.gr_issued_steps = 1
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {request.request_id: request}
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 4}
    output.total_num_scheduled_tokens = 4
    output.beam_data = {
        request.request_id: {
            "is_constraint_prefill": True,
            "is_last_prefill": True,
        }
    }

    assert scheduler._can_bundle_session(output) is True
    output.beam_data[request.request_id]["is_last_prefill"] = False
    assert scheduler._can_bundle_session(output) is False


def test_bundled_update_reconciles_every_stage_but_returns_only_final() -> None:
    scheduler = object.__new__(GRAsyncScheduler)
    request = _request(step=0)
    request.gr_persistent_runner_state = True
    request.gr_bundled_inflight = True
    scheduler.requests = {request.request_id: request}
    first = SimpleNamespace(num_scheduled_tokens={request.request_id: 4})
    decode_1 = SimpleNamespace(num_scheduled_tokens={request.request_id: 3})
    decode_2 = SimpleNamespace(num_scheduled_tokens={request.request_id: 3})
    first.gr_followup_outputs = [decode_1, decode_2]
    final_model_output = SimpleNamespace(sampled_token_ids=[[8]])
    calls = []

    def update_one(scheduler_output, model_output):
        calls.append((scheduler_output, model_output))
        return {len(calls): scheduler_output}

    scheduler._update_single_output = update_one
    result = scheduler.update_from_output(first, final_model_output)

    assert [item[0] for item in calls] == [first, decode_1, decode_2]
    assert calls[0][1].sampled_token_ids == [[7]]
    assert calls[1][1].sampled_token_ids == [[7]]
    assert calls[2][1] is final_model_output
    assert result == {3: decode_2}
    assert request.gr_bundled_inflight is False
