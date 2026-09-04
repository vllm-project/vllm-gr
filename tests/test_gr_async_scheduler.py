# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import pickle
import struct
from types import SimpleNamespace
from unittest import mock

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import PauseState, Scheduler, SchedulingPolicy
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus

from vllm_gr.v1.core.sched.async_scheduler import GRAsyncScheduler
from vllm_gr.v1.engine.engine_core_patch import make_patched_engine_core_init
from vllm_gr.v1.engine.types import (
    GR_CONTROL_TOKEN_ID,
    GR_REQUEST_METADATA_KEY,
    GRRequestMetadata,
    GRRequestStage,
    GRSchedulerOutput,
)
from vllm_gr.v1.metrics.stats import GRRequestCounters

_DIGEST = "sha256:" + "a" * 64


def _metadata() -> GRRequestMetadata:
    return GRRequestMetadata(
        request_id="gr-0",
        beam_width=3,
        generation_steps=3,
        prefix_len=4,
        result_width=3,
        constraint_digest=_DIGEST,
    )


def _native_request(
    raw_metadata: dict[str, object],
    *,
    request_id: str = "gr-0",
    prefix_len: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prefix_len,
        sampling_params=SimpleNamespace(
            extra_args={GR_REQUEST_METADATA_KEY: raw_metadata}
        ),
    )


def _request(step: int = 0):
    return SimpleNamespace(
        request_id="gr-0",
        is_gr_request=True,
        gr_metadata=_metadata(),
        gr_step=step,
        gr_issued_steps=step,
        gr_completed_steps=step,
        gr_parent_ids=[0, 1, 2],
        gr_beam_suffix=[[11], [12], [13]] if step else [],
        gr_counters=GRRequestCounters(),
        gr_persistent_runner_state=False,
        async_tokens_to_discard=0,
    )


def test_add_request_reads_metadata_once() -> None:
    metadata = _metadata()
    request = _native_request(
        {
            field: getattr(metadata, field) for field in metadata.__struct_fields__
        }
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


@pytest.mark.parametrize(
    ("metadata_override", "request_override", "message"),
    [
        ({"protocol_version": 99}, {}, "protocol_version"),
        ({}, {"request_id": "native-other"}, "request_id"),
        ({}, {"prefix_len": 5}, "prefix_len"),
        ({"beam_width": 0}, {}, "beam_width"),
        ({"result_width": 4}, {}, "result_width"),
    ],
)
def test_add_request_rejects_invalid_persistent_metadata_before_upstream_add(
    metadata_override: dict[str, object],
    request_override: dict[str, object],
    message: str,
) -> None:
    metadata = _metadata()
    raw = {
        field: getattr(metadata, field) for field in metadata.__struct_fields__
    }
    raw.update(metadata_override)
    request = _native_request(raw, **request_override)
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {}

    with mock.patch.object(AsyncScheduler, "add_request") as upstream_add:
        with pytest.raises(ValueError, match=message):
            scheduler.add_request(request)

    upstream_add.assert_not_called()


def test_add_request_rejects_second_persistent_request_before_upstream_add() -> None:
    metadata = _metadata()
    raw = {
        field: getattr(metadata, field) for field in metadata.__struct_fields__
    }
    raw["request_id"] = "gr-1"
    request = _native_request(raw, request_id="gr-1")
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": _request()}

    with mock.patch.object(AsyncScheduler, "add_request") as upstream_add:
        with pytest.raises(ValueError, match="B=1"):
            scheduler.add_request(request)

    upstream_add.assert_not_called()


def test_add_request_rejects_configured_constraint_digest_mismatch() -> None:
    metadata = _metadata()
    request = _native_request(
        {
            field: getattr(metadata, field) for field in metadata.__struct_fields__
        }
    )
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {}
    configured = SimpleNamespace(
        constraint_table=SimpleNamespace(
            artifact_digest="sha256:" + "b" * 64,
        ),
        persistent_runner_state=True,
    )

    with (
        mock.patch(
            "vllm_gr.v1.core.sched.async_scheduler.GRConfig.from_vllm_config",
            return_value=configured,
        ),
        mock.patch.object(AsyncScheduler, "add_request") as upstream_add,
    ):
        with pytest.raises(ValueError, match="constraint_digest"):
            scheduler.add_request(request)

    upstream_add.assert_not_called()


def test_force_prefix_reset_fails_before_mutating_active_persistent_gr() -> None:
    request = _request(step=1)
    request.gr_persistent_runner_state = True
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {request.request_id: request}

    with (
        mock.patch.object(AsyncScheduler, "reset_prefix_cache") as upstream_reset,
        pytest.raises(RuntimeError, match="abort the request first"),
    ):
        scheduler.reset_prefix_cache(reset_running_requests=True)

    upstream_reset.assert_not_called()
    assert scheduler.requests == {request.request_id: request}
    assert request.gr_issued_steps == 1
    assert request.gr_completed_steps == 1


def test_non_force_prefix_reset_keeps_upstream_behavior() -> None:
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {}

    with mock.patch.object(
        AsyncScheduler, "reset_prefix_cache", return_value=True
    ) as upstream_reset:
        assert scheduler.reset_prefix_cache(reset_connector=True) is True

    upstream_reset.assert_called_once_with(
        reset_running_requests=False,
        reset_connector=True,
    )


def test_persistent_gr_native_preemption_fails_before_state_reset() -> None:
    request = _request(step=1)
    request.gr_persistent_runner_state = True
    request.num_computed_tokens = request.gr_metadata.prefix_len
    request.status = RequestStatus.RUNNING
    scheduler = object.__new__(GRAsyncScheduler)

    with (
        mock.patch.object(Scheduler, "_preempt_request") as upstream_preempt,
        pytest.raises(RuntimeError, match="increase the native KV cache budget"),
    ):
        scheduler._preempt_request(request, 123.0)

    upstream_preempt.assert_not_called()
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == request.gr_metadata.prefix_len
    assert request.gr_issued_steps == request.gr_completed_steps == 1


def test_ordinary_request_native_preemption_delegates_upstream() -> None:
    request = SimpleNamespace(is_gr_request=False)
    scheduler = object.__new__(GRAsyncScheduler)

    with mock.patch.object(Scheduler, "_preempt_request") as upstream_preempt:
        scheduler._preempt_request(request, 123.0)

    upstream_preempt.assert_called_once_with(request, 123.0)


def test_schedule_keeps_one_logical_token_and_dispatches_beam_rows() -> None:
    request = _request(step=1)
    output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        total_num_scheduled_tokens=1,
    )
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}

    with mock.patch.object(
        AsyncScheduler, "schedule", return_value=output
    ) as upstream_schedule:
        result = scheduler.schedule()

    upstream_schedule.assert_called_once_with()
    assert not hasattr(result, "gr_followup_outputs")
    assert result.num_scheduled_tokens == {"gr-0": 1}
    assert result.total_num_scheduled_tokens == 1
    assert result.gr_outputs["gr-0"].logical_token_count == 1
    assert result.gr_outputs["gr-0"].physical_row_count == 3
    assert result.beam_data["gr-0"]["next_token_ids"] == [11, 12, 13]
    assert result.beam_data["gr-0"]["decode_step"] == 0
    assert result.beam_data["gr-0"]["generation_index"] == 1
    assert request.gr_issued_steps == 2
    assert request.gr_counters.decode_scheduler_calls == 1


def test_typed_sidecar_survives_worker_process_pickle_transport() -> None:
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"gr-0": 1}
    output.total_num_scheduled_tokens = 1
    output.gr_outputs = {
        "gr-0": GRSchedulerOutput(
            request_id="gr-0",
            stage=GRRequestStage.DECODE_RUNNING,
            step=1,
            device_step=1,
            is_prefill=False,
            logical_token_count=1,
            physical_row_count=3,
        )
    }

    restored = pickle.loads(pickle.dumps(output))

    assert restored.num_scheduled_tokens == {"gr-0": 1}
    assert restored.gr_outputs == output.gr_outputs
    assert restored.gr_outputs["gr-0"].physical_row_count == 3


def test_decode_schedule_does_not_query_prefix_cache() -> None:
    request = _request(step=1)
    request.gr_persistent_runner_state = True
    request.num_output_placeholders = 1
    request.num_computed_tokens = request.gr_metadata.prefix_len
    request.num_prompt_tokens = request.gr_metadata.prefix_len
    request.num_tokens = request.gr_metadata.prefix_len
    request.num_tokens_with_spec = request.gr_metadata.prefix_len
    request.max_tokens = request.gr_metadata.max_tokens
    request.has_encoder_inputs = False
    request.spec_token_ids = []
    request.use_structured_output = False

    kv_cache_manager = mock.Mock()
    kv_cache_manager.allocate_slots.return_value = SimpleNamespace()
    kv_cache_manager.get_num_common_prefix_blocks.return_value = 0
    kv_cache_manager.take_new_block_ids.return_value = []

    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.max_num_scheduled_tokens = 1
    scheduler.max_num_running_reqs = 1
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_model_len = 64
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.kv_cache_manager = kv_cache_manager
    scheduler.encoder_cache_manager = mock.Mock()
    scheduler.encoder_cache_manager.get_freed_mm_hashes.return_value = []
    scheduler.num_lookahead_tokens = 0
    scheduler.need_mamba_block_aligned_split = False
    scheduler.use_eagle = False
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.lora_config = None
    scheduler.log_stats = False
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.use_v2_model_runner = False
    scheduler.needs_kv_cache_zeroing = False
    scheduler.enable_return_routed_experts = False
    scheduler.finished_req_ids = set()
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler._make_cached_request_data = mock.Mock(
        return_value=CachedRequestData.make_empty()
    )

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 1}
    assert output.gr_outputs[request.request_id].physical_row_count == 3
    kv_cache_manager.get_computed_blocks.assert_not_called()


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
    request.gr_issued_steps = 2
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        total_num_scheduled_tokens=1,
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
    assert model_output.sampled_token_ids == [[GR_CONTROL_TOKEN_ID]]
    assert request.gr_step == 2
    assert request.gr_completed_steps == 2
    assert request.gr_parent_ids == [2, 0, 1]
    assert request.gr_beam_suffix == [[13, 21], [11, 22], [12, 23]]


def test_runner_state_intermediate_output_is_control_only() -> None:
    request = _request(step=1)
    request.gr_issued_steps = 2
    request.gr_persistent_runner_state = True
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        total_num_scheduled_tokens=1,
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
    assert model_output.sampled_token_ids == [[GR_CONTROL_TOKEN_ID]]
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


def test_ordinary_batch_delegates_upstream_async_bookkeeping() -> None:
    request = SimpleNamespace(request_id="plain", is_gr_request=False)
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {request.request_id: request}
    output = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})

    with mock.patch.object(
        AsyncScheduler, "_update_after_schedule"
    ) as upstream_update:
        scheduler._update_after_schedule(output)

    upstream_update.assert_called_once_with(output)
    assert upstream_update.call_args.args[0] is output


def test_mixed_batch_keeps_gr_placeholder_bookkeeping() -> None:
    gr_request = _request(step=0)
    gr_request.gr_persistent_runner_state = True
    gr_request.is_prefill_chunk = False
    gr_request.num_output_placeholders = 0
    plain_request = SimpleNamespace(
        request_id="plain",
        is_gr_request=False,
        is_prefill_chunk=True,
    )
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {
        gr_request.request_id: gr_request,
        plain_request.request_id: plain_request,
    }
    output = SimpleNamespace(
        num_scheduled_tokens={gr_request.request_id: 1, plain_request.request_id: 1},
        scheduled_spec_decode_tokens={},
        pending_structured_output_tokens=False,
    )

    with (
        mock.patch.object(Scheduler, "_update_after_schedule") as base_update,
        mock.patch.object(AsyncScheduler, "_update_after_schedule") as upstream_update,
    ):
        scheduler._update_after_schedule(output)

    base_update.assert_called_once_with(scheduler, output)
    upstream_update.assert_not_called()
    assert gr_request.num_output_placeholders == 1


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
    # Both stages crossed execute_model(non_block=True), so both are issued:
    # one executing and one queued successor.
    assert request.gr_issued_steps - request.gr_completed_steps == 2

    GRAsyncScheduler._complete_gr_stage(request, generation_index=0)
    assert request.gr_completed_steps == 1
    assert request.gr_issued_steps == 2
    assert request.gr_issued_steps - request.gr_completed_steps == 1


def test_third_outstanding_stage_is_rejected() -> None:
    request = _request(step=0)
    request.gr_persistent_runner_state = True
    request.gr_issued_steps = 2
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}

    with (
        mock.patch.object(AsyncScheduler, "schedule") as upstream_schedule,
        pytest.raises(RuntimeError, match="look-ahead depth"),
    ):
        scheduler.schedule()

    upstream_schedule.assert_not_called()


def test_two_reset_stale_frames_do_not_advance_completion_cursor() -> None:
    request = _request(step=0)
    request.gr_persistent_runner_state = False
    request.gr_issued_steps = 2
    request.async_tokens_to_discard = 2
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {"gr-0": request}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        total_num_scheduled_tokens=1,
        beam_data={"gr-0": {"generation_index": 0}},
    )
    stale_payload = struct.pack(
        "<6i3f", 21, 22, 23, 0, 1, 2, -1.0, -2.0, -3.0
    )
    model_output = SimpleNamespace(
        beam_search_result={
            "buffer_format": "beam_result_i32_f32_v1",
            "shape": [1, 3],
            "buffer": stale_payload,
            "is_last_step": False,
        },
        req_id_to_index={"gr-0": 0},
        sampled_token_ids=[[GR_CONTROL_TOKEN_ID]],
    )

    def consume_async_stale(_scheduler_output, _model_output):
        scheduler._update_request_with_output(request, [GR_CONTROL_TOKEN_ID])
        return {}

    with mock.patch.object(
        Scheduler,
        "update_from_output",
        side_effect=consume_async_stale,
    ):
        scheduler.update_from_output(scheduler_output, model_output)
        scheduler.update_from_output(scheduler_output, model_output)

    assert request.async_tokens_to_discard == 0
    assert request.gr_completed_steps == 0
    assert request.gr_step == 0


def test_aborted_inflight_output_is_ignored_by_gr_reconciliation() -> None:
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr-0": 1},
        total_num_scheduled_tokens=1,
        beam_data={"gr-0": {"generation_index": 0}},
        gr_outputs={
            "gr-0": GRSchedulerOutput(
                "gr-0",
                GRRequestStage.PREFILL_RUNNING,
                0,
                0,
                True,
                is_last_prefill=True,
            )
        },
    )
    model_output = SimpleNamespace(beam_search_result=None)

    with mock.patch.object(
        Scheduler, "update_from_output", return_value={}
    ) as upstream_update:
        result = scheduler.update_from_output(scheduler_output, model_output)

    assert result == {}
    upstream_update.assert_called_once_with(scheduler_output, model_output)


class _TrackingDeque(deque):
    def __init__(self) -> None:
        super().__init__(maxlen=2)
        self.high_watermark = 0

    def appendleft(self, item) -> None:
        super().appendleft(item)
        self.high_watermark = max(self.high_watermark, len(self))


class _DeferredResult:
    def __init__(self, name: str, trace: list[str]) -> None:
        self.name = name
        self.trace = trace

    def done(self) -> bool:
        return False

    def result(self):
        self.trace.append(f"result:{self.name}")
        return SimpleNamespace(name=self.name)


def test_engine_batch_queue_prepares_next_stage_before_first_future_result() -> None:
    trace: list[str] = []
    outputs = [SchedulerOutput.make_empty(), SchedulerOutput.make_empty()]
    for index, output in enumerate(outputs):
        output.num_scheduled_tokens = {"gr-0": 1}
        output.total_num_scheduled_tokens = 1
        output.pending_structured_output_tokens = False
        output.marker = index

    class FakeScheduler:
        def __init__(self) -> None:
            self.index = 0

        def has_requests(self) -> bool:
            return self.index < len(outputs)

        def schedule(self):
            output = outputs[self.index]
            trace.append(f"schedule:{self.index}")
            self.index += 1
            return output

        def get_grammar_bitmask(self, _output):
            return None

        def update_from_output(self, output, model_output):
            trace.append(f"update:{output.marker}:{model_output.name}")
            return {0: output.marker}

    class FakeExecutor:
        def execute_model(self, output, *, non_block):
            assert non_block is True
            trace.append(f"execute:{output.marker}")
            return _DeferredResult(f"execute-{output.marker}", trace)

        def sample_tokens(self, _grammar, *, non_block):
            assert non_block is True
            marker = len([event for event in trace if event.startswith("sample:")])
            trace.append(f"sample:{marker}")
            return _DeferredResult(f"sample-{marker}", trace)

    batch_queue = _TrackingDeque()
    engine = SimpleNamespace(
        batch_queue=batch_queue,
        batch_queue_size=2,
        scheduler=FakeScheduler(),
        model_executor=FakeExecutor(),
        is_ec_consumer=True,
        is_pooling_model=False,
        use_spec_decode=False,
        log_error_detail=lambda _output: nullcontext(),
        log_iteration_details=lambda _output: nullcontext(),
        _process_aborts_queue=lambda: None,
    )

    first_outputs, first_progress = EngineCore.step_with_batch_queue(engine)
    second_outputs, second_progress = EngineCore.step_with_batch_queue(engine)

    assert first_outputs is None
    assert first_progress is True
    assert second_outputs == {0: 0}
    assert second_progress is True
    assert trace.index("schedule:1") < trace.index("result:sample-0")
    assert batch_queue.high_watermark == 2


def test_gr_batch_queue_tracks_two_outstanding_and_one_step_lookahead() -> None:
    trace: list[str] = []
    request = _request(step=0)
    request.gr_persistent_runner_state = True
    request.num_computed_tokens = request.gr_metadata.prefix_len
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.get_grammar_bitmask = lambda _output: None

    native_outputs = [SchedulerOutput.make_empty(), SchedulerOutput.make_empty()]
    for index, output in enumerate(native_outputs):
        output.num_scheduled_tokens = {request.request_id: 1}
        output.total_num_scheduled_tokens = 1
        output.pending_structured_output_tokens = False
        output.marker = index
    scheduler.has_requests = lambda: bool(native_outputs)

    class ResultFuture:
        def __init__(self, stage: int, *, sample: bool) -> None:
            self.stage = stage
            self.sample = sample

        def done(self) -> bool:
            return False

        def result(self):
            kind = "sample" if self.sample else "execute"
            trace.append(f"result:{kind}:{self.stage}")
            if not self.sample:
                return None
            trace.append(
                "outstanding-before-result:"
                f"{request.gr_issued_steps - request.gr_completed_steps}"
            )
            return SimpleNamespace(
                beam_search_result=None,
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[GR_CONTROL_TOKEN_ID]],
            )

    class FakeExecutor:
        def __init__(self) -> None:
            self.stage = -1

        def execute_model(self, output, *, non_block):
            assert non_block is True
            self.stage = output.gr_outputs[request.request_id].step
            trace.append(f"execute:{self.stage}")
            return ResultFuture(self.stage, sample=False)

        def sample_tokens(self, _grammar, *, non_block):
            assert non_block is True
            trace.append(f"sample:{self.stage}")
            return ResultFuture(self.stage, sample=True)

    batch_queue = _TrackingDeque()
    engine = SimpleNamespace(
        batch_queue=batch_queue,
        batch_queue_size=2,
        scheduler=scheduler,
        model_executor=FakeExecutor(),
        is_ec_consumer=True,
        is_pooling_model=False,
        use_spec_decode=False,
        log_error_detail=lambda _output: nullcontext(),
        log_iteration_details=lambda _output: nullcontext(),
        _process_aborts_queue=lambda: None,
    )

    def upstream_schedule():
        output = native_outputs.pop(0)
        trace.append(f"schedule:{output.marker}")
        return output

    def upstream_update(_scheduler_output, _model_output):
        if request.async_tokens_to_discard:
            scheduler._update_request_with_output(
                request, [GR_CONTROL_TOKEN_ID]
            )
            return {}
        return {0: "stage-0"}

    with (
        mock.patch.object(AsyncScheduler, "schedule", side_effect=upstream_schedule),
        mock.patch.object(Scheduler, "update_from_output", side_effect=upstream_update),
    ):
        first_outputs, first_progress = EngineCore.step_with_batch_queue(engine)
        second_outputs, second_progress = EngineCore.step_with_batch_queue(engine)

        # Simulate an already-marked stale frame after stage 0 reconciled while
        # stage 1's Future is still pending. Force prefix reset for persistent
        # GR is separately rejected before mutation, but stale output handling
        # remains fail-closed for abort/error recovery paths.
        assert len(batch_queue) == 1
        request.async_tokens_to_discard = 1
        stale_outputs, stale_progress = EngineCore.step_with_batch_queue(engine)

    assert first_outputs is None
    assert first_progress is True
    assert second_outputs == {0: "stage-0"}
    assert second_progress is True
    assert trace.index("schedule:1") < trace.index("result:sample:0")
    assert "outstanding-before-result:2" in trace
    assert request.gr_issued_steps == 2
    assert request.gr_completed_steps == 1
    assert request.gr_issued_steps - request.gr_completed_steps == 1
    assert stale_outputs == {}
    assert stale_progress is False
    assert request.async_tokens_to_discard == 0
    assert request.gr_completed_steps == 1
    assert batch_queue.high_watermark == 2


def test_engine_init_accepts_exact_gr_batch_queue_depth_two() -> None:
    original_step = object()

    def original_init(engine) -> None:
        engine.scheduler = object.__new__(GRAsyncScheduler)
        engine.batch_queue_size = 2
        engine.batch_queue = deque(maxlen=2)
        engine.step_fn = original_step

    engine = SimpleNamespace()
    make_patched_engine_core_init(original_init)(engine)

    assert engine.batch_queue_size == 2
    assert engine.batch_queue.maxlen == 2
    assert engine.step_fn is original_step


@pytest.mark.parametrize("depth", [1, 3, 4])
def test_engine_init_rejects_unsupported_gr_batch_queue_depth(depth: int) -> None:
    def original_init(engine) -> None:
        engine.scheduler = object.__new__(GRAsyncScheduler)
        engine.batch_queue_size = depth
        engine.batch_queue = deque(maxlen=depth) if depth > 1 else None
        engine.step_fn = object()

    with pytest.raises(ValueError, match="BatchQueue depth == 2"):
        make_patched_engine_core_init(original_init)(SimpleNamespace())


def test_engine_init_does_not_restrict_ordinary_scheduler_queue_depth() -> None:
    def original_init(engine) -> None:
        engine.scheduler = SimpleNamespace()
        engine.batch_queue_size = 4
        engine.batch_queue = deque(maxlen=4)
        engine.step_fn = object()

    engine = SimpleNamespace()
    make_patched_engine_core_init(original_init)(engine)
    assert engine.batch_queue_size == 4
    assert engine.batch_queue.maxlen == 4


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
        generation_steps=1,
        prefix_len=4,
        result_width=3,
        constraint_digest=_DIGEST,
    )
    request.gr_persistent_runner_state = True
    request.gr_issued_steps = 1
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
    assert model_output.sampled_token_ids == [[GR_CONTROL_TOKEN_ID]]
    assert request.gr_completed_steps == 1


def test_ordinary_request_delegates_one_schedule_unchanged() -> None:
    request = SimpleNamespace(request_id="plain", is_gr_request=False)
    output = SchedulerOutput.make_empty()
    scheduler = object.__new__(GRAsyncScheduler)
    scheduler.requests = {request.request_id: request}

    with mock.patch.object(
        AsyncScheduler, "schedule", return_value=output
    ) as upstream_schedule:
        result = scheduler.schedule()

    upstream_schedule.assert_called_once_with()
    assert result is output
    assert result.num_scheduled_tokens == {}
    assert result.total_num_scheduled_tokens == 0
    assert not hasattr(result, "beam_data")
