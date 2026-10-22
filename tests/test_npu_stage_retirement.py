# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completed NPU Beam sessions retire their native CPU request state."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_gr.v1.engine.beam_search_v1 import GRRetireMetadata
from vllm_gr.v1.engine.gr_async_scheduler import GR_RETIRE_OUTPUT_ATTR
from vllm_gr.v1.worker.model_runner_common import InputBatchProxy
from vllm_gr.v1.worker.npu_beam_stage_runner import (
    NPUEagerBeamStageRunner,
    execute_npu_gr_retirement,
)
from vllm_gr.v1.worker.npu_model_runner_patch import make_patched_update_states


def _native_worker(monkeypatch: pytest.MonkeyPatch, request_ids: tuple[str, ...]) -> Any:
    # Use the actual native method inherited by Ascend, with only its hardware
    # services and fixed-capacity InputBatch storage replaced by small doubles.
    from vllm.v1.worker import gpu_model_runner

    monkeypatch.setattr(
        gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    order: list[str] = []
    all_ids = (*request_ids, "unrelated")
    batch = SimpleNamespace(req_id_to_index={key: i for i, key in enumerate(all_ids)})

    def remove(request_id: str) -> None:
        assert order and order[0] == "input_dma_drained"
        batch.req_id_to_index.pop(request_id, None)

    batch.remove_request = Mock(side_effect=remove)
    batch.condense = Mock()
    batch.refresh_metadata = Mock()
    worker = SimpleNamespace(
        input_batch=batch,
        requests={key: SimpleNamespace(prompt_token_ids=[1, 2, 3]) for key in all_ids},
        num_prompt_logprobs=dict.fromkeys(all_ids, 5),
        encoder_cache={"free-me": object(), "keep-me": object()},
        late_interaction_runner=SimpleNamespace(on_requests_finished=Mock()),
        speculative_config=None,
        use_async_spec_decode=False,
        _may_reorder_batch=Mock(),
        prepare_inputs_event=SimpleNamespace(
            synchronize=Mock(side_effect=lambda: order.append("input_dma_drained"))
        ),
        _beam_sessions={key: object() for key in all_ids},
        _release_order=order,
    )
    native = make_patched_update_states(gpu_model_runner.GPUModelRunner._update_states)
    worker._update_states = Mock(wraps=native.__get__(worker))
    return worker


def _owner(worker: Any, request_ids: tuple[str, ...], *, pending: bool) -> Any:
    execution = SimpleNamespace(
        session_id="batch-0", owner_generation=1, request_ids=request_ids, kind="decode"
    )
    owner: Any = object.__new__(NPUEagerBeamStageRunner)
    owner.worker = worker
    owner.closed = False
    owner.pending_plan = SimpleNamespace(execution=execution) if pending else None
    owner.previous = None if pending else execution
    owner.logical_scheduler_output = object()
    owner.execution_error = None
    owner.attention_batch = SimpleNamespace(
        sessions=tuple(SimpleNamespace(session_id=key) for key in request_ids),
        end_session=Mock(side_effect=lambda: worker._release_order.append("beam_released")),
    )
    worker._gr_npu_stage_runner = owner
    worker._current_scheduler_output = owner.logical_scheduler_output
    return owner


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("single_step", [False, True])  # type: ignore[misc]
def test_terminal_release_consumes_native_requests_once(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, single_step: bool
) -> None:
    request_ids = tuple(f"request-{index}" for index in range(batch_size))
    worker = _native_worker(monkeypatch, request_ids)
    owner = _owner(worker, request_ids, pending=True)
    if single_step:
        # N=1 finishes in Prefill, before an attention batch is allocated.
        owner.pending_plan.execution.kind = "prefill"
        owner.attention_batch = None
    original_batch = worker.input_batch
    worker.input_batch = InputBatchProxy(
        original_batch, list(request_ids), list(range(batch_size)), list(range(batch_size))
    )

    owner._release()
    owner._release()

    assert owner.closed and owner.logical_scheduler_output is None
    assert worker.input_batch is original_batch
    assert set(worker.requests) == {"unrelated"}
    assert set(worker.num_prompt_logprobs) == {"unrelated"}
    # Native empty-frame bookkeeping evicts unscheduled rows but retains their
    # request caches, just as an ordinary preempted/paused request requires.
    assert not original_batch.req_id_to_index
    worker._update_states.assert_called_once()
    cleanup = worker._update_states.call_args.args[0]
    assert cleanup.finished_req_ids == set(request_ids)
    assert not cleanup.num_scheduled_tokens and cleanup.total_num_scheduled_tokens == 0
    worker.late_interaction_runner.on_requests_finished.assert_called_once_with(set(request_ids))
    worker.prepare_inputs_event.synchronize.assert_called_once()
    original_batch.condense.assert_called_once()
    original_batch.refresh_metadata.assert_called_once()
    if not single_step:
        assert set(worker._beam_sessions) == {"unrelated"}
        owner.attention_batch.end_session.assert_called_once()
        assert worker._release_order == ["input_dma_drained", "beam_released"]


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("failed_dispatch", [False, True])  # type: ignore[misc]
def test_retirement_preserves_native_cleanup_and_replays_receipt_once(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, failed_dispatch: bool
) -> None:
    request_ids = tuple(f"request-{index}" for index in range(batch_size))
    worker = _native_worker(monkeypatch, request_ids)
    owner = _owner(worker, request_ids, pending=False)
    if failed_dispatch:
        owner.execution_error = ValueError("failed later dispatch")
    scheduled = SchedulerOutput.make_empty()
    scheduled.finished_req_ids.add("batch-0")
    scheduled.free_encoder_mm_hashes = ["free-me"]
    metadata = GRRetireMetadata("batch-0", 1, 0)
    setattr(scheduled, GR_RETIRE_OUTPUT_ATTR, metadata)

    output = execute_npu_gr_retirement(worker, scheduled)
    assert execute_npu_gr_retirement(worker, scheduled) is output

    assert owner.closed and owner.execution_error is None
    assert set(worker.requests) == {"unrelated"}
    assert set(worker.num_prompt_logprobs) == {"unrelated"}
    assert set(worker.encoder_cache) == {"keep-me"}
    assert scheduled.finished_req_ids == {"batch-0"}
    worker._update_states.assert_called_once()
    cleanup = worker._update_states.call_args.args[0]
    assert cleanup is not scheduled
    assert cleanup.finished_req_ids == {"batch-0", *request_ids}
    worker.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"batch-0", *request_ids}
    )
    owner.attention_batch.end_session.assert_called_once()
    worker.prepare_inputs_event.synchronize.assert_called_once()


@pytest.mark.parametrize("failure_point", ["input_dma", "native_cleanup"])  # type: ignore[misc]
def test_release_failure_does_not_claim_resources_released(
    monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    worker = _native_worker(monkeypatch, ("request-0",))
    owner = _owner(worker, ("request-0",), pending=True)
    failure = RuntimeError("cleanup not ready")
    if failure_point == "input_dma":
        worker.prepare_inputs_event.synchronize.side_effect = failure
    else:
        worker._update_states.side_effect = failure

    with pytest.raises(RuntimeError, match="cleanup not ready"):
        owner._release()

    assert not owner.closed
    assert "request-0" in worker.requests
    owner.attention_batch.end_session.assert_not_called()
