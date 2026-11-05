# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU fault injection for persistent NPU stage preparation ownership."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.engine.beam_search_v1 import GRRetireMetadata
from vllm_gr.v1.engine.gr_async_scheduler import GR_RETIRE_RESULT_ATTR
from vllm_gr.v1.worker import npu_beam_stage_runner as stage_module
from vllm_gr.v1.worker.npu_beam_stage_adapter import NPUEagerStagePlan
from vllm_gr.v1.worker.npu_beam_stage_runner import (
    NPUEagerBeamStageRunner,
    execute_npu_gr_retirement,
)

pytestmark = pytest.mark.cpu_test


def _scheduled(*, prefill: bool = False, dynamic: bool = False) -> Any:
    request_ids = ("request-0", "request-1")
    lengths = (2, 3)
    metadata = SimpleNamespace(
        session_id="batch-0",
        dispatch_id=0 if prefill else 1,
        stage_index=0 if prefill else 1,
        kind="prefill" if prefill else "decode",
        decode_step=None if prefill else 0,
        owner_generation=7,
        request_ids=request_ids,
        num_prompt_tokens_by_request=lengths,
        num_prompt_tokens=sum(lengths),
        beam_params=SimpleNamespace(
            beam_width=4,
            max_tokens=3,
            dynamic_beam_widths=(4, 2, 4) if dynamic else None,
        ),
        produces_output=True,
        execution_capacity=SimpleNamespace(beam_width=4),
        is_last_stage=False,
    )
    counts = lengths if prefill else (1, 1)
    return SimpleNamespace(
        gr_stage_metadata={metadata.session_id: metadata},
        num_scheduled_tokens=dict(zip(request_ids, counts)),
        total_num_scheduled_tokens=sum(counts),
        beam_data={},
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )


def _stage(*, dynamic: bool = False) -> tuple[Any, Any, Any, BeamAttentionContext]:
    context = BeamAttentionContext(2, 4, 1, 3, 2, 1, torch.device("cpu"), torch.float16)
    scheduled = _scheduled(dynamic=dynamic)
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _current_scheduler_output=scheduled,
        _gr_npu_graph_context=context,
        _gr_npu_beam_graph_shapes={(2, 4)},
        _vllm_gr_config=SimpleNamespace(beam_graph_widths=None),
        _beam_context=None,
        _beam_sessions={},
        execute_model_state=None,
        prepare_inputs_event=None,
        _update_states=Mock(),
    )
    stage: Any = object.__new__(NPUEagerBeamStageRunner)
    stage.worker = worker
    stage.previous = NPUEagerStagePlan.from_scheduler_output(
        _scheduled(prefill=True, dynamic=dynamic)
    ).execution
    stage.pending_plan = None
    stage.execution_error = None
    stage.logical_scheduler_output = None
    stage.handoff = None
    stage.prefill_candidates = ConstraintOutput(
        token_ids=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32),
        logprobs=torch.tensor([[-0.1, -0.2, -0.3, -0.4]] * 2),
        valid_mask=torch.ones((2, 4), dtype=torch.bool),
    )
    stage.latest_token_ids = stage.prefill_candidates.token_ids
    stage.parent_history = None
    stage.session_request = None
    stage.decode_candidates = None
    stage.logits = torch.ones((8, 32))
    stage.attention_batch = None
    stage.dynamic_state = None
    stage.closed = False
    worker._gr_npu_stage_runner = stage
    return stage, worker, scheduled, context


def _retirement(*, session_id: str = "batch-0", generation: int = 7) -> Any:
    return SimpleNamespace(
        gr_retire_metadata=GRRetireMetadata(session_id, generation, 1),
        total_num_scheduled_tokens=0,
        num_scheduled_tokens={},
        gr_stage_metadata={},
        finished_req_ids={session_id},
    )


def _assert_failed(stage: Any, scheduled: Any, failure: Exception) -> None:
    assert stage.execution_error is failure
    assert stage.pending_plan is None
    assert stage.logits is None
    original = Mock(side_effect=AssertionError("a failed owner must be retired"))
    with pytest.raises(RuntimeError, match="retirement is required") as retry:
        stage.execute(scheduled, original)
    assert retry.value.__cause__ is failure
    original.assert_not_called()


def _retire_failed_owner(stage: Any, worker: Any) -> None:
    for retirement in (
        _retirement(session_id="other-batch"),
        _retirement(generation=8),
    ):
        with pytest.raises(RuntimeError, match="no matching live NPU owner"):
            execute_npu_gr_retirement(worker, retirement)
        assert not stage.closed
    worker._update_states.assert_not_called()
    retirement = _retirement()
    output = execute_npu_gr_retirement(worker, retirement)
    receipt = getattr(output, GR_RETIRE_RESULT_ATTR)
    assert receipt.released
    assert receipt.session_id == "batch-0" and receipt.owner_generation == 7
    assert stage.closed and stage.execution_error is None
    assert execute_npu_gr_retirement(worker, retirement) is output
    worker._update_states.assert_called_once_with(retirement)


@pytest.mark.parametrize(  # type: ignore[misc]
    "failure_point", ["allocate", "construct", "begin", "batch", "dynamic", "parent_history"]
)
def test_new_attention_batch_preparation_rolls_back_all_owned_slots(
    monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    stage, worker, scheduled, context = _stage(dynamic=failure_point == "dynamic")
    failure = ValueError(f"injected {failure_point} failure")
    original = Mock(side_effect=AssertionError("preparation must fail before forward"))
    allocate = context.registry.allocate
    begin = BeamSearchSession.begin_session
    allocated: list[str] = []
    begun: list[str] = []

    def allocate_slot(request_id: str, *, beam_width: int, prefix_len: int) -> int:
        if failure_point == "allocate" and allocated:
            raise failure
        slot = allocate(request_id, beam_width, prefix_len)
        allocated.append(request_id)
        return cast(int, slot)

    def begin_session(session: Any, *args: Any, **kwargs: Any) -> None:
        if failure_point == "begin" and begun:
            raise failure
        begin(session, *args, **kwargs)
        begun.append(session.session_id)

    monkeypatch.setattr(context.registry, "allocate", allocate_slot)
    monkeypatch.setattr(BeamSearchSession, "begin_session", begin_session)
    if failure_point == "construct":

        def construct_session(request_id: str, slot: int, owner: Any) -> BeamSearchSession:
            if request_id == "request-1":
                raise failure
            return BeamSearchSession(request_id, slot, owner)

        monkeypatch.setattr(stage_module, "BeamSearchSession", construct_session)
    elif failure_point == "batch":
        monkeypatch.setattr(stage_module, "NPUBeamExecutionBatch", Mock(side_effect=failure))
    elif failure_point == "dynamic":
        from vllm_gr.v1.beam.npu_dynamic_beam import NPUDynamicBeamState

        monkeypatch.setattr(NPUDynamicBeamState, "initialize", Mock(side_effect=failure))
    elif failure_point == "parent_history":
        monkeypatch.setattr(torch, "zeros_like", Mock(side_effect=failure))

    outer_data = {"sentinel": object()}
    token = BEAM_DATA_VAR.set(outer_data)
    try:
        with pytest.raises(ValueError) as caught:
            stage.execute(scheduled, original)
        assert caught.value is failure
        assert BEAM_DATA_VAR.get() is outer_data
    finally:
        BEAM_DATA_VAR.reset(token)
    expected_allocations = (
        ["request-0"] if failure_point == "allocate" else ["request-0", "request-1"]
    )
    assert allocated == expected_allocations
    assert context.registry.num_active_sessions == 0
    assert not context.registry.has_session("request-0")
    assert not context.registry.has_session("request-1")
    assert stage.attention_batch is None
    assert stage.dynamic_state is None and stage.parent_history is None
    assert worker._beam_context is None and worker._beam_sessions == {}
    original.assert_not_called()
    _assert_failed(stage, scheduled, failure)
    _retire_failed_owner(stage, worker)


@pytest.mark.parametrize("existing_id", ["foreign-request", "request-0"])  # type: ignore[misc]
def test_active_context_is_rejected_without_touching_existing_session(
    monkeypatch: pytest.MonkeyPatch, existing_id: str
) -> None:
    stage, worker, scheduled, context = _stage()
    slot = context.registry.allocate(existing_id, beam_width=4, prefix_len=17)
    existing = BeamSearchSession(existing_id, slot, context)
    existing.begin_session(4, 17)
    existing._state.decode_step = 2
    context.decision.sequence.fill_(91)
    context.decision.beam_scores.fill_(-13)
    context.decision.worker_decode_steps = 2
    sequence = context.decision.sequence.clone()
    scores = context.decision.beam_scores.clone()
    worker._beam_context = context
    worker._beam_sessions = {existing_id: existing}
    allocate = Mock(wraps=context.registry.allocate)
    monkeypatch.setattr(context.registry, "allocate", allocate)
    original = Mock(side_effect=AssertionError("an occupied context cannot run this owner"))
    with pytest.raises(RuntimeError, match="active") as caught:
        stage.execute(scheduled, original)
    _assert_failed(stage, scheduled, caught.value)
    assert context.registry.num_active_sessions == 1
    assert context.registry.get_slot(existing_id) == slot
    assert existing.is_active and existing.decode_step == 2
    assert existing.beam_width == 4 and existing._state.prefix_len == 17
    assert torch.equal(context.decision.sequence, sequence)
    assert torch.equal(context.decision.beam_scores, scores)
    assert context.decision.worker_decode_steps == 2
    assert worker._beam_context is context
    assert worker._beam_sessions == {existing_id: existing}
    assert stage.attention_batch is None
    allocate.assert_not_called()
    original.assert_not_called()
    _retire_failed_owner(stage, worker)
    assert context.registry.num_active_sessions == 1 and existing.is_active
    assert worker._beam_sessions == {existing_id: existing}
    existing.end_session()


@pytest.mark.parametrize(  # type: ignore[misc]
    "failure_point", ["runtime_output", "ensure", "metadata", "beam_data", "step_buffer", "forward"]
)
def test_existing_attention_batch_is_retained_until_failed_owner_retirement(
    monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    stage, worker, scheduled, context = _stage()
    stage._ensure_attention_batch(NPUEagerStagePlan.from_scheduler_output(scheduled))
    attention_batch = stage.attention_batch
    assert attention_batch is not None
    sessions = dict(worker._beam_sessions)
    failure = ValueError(f"injected {failure_point} failure")

    def forward(*args: Any, **kwargs: Any) -> None:
        assert BEAM_DATA_VAR.get() is worker._current_beam_info
        raise failure

    original = Mock(side_effect=forward)
    if failure_point == "runtime_output":
        monkeypatch.setattr(stage, "_runtime_scheduler_output", Mock(side_effect=failure))
    elif failure_point == "ensure":
        monkeypatch.setattr(stage, "_ensure_attention_batch", Mock(side_effect=failure))
    elif failure_point == "metadata":
        metadata = next(iter(scheduled.gr_stage_metadata.values()))
        del metadata.is_last_stage
    elif failure_point == "beam_data":
        monkeypatch.setattr(stage, "_beam_data", Mock(side_effect=failure))
    elif failure_point == "step_buffer":
        monkeypatch.setattr(
            context, "decode_step_buf", SimpleNamespace(fill_=Mock(side_effect=failure))
        )

    outer_data = {"sentinel": object()}
    token = BEAM_DATA_VAR.set(outer_data)
    try:
        with pytest.raises((ValueError, AttributeError)) as caught:
            stage.execute(scheduled, original)
        if failure_point != "metadata":
            assert caught.value is failure
        assert BEAM_DATA_VAR.get() is outer_data
    finally:
        BEAM_DATA_VAR.reset(token)
    _assert_failed(stage, scheduled, caught.value)
    assert stage.attention_batch is attention_batch
    assert worker._beam_sessions == sessions
    assert context.registry.num_active_sessions == 2
    assert all(session.is_active for session in attention_batch.sessions)
    if failure_point == "forward":
        original.assert_called_once()
    else:
        original.assert_not_called()
    _retire_failed_owner(stage, worker)
    assert context.registry.num_active_sessions == 0
    assert worker._beam_sessions == {}


def test_retirement_allows_a_fresh_owner_to_start_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, worker, scheduled, context = _stage()
    failure = ValueError("injected Decode preparation failure")
    monkeypatch.setattr(stage, "_ensure_attention_batch", Mock(side_effect=failure))
    original = Mock(side_effect=AssertionError("preparation must fail before forward"))
    with pytest.raises(ValueError) as caught:
        stage.execute(scheduled, original)
    assert caught.value is failure
    _assert_failed(stage, scheduled, failure)
    _retire_failed_owner(stage, worker)

    # Use actual deployment validation without requiring an NPU allocation.
    worker.device = "npu"
    worker.use_async_scheduling = True
    worker.model_config = SimpleNamespace(dtype=torch.float16)
    worker.vllm_config = SimpleNamespace(additional_config={"vllm_gr_beam_execution_mode": "v1"})
    prefill = _scheduled(prefill=True)
    replacement = stage_module.get_npu_stage_runner(worker, prefill)
    assert replacement is not stage
    assert not replacement.closed and replacement.execution_error is None
    assert replacement.previous is None and replacement.pending_plan is None
    assert worker._gr_npu_stage_runner is replacement
    result = SimpleNamespace(logits=torch.ones((2, 32)))
    assert replacement.execute(prefill, Mock(return_value=result)) is result
    assert replacement.pending_plan is not None
    assert context.registry.num_active_sessions == 0
