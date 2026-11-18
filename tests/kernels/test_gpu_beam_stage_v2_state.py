# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise GR stage boundaries against real ModelRunner V2 request state."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from tests.kernels.beam_stage_test_utils import (
    admission_worker,
    dispatch,
    metadata,
    runtime,
)
from tests.kernels.v2_input_test_utils import make_native_v2_worker
from vllm_gr.v1.beam.execution_batch import BeamExecutionContractError
from vllm_gr.v1.engine.beam_search_v1 import validate_dispatch_session
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
)
from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import get_stage_runner
from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


@pytest.mark.parametrize("cached", [0, 2, 7])
@pytest.mark.parametrize("specialized", [False, True])
def test_full_prefill_advances_native_state_without_counting_padding(
    monkeypatch, cached, specialized
):
    prompt = list(range(10, 18))
    worker = make_native_v2_worker(prompt, cached)
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    if specialized:
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState

        worker.model_state = DefaultModelState.__new__(DefaultModelState)
        worker.model_state.rope_state = None
        worker.use_pp = False
        worker.dp_size = 1
        worker.supports_mm_inputs = False
        worker.is_encoder_decoder = False
        worker.is_pooling_model = False
        worker.lora_config = None
        worker.speculative_config = None
        worker.max_num_tokens = 64
    hooks = CudaStageHooksV2(worker)
    req_index = worker.req_states.req_id_to_index["session-0"]
    observed = []
    kv = torch.zeros(len(prompt), dtype=torch.int32, device=worker.device)
    kv[:cached] = torch.tensor(prompt[:cached], dtype=torch.int32, device=worker.device)

    def native_prefill(runner, scheduled):
        runner.update_requests(scheduled)
        count = scheduled.total_num_scheduled_tokens
        # Padding must not advance the request's logical token count.
        desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 1)
        batch_state, _ = runner.gather_batch_req_state(scheduled, dummy_run=False)
        if specialized:
            from vllm_gr.v1.worker.cuda.v2.gpu_prefill_v2 import (
                try_prepare_gr_prefill_inputs_v2,
            )

            batch = try_prepare_gr_prefill_inputs_v2(runner, scheduled, batch_state, desc)
            assert batch is not None
            assert batch.has_prefill and batch.is_prefilling_np.tolist() == [True]
            assert batch.num_computed_tokens_np.tolist() == [cached]
            assert batch.prefill_len_np.tolist() == [len(prompt)]
            assert batch.num_computed_prefill_tokens_np.tolist() == [cached]
            assert batch.input_ids.data_ptr() == runner.input_buffers.input_ids.data_ptr()
        else:
            batch = runner.prepare_inputs(scheduled, batch_state, desc)
        assert batch.is_padding.tolist() == [False] * count + [True] * (8 - count)
        ids = batch.input_ids[:count].clone()
        positions = batch.positions[:count].clone()
        observed.append((ids, positions))
        kv.index_copy_(0, positions, ids)
        runner.execute_model_state = SimpleNamespace(
            input_batch=batch, hidden_states=batch.input_ids.float().unsqueeze(1)
        )
        # Only observe the GR hook after native input preparation completes.
        for method in ("item", "tolist", "cpu"):
            guard.setattr(torch.Tensor, method, forbidden)

    def forbidden(self, *args, **kwargs):
        raise AssertionError("Prefill logits must not copy device metadata to the CPU")

    count = len(prompt) - cached
    scheduled = SchedulerOutput.make_empty()
    from dataclasses import replace

    scheduled.gr_stage_metadata = {"session-0": replace(metadata(), num_prompt_tokens=len(prompt))}
    scheduled.num_scheduled_tokens = {"session-0": count}
    scheduled.total_num_scheduled_tokens = count
    scheduled.scheduled_cached_reqs.req_ids = ["session-0"]
    scheduled.scheduled_cached_reqs.num_computed_tokens = [cached]
    scheduled.scheduled_cached_reqs.new_block_ids = [None]
    with monkeypatch.context() as guard:
        logits = hooks.prefill_logits(native_prefill, scheduled)
    assert worker.execute_model_state is None
    assert logits.flatten().tolist() == [float(prompt[-1])]
    ids, positions = observed[0]
    assert ids.tolist() == prompt[cached:]
    assert positions.tolist() == list(range(cached, len(prompt)))
    assert worker.req_states.num_computed_tokens.gpu[req_index].item() == len(prompt)
    assert kv.tolist() == prompt
    # Beam outputs belong to GR; native token history must remain the prompt.
    assert worker.req_states.total_len.gpu[req_index].item() == len(prompt)


def test_terminal_output_retires_native_v2_request():
    owner = runtime(n=1)
    worker = make_native_v2_worker(list(range(7)))
    worker._beam_sessions = owner.worker._beam_sessions
    owner.worker = worker
    owner.hooks = CudaStageHooksV2(worker)
    owner.admit(dispatch(metadata(n=1)))
    owner.logits = torch.zeros((1, 6), dtype=torch.float16, device=worker.device)

    output = owner.sample().get_output()
    result = getattr(output, GR_WORKER_RESULT_ATTR)["session-0"]
    assert result.finished and result.resources_released
    assert owner.closed
    assert worker.req_states.num_reqs == 0
    assert worker._beam_sessions == {}
    worker.lora_state.remove_request.assert_called_once_with("session-0")

    # The request row can be admitted again after the fused terminal cleanup.
    worker.req_states.add_request("next-session", 1, [1], 0, max_tokens=3)
    assert worker.req_states.num_reqs == 1


@pytest.mark.l2
def test_ordinary_prefill_chunks_keep_native_state_and_token_progress():
    """Use native input/count operations and stand-in forward/sampling callbacks."""
    from unittest.mock import Mock

    from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch import (
        make_patched_execute_model_v2,
        make_patched_sample_tokens_v2,
    )

    prompt = list(range(10, 18))
    worker = make_native_v2_worker(prompt)
    worker.execute_model_state = None
    finish = Mock(side_effect=AssertionError("ordinary chunk reached GR cleanup"))
    worker._gr_gpu_stage_runner = SimpleNamespace(
        pending=None, closed=True, hooks=SimpleNamespace(finish_native_output=finish)
    )
    observed = []

    def native_execute(runner, scheduled):
        assert runner.execute_model_state is None
        runner.update_requests(scheduled)
        desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, scheduled.total_num_scheduled_tokens, 1)
        batch_state, _ = runner.gather_batch_req_state(scheduled, dummy_run=False)
        batch = runner.prepare_inputs(scheduled, batch_state, desc)
        count = scheduled.total_num_scheduled_tokens
        observed.append((batch.input_ids[:count].clone(), batch.positions[:count].clone()))
        runner.execute_model_state = SimpleNamespace(input_batch=batch)

    def native_sample(runner, grammar):
        assert grammar is None
        state = runner.execute_model_state
        assert state is not None
        runner.execute_model_state = None
        runner.postprocess_num_computed_tokens(state.input_batch)
        return state

    execute = make_patched_execute_model_v2(native_execute)
    sample = make_patched_sample_tokens_v2(native_sample)
    computed = 0
    req_index = worker.req_states.req_id_to_index["session-0"]
    for count in (2, 3, 3):
        scheduled = SchedulerOutput.make_empty()
        scheduled.num_scheduled_tokens = {"session-0": count}
        scheduled.total_num_scheduled_tokens = count
        scheduled.scheduled_cached_reqs.req_ids = ["session-0"]
        scheduled.scheduled_cached_reqs.num_computed_tokens = [computed]
        scheduled.scheduled_cached_reqs.new_block_ids = [None]
        assert execute(worker, scheduled) is None
        state = worker.execute_model_state
        assert state is not None
        assert sample(worker, None) is state
        assert worker.execute_model_state is None
        ids, positions = observed[-1]
        assert ids.tolist() == prompt[computed : computed + count]
        assert positions.tolist() == list(range(computed, computed + count))
        computed += count
        assert worker.req_states.num_computed_tokens.gpu[req_index].item() == computed
    finish.assert_not_called()
    assert worker.req_states.total_len.gpu[req_index].item() == len(prompt)


def test_multibatch_native_sort_and_reused_slots_preserve_member_results():
    worker = make_native_v2_worker([1], capacity=7)
    states = worker.req_states
    prompts = {"a": [101] * 9, "b": [202] * 2, "c": [303] * 7, "d": [404] * 5}
    for request_id in ("unused-x", "a", "unused-y", "b", "c"):
        tokens = prompts.get(request_id, [1])
        states.add_request(request_id, len(tokens), tokens, 0, max_tokens=3)
    worker._remove_request("unused-x")
    worker._remove_request("unused-y")
    states.add_request("d", 5, prompts["d"], 0, max_tokens=3)
    states.apply_staged_writes()
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {r: len(tokens) for r, tokens in prompts.items()}
    scheduled.total_num_scheduled_tokens = 23
    scheduled.gr_stage_metadata = {
        "a": replace(
            metadata(),
            request_ids=tuple(prompts),
            num_prompt_tokens=23,
            num_prompt_tokens_by_request=(9, 2, 7, 5),
        )
    }

    def native_prefill(runner, scheduled):
        batch_state, _ = runner.gather_batch_req_state(scheduled, dummy_run=False)
        batch = runner.prepare_inputs(
            scheduled, batch_state, BatchExecutionDescriptor(CUDAGraphMode.FULL, 24, 4)
        )
        assert batch.req_ids == ["b", "d", "c", "a"]
        runner.execute_model_state = SimpleNamespace(
            input_batch=batch, hidden_states=batch.input_ids.float()[:, None]
        )

    result = CudaStageHooksV2(worker).prefill_logits(native_prefill, scheduled)
    assert result[:, 0].tolist() == [101, 202, 303, 404]
    for request_id, tokens in prompts.items():
        row = states.req_id_to_index[request_id]
        assert states.num_computed_tokens.gpu[row].item() == len(tokens)
    assert states.num_computed_tokens.gpu[states.req_id_to_index["session-0"]].item() == 0


# The V2 surface shares the stage-runner core: ``CudaStageHooksV2`` only
# replaces the native execution surface, so the dispatch-session record has to
# be built, carried and re-checked here exactly as on the V1 surface.
@pytest.mark.l2
def test_v2_hooked_admission_records_the_session_once():
    worker = admission_worker()
    first = metadata(0)
    owner = get_stage_runner(worker, dispatch(first), hooks_type=CudaStageHooksV2)
    assert isinstance(owner.hooks, CudaStageHooksV2)
    record = owner._dispatch_session
    assert record is not None
    assert record.session_id == first.session_id
    owner.previous = first
    assert owner.admit(dispatch(metadata(1))) is not None
    assert owner._dispatch_session is record


@pytest.mark.l2
def test_v2_decode_does_not_revalidate_the_session_facts(monkeypatch):
    from vllm_gr.v1.worker.cuda import gpu_beam_stage_runner as stage_runner_module

    worker = admission_worker()
    first = metadata(0)
    owner = get_stage_runner(worker, dispatch(first), hooks_type=CudaStageHooksV2)
    owner.previous = first

    def explode(*args, **kwargs):
        raise AssertionError("Decode revalidation rebuilt the session record")

    monkeypatch.setattr(stage_runner_module, "validate_dispatch_session", explode)
    assert owner.admit(dispatch(metadata(1))) is not None


@pytest.mark.l2
def test_v2_decode_rejects_a_record_from_another_logical_batch():
    worker = admission_worker()
    first = metadata(0)
    owner = get_stage_runner(worker, dispatch(first), hooks_type=CudaStageHooksV2)
    owner.previous = first
    # Same session id, different logical batch: only the batch identity can
    # catch it, which is the check the step path kept.
    foreign = replace(metadata(0), request_ids=("session-0:1",))
    foreign_dispatch = SimpleNamespace(num_scheduled_tokens={"session-0:1": 7})
    setattr(foreign_dispatch, GR_STAGE_OUTPUT_ATTR, {"session-0": foreign})
    owner._dispatch_session = validate_dispatch_session(foreign_dispatch)
    with pytest.raises(
        BeamExecutionContractError, match="logical Beam batch identity changed across stages"
    ):
        owner.admit(dispatch(metadata(1)))


@pytest.mark.l2
def test_v2_decode_still_checks_native_token_extent():
    worker = admission_worker()
    first = metadata(0)
    owner = get_stage_runner(worker, dispatch(first), hooks_type=CudaStageHooksV2)
    owner.previous = first
    scheduled = dispatch(metadata(1))
    scheduled.scheduled_cached_reqs = SimpleNamespace(req_ids=[], resumed_req_ids=set())
    with pytest.raises(ValueError, match="no matching native request"):
        owner.admit(scheduled)
