# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise GR stage boundaries against real ModelRunner V2 request state."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.states import RequestState

from tests.kernels.test_gpu_beam_stage_runner import dispatch, metadata, runtime
from vllm_gr.v1.engine.gr_async_scheduler import GR_WORKER_RESULT_ATTR
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import make_v2_stage_hooks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


def _native_worker(prompt, computed=0, capacity=2):
    worker: Any = GPUModelRunner.__new__(GPUModelRunner)
    worker.device = torch.device("cuda", torch.accelerator.current_device_index())
    worker.max_num_reqs = capacity
    worker.use_dcp = False
    worker.input_buffers = InputBuffers(capacity, 64, worker.device)
    worker.req_states = RequestState(capacity, 16, 64, 0, 512, worker.device)
    worker.req_states.add_request("session-0", len(prompt), prompt, computed)
    worker.req_states.apply_staged_writes()
    worker.block_tables = SimpleNamespace(apply_staged_writes=Mock())
    worker.encoder_cache = None
    worker.prompt_logprobs_worker = None
    worker.lora_state = SimpleNamespace(remove_request=Mock())
    worker.sampler = None
    worker.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    return worker


@pytest.mark.parametrize("cached", [0, 2, 7])
def test_full_prefill_advances_native_state_without_counting_padding(monkeypatch, cached):
    prompt = list(range(10, 18))
    worker = _native_worker(prompt, cached)
    hooks = make_v2_stage_hooks(worker)
    req_index = worker.req_states.req_id_to_index["session-0"]
    observed = []
    kv = torch.zeros(len(prompt), dtype=torch.int32, device=worker.device)
    kv[:cached] = torch.tensor(prompt[:cached], dtype=torch.int32, device=worker.device)

    def native_prefill(runner, scheduled):
        runner.update_requests(scheduled)
        count = scheduled.total_num_scheduled_tokens
        # Padding must not advance the request's logical token count.
        desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 1)
        batch = runner.prepare_inputs(scheduled, desc)
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
    scheduled.gr_stage_metadata = {"session-0": metadata()}
    scheduled.num_scheduled_tokens = {"session-0": count}
    scheduled.total_num_scheduled_tokens = count
    scheduled.scheduled_cached_reqs.req_ids = ["session-0"]
    scheduled.scheduled_cached_reqs.num_computed_tokens = [cached]
    scheduled.scheduled_cached_reqs.new_block_ids = [None]
    with monkeypatch.context() as guard:
        logits = hooks.prefill_logits(native_prefill, scheduled)
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
    worker = _native_worker(list(range(7)))
    worker._beam_sessions = owner.worker._beam_sessions
    owner.worker = worker
    owner.hooks = make_v2_stage_hooks(worker)
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
    worker.req_states.add_request("next-session", 1, [1], 0)
    assert worker.req_states.num_reqs == 1


def test_multibatch_native_sort_and_reused_slots_preserve_member_results():
    from dataclasses import replace

    worker = _native_worker([1], capacity=7)
    states = worker.req_states
    prompts = {"a": [101] * 9, "b": [202] * 2, "c": [303] * 7, "d": [404] * 5}
    for request_id in ("unused-x", "a", "unused-y", "b", "c"):
        tokens = prompts.get(request_id, [1])
        states.add_request(request_id, len(tokens), tokens, 0)
    worker._remove_request("unused-x")
    worker._remove_request("unused-y")
    states.add_request("d", 5, prompts["d"], 0)
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
        batch = runner.prepare_inputs(
            scheduled, BatchExecutionDescriptor(CUDAGraphMode.FULL, 24, 4)
        )
        assert batch.req_ids == ["b", "d", "c", "a"]
        runner.execute_model_state = SimpleNamespace(
            input_batch=batch, hidden_states=batch.input_ids.float()[:, None]
        )

    result = make_v2_stage_hooks(worker).prefill_logits(native_prefill, scheduled)
    assert result[:, 0].tolist() == [101, 202, 303, 404]
    for request_id, tokens in prompts.items():
        row = states.req_id_to_index[request_id]
        assert states.num_computed_tokens.gpu[row].item() == len(tokens)
    assert states.num_computed_tokens.gpu[states.req_id_to_index["session-0"]].item() == 0
