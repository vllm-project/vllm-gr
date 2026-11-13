# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compare GR Prefill fast paths with native V2 request and input state."""

from dataclasses import fields, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from tests.kernels.v2_input_test_utils import (
    make_native_v2_worker,
    make_prefill_metadata,
)
from vllm_gr.v1.worker.gpu_prefill_v2 import (
    try_add_gr_prefill_requests,
    try_prepare_gr_prefill_inputs_v2,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


def _registration_worker(capacity=8):
    worker = make_native_v2_worker([1], capacity=capacity)
    worker.use_pp = False
    worker.dp_size = 1
    worker.parallel_config = SimpleNamespace(
        tensor_parallel_size=1, prefill_context_parallel_size=1
    )
    worker.supports_mm_inputs = False
    worker.is_encoder_decoder = False
    worker.is_pooling_model = False
    worker.is_first_pp_rank = worker.is_last_pp_rank = True
    worker.lora_config = worker.speculative_config = None
    worker.model_state.add_request = Mock()
    worker.model_state.apply_staged_writes = Mock()
    worker.lora_state.add_request = Mock()
    worker.prompt_logprobs_worker = Mock()
    worker.sampler = Mock()
    worker.block_tables = BlockTables([16], capacity, 64, [4], worker.device, [16])
    return worker


def _scheduled(batch, cached=0):
    scheduled = SchedulerOutput.make_empty()
    for i in range(batch):
        tokens = [10 + i] * (9 - i)
        request = NewRequestData(
            req_id=f"member-{i}",
            prompt_token_ids=tokens,
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=3),
            pooling_params=None,
            block_ids=([i + 1],),
            num_computed_tokens=cached,
            lora_request=None,
            prefill_token_ids=tokens,
        )
        scheduled.scheduled_new_reqs.append(request)
        scheduled.num_scheduled_tokens[request.req_id] = len(tokens) - cached
    requests = scheduled.scheduled_new_reqs
    counts = tuple(len(r.prompt_token_ids) for r in requests)
    scheduled.total_num_scheduled_tokens = sum(scheduled.num_scheduled_tokens.values())
    scheduled.gr_stage_metadata = {
        "session-0": replace(
            make_prefill_metadata(),
            request_ids=tuple(r.req_id for r in requests),
            num_prompt_tokens=sum(counts),
            num_prompt_tokens_by_request=counts,
        )
    }
    return scheduled


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("cached", [0, 2])
def test_registration_preserves_native_state_and_slot_reuse(batch, cached):
    native, fast = _registration_worker(), _registration_worker()
    # Replacing the same IDs exercises removal and re-admission of native slots.
    for iteration in range(2):
        scheduled = _scheduled(batch, cached)
        for worker in (native, fast):
            worker._current_scheduler_output = scheduled
        GPUModelRunner.add_requests(native, scheduled)
        assert try_add_gr_prefill_requests(fast, scheduled)
        for worker in (native, fast):
            worker.block_tables.apply_staged_writes()
        assert fast.req_states.req_id_to_index == native.req_states.req_id_to_index
        for request in scheduled.scheduled_new_reqs:
            index = fast.req_states.req_id_to_index[request.req_id]
            for name in ("prompt_len", "prefill_len", "total_len", "num_computed_tokens"):
                torch.testing.assert_close(
                    getattr(fast.req_states, name).gpu[index],
                    getattr(native.req_states, name).gpu[index],
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                fast.req_states.all_token_ids.gpu[index, : request.prompt_len],
                native.req_states.all_token_ids.gpu[index, : request.prompt_len],
                rtol=0,
                atol=0,
            )
            assert fast.block_tables.block_tables[0].gpu[index, 0].item() == request.block_ids[0][0]
        assert (
            fast.model_state.add_request.call_args_list
            == native.model_state.add_request.call_args_list
        )
        assert (
            fast.prompt_logprobs_worker.add_request.call_args_list
            == native.prompt_logprobs_worker.add_request.call_args_list
        )
        assert fast.model_state.apply_staged_writes.call_count == iteration + 1
        fast.sampler.add_request.assert_not_called()
        fast.sampler.apply_staged_writes.assert_not_called()
        assert native.sampler.add_request.call_count == batch * (iteration + 1)


@pytest.mark.parametrize("unsupported", ["ordinary", "mixed", "cached", "logprobs", "spec"])
def test_registration_declines_entire_batch_before_mutation(unsupported):
    worker = _registration_worker()
    scheduled = _scheduled(4)
    worker._current_scheduler_output = scheduled
    if unsupported == "ordinary":
        scheduled.gr_stage_metadata = {}
    elif unsupported == "mixed":
        scheduled.scheduled_new_reqs[-1].req_id = "ordinary"
    elif unsupported == "cached":
        scheduled.scheduled_cached_reqs.req_ids = ["cached"]
    elif unsupported == "logprobs":
        scheduled.scheduled_new_reqs[-1].sampling_params.prompt_logprobs = 1
    else:
        scheduled.scheduled_spec_decode_tokens = {"member-0": [3]}
    original = worker.req_states.req_id_to_index.copy()
    assert not try_add_gr_prefill_requests(worker, scheduled)
    assert worker.req_states.req_id_to_index == original
    worker.model_state.add_request.assert_not_called()
    worker.model_state.apply_staged_writes.assert_not_called()


def _fresh_registration_worker():
    worker = _registration_worker(capacity=10)
    worker.req_states = RequestState(10, 2048, 8192, 0, 512, worker.device)
    worker.model_state = DefaultModelState.__new__(DefaultModelState)
    worker.model_state.rope_state = None
    worker.model_state.prompt_embeds_state = None
    worker.block_tables = BlockTables([16], 10, 8192, [128], worker.device, [16])
    worker.req_states.all_token_ids.gpu.fill_(-11)
    worker.block_tables.block_tables[0].gpu.fill_(-12)
    return worker


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("numpy_tokens", [False, True])
@pytest.mark.parametrize("cached", [0, 2])
def test_fused_registration_matches_native_after_batch_reuse(batch, numpy_tokens, cached):
    from vllm_gr.v1.worker.numpy_staging import install_numpy_staging

    install_numpy_staging()
    native, fast = _fresh_registration_worker(), _fresh_registration_worker()
    fast.req_states.apply_staged_writes = Mock(wraps=fast.req_states.apply_staged_writes)
    for iteration in range(3):
        scheduled = _scheduled(batch)
        for i, request in enumerate(scheduled.scheduled_new_reqs):
            length = (1, 257, 513, 1024)[(i + iteration) % 4]
            tokens = np.arange(length, dtype=np.int32) % 511
            tokens.flags.writeable = False
            request.prompt_token_ids = tokens if numpy_tokens else tokens.tolist()
            request.prefill_token_ids = request.prompt_token_ids
            request.block_ids = (list(range(1 + i * 128, 1 + i * 128 + (length + 15) // 16)),)
            request.num_computed_tokens = min(cached, length - 1)
            scheduled.num_scheduled_tokens[request.req_id] = length - request.num_computed_tokens
        scheduled.total_num_scheduled_tokens = sum(scheduled.num_scheduled_tokens.values())
        counts = tuple(len(r.prompt_token_ids) for r in scheduled.scheduled_new_reqs)
        scheduled.gr_stage_metadata["session-0"] = replace(
            scheduled.gr_stage_metadata["session-0"],
            num_prompt_tokens=sum(counts),
            num_prompt_tokens_by_request=counts,
        )
        if iteration:
            scheduled.scheduled_new_reqs.reverse()
        for worker in (native, fast):
            worker._current_scheduler_output = scheduled
        GPUModelRunner.add_requests(native, scheduled)
        assert try_add_gr_prefill_requests(fast, scheduled)
        fast.req_states.apply_staged_writes.assert_not_called()
        assert fast._gr_prefill_upload_buffers.slots[0] is not None
        assert not fast.block_tables.block_tables[0]._staged_write_indices
        for worker in (native, fast):
            worker.block_tables.apply_staged_writes()
        assert fast.req_states.req_id_to_index == native.req_states.req_id_to_index
        for name in (
            "prompt_len",
            "prefill_len",
            "total_len",
            "num_computed_tokens",
            "all_token_ids",
        ):
            torch.testing.assert_close(
                getattr(fast.req_states, name).gpu,
                getattr(native.req_states, name).gpu,
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            fast.block_tables.block_tables[0].gpu,
            native.block_tables.block_tables[0].gpu,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            fast.block_tables.num_blocks.gpu, native.block_tables.num_blocks.gpu, rtol=0, atol=0
        )
        for worker in (native, fast):
            for request in reversed(scheduled.scheduled_new_reqs):
                worker._remove_request(request.req_id)


@pytest.mark.parametrize("pending", ["total_len", "all_token_ids", "num_computed_tokens", "blocks"])
def test_registration_preserves_pending_writes_via_native_flush(pending):
    worker = _fresh_registration_worker()
    scheduled = _scheduled(4)
    tensor = (
        worker.block_tables.block_tables[0]
        if pending == "blocks"
        else getattr(worker.req_states, pending)
    )
    tensor.stage_write(0, 0, [37])
    worker.req_states.apply_staged_writes = Mock(wraps=worker.req_states.apply_staged_writes)
    worker._current_scheduler_output = scheduled
    assert try_add_gr_prefill_requests(worker, scheduled)
    worker.req_states.apply_staged_writes.assert_called_once()
    worker.block_tables.apply_staged_writes()
    assert tensor.gpu[0].reshape(-1)[0].item() == 37


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("after_launch", [False, True])
def test_fused_registration_submission_failure_keeps_queued_writes(
    monkeypatch, batch, after_launch
):
    from vllm_gr.ops.cuda import gr_prefill_inputs
    from vllm_gr.v1.worker import gpu_prefill_v2

    worker = _fresh_registration_worker()
    scheduled = _scheduled(batch)
    worker._current_scheduler_output = scheduled
    name = (
        "apply_gr_prefill_state_writes_v2"
        if batch == 1
        else "apply_gr_prefill_batch_state_writes_v2"
    )
    original_kernel = getattr(gr_prefill_inputs, name)
    recorded = []
    original_record = gpu_prefill_v2._PrefillUploadSlot.record

    def record(slot):
        original_record(slot)
        recorded.append(slot)

    def select(grid):
        def submit(*args, **kwargs):
            if after_launch:
                original_kernel[grid](*args, **kwargs)
            raise RuntimeError("submission")

        return submit

    monkeypatch.setattr(gpu_prefill_v2._PrefillUploadSlot, "record", record)
    failing_kernel = Mock()
    failing_kernel.__getitem__ = Mock(side_effect=select)
    monkeypatch.setattr(gr_prefill_inputs, name, failing_kernel)
    with pytest.raises(RuntimeError, match="submission"):
        try_add_gr_prefill_requests(worker, scheduled)
    (slot,) = recorded
    assert slot.submitted
    slot.done.synchronize()
    for tensor in (
        worker.req_states.total_len,
        worker.req_states.all_token_ids,
        worker.req_states.num_computed_tokens,
        worker.block_tables.block_tables[0],
    ):
        assert len(tensor._staged_write_indices) == batch


@pytest.mark.l2
@pytest.mark.gpu_test
def test_prefill_upload_slots_wait_before_reuse_and_growth():
    from vllm_gr.v1.worker.gpu_prefill_v2 import _PrefillUploadBuffers

    buffers = _PrefillUploadBuffers(torch.device("cuda"))
    first = buffers.acquire(8)
    second = buffers.acquire(8)
    assert first is not second
    first.submitted = second.submitted = True
    first.done = Mock(query=Mock(return_value=False))
    second.done = Mock(query=Mock(return_value=False))
    assert buffers.acquire(4) is first
    first.done.synchronize.assert_called_once()
    grown = buffers.acquire(16)
    second.done.synchronize.assert_called_once()
    assert grown is not second
    assert grown.cpu.numel() == 16
    assert len(buffers.slots) == 2


@pytest.mark.l2
@pytest.mark.gpu_test
@pytest.mark.parametrize("computed,scheduled_tokens", [(-1, 10), (9, 0), (2, 6), (2, 8)])
def test_fused_registration_requires_complete_remaining_prefill(computed, scheduled_tokens):
    from vllm_gr.v1.worker.gpu_prefill_v2 import _can_fuse_prefill_state_writes

    worker = _fresh_registration_worker()
    scheduled = _scheduled(1, cached=2)
    assert _can_fuse_prefill_state_writes(worker, scheduled, scheduled.scheduled_new_reqs)
    scheduled.scheduled_new_reqs[0].num_computed_tokens = computed
    scheduled.num_scheduled_tokens["member-0"] = scheduled_tokens
    assert not _can_fuse_prefill_state_writes(worker, scheduled, scheduled.scheduled_new_reqs)
    assert worker.req_states.num_reqs == 0


def _input_case(lengths, computed, query_lens, padded_tokens, cpu_ahead):
    worker = _fresh_registration_worker()
    worker.max_num_tokens = padded_tokens
    worker.input_buffers = InputBuffers(worker.max_num_reqs, padded_tokens, worker.device)
    # Native and fused preparation must each populate their own buffers. Matching
    # sentinels also make untouched padding deterministic for full-field parity.
    for name in ("input_ids", "positions", "query_start_loc", "seq_lens"):
        getattr(worker.input_buffers, name).fill_(-777)
    worker.input_buffers.is_padding.fill_(True)
    states = worker.req_states
    states.next_prefill_tokens.fill_(-777)
    states.add_request("unused", 1, [1], 0, max_tokens=3)
    states.add_request("occupied", 1, [1], 0, max_tokens=3)
    states.apply_staged_writes()
    worker._remove_request("unused")
    scheduled = _scheduled(len(lengths))
    for i, request in enumerate(scheduled.scheduled_new_reqs):
        # Distinguish both request rows and token offsets within each prompt.
        tokens = ((np.arange(lengths[i]) * 7 + i * 19) % 511).tolist()
        request.prompt_token_ids = request.prefill_token_ids = tokens
        request.num_computed_tokens = computed[i]
        request.block_ids = (list(range(1 + i * 128, 1 + i * 128 + (lengths[i] + 15) // 16)),)
        scheduled.num_scheduled_tokens[request.req_id] = query_lens[i]
    scheduled.total_num_scheduled_tokens = sum(query_lens)
    scheduled.gr_stage_metadata["session-0"] = replace(
        scheduled.gr_stage_metadata["session-0"],
        num_prompt_tokens=sum(lengths),
        num_prompt_tokens_by_request=tuple(lengths),
    )
    # Native slot allocation differs from both scheduler and sorted batch order.
    scheduled.scheduled_new_reqs.reverse()
    GPUModelRunner.add_requests(worker, scheduled)
    worker._remove_request("occupied")
    if cpu_ahead:
        for req_id in scheduled.num_scheduled_tokens:
            states.num_computed_tokens_np[states.req_id_to_index[req_id]] += 1
    worker.update_requests(scheduled)
    batch_state, _ = worker.gather_batch_req_state(scheduled, dummy_run=False)
    return worker, scheduled, batch_state


def _assert_inputs_match_native(monkeypatch, lengths, computed, query_lens, desc, cpu_ahead=False):
    native, native_scheduled, native_state = _input_case(
        lengths, computed, query_lens, desc.num_tokens, cpu_ahead
    )
    fast, scheduled, batch_state = _input_case(
        lengths, computed, query_lens, desc.num_tokens, cpu_ahead
    )
    expected = GPUModelRunner.prepare_inputs(native, native_scheduled, native_state, desc)

    def forbidden(*args, **kwargs):
        raise AssertionError("Input preparation must not read device state on the host")

    with monkeypatch.context() as guard:
        for method in ("item", "tolist", "cpu"):
            guard.setattr(torch.Tensor, method, forbidden)
        actual = try_prepare_gr_prefill_inputs_v2(fast, scheduled, batch_state, desc)
    assert actual is not None
    for field in fields(expected):
        name = field.name
        value = getattr(expected, name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(actual, name), value, rtol=0, atol=0, msg=name)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(actual, name), value)
        else:
            assert getattr(actual, name) == value, name
    torch.testing.assert_close(
        fast.req_states.next_prefill_tokens, native.req_states.next_prefill_tokens, rtol=0, atol=0
    )
    for name in ("input_ids", "positions", "query_start_loc", "seq_lens", "is_padding"):
        assert getattr(actual, name).data_ptr() == getattr(fast.input_buffers, name).data_ptr()
        assert getattr(actual, name).data_ptr() != getattr(expected, name).data_ptr()
    # A subsequent dispatch must not overwrite sampling indices retained by
    # asynchronous native postprocessing of the previous batch.
    another = try_prepare_gr_prefill_inputs_v2(fast, scheduled, batch_state, desc)
    assert another is not None
    assert another.idx_mapping.data_ptr() != actual.idx_mapping.data_ptr()
    assert another.logits_indices.data_ptr() != actual.logits_indices.data_ptr()
    assert another.cu_num_logits.data_ptr() != actual.cu_num_logits.data_ptr()


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 7, 8])
@pytest.mark.parametrize("case", ["fresh", "cached", "chunk", "cpu_ahead"])
@pytest.mark.parametrize("write_padding", [False, True])
def test_fused_inputs_match_native_with_padding_and_reused_slots(
    monkeypatch, batch, case, write_padding
):
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", str(int(write_padding)))
    lengths = list(range(9, 9 - batch, -1))
    computed = [int(case == "cached")] * batch
    query_lens = [
        length - cached - int(case in ("chunk", "cpu_ahead"))
        for length, cached in zip(lengths, computed)
    ]
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 64, batch + 1)
    _assert_inputs_match_native(
        monkeypatch, lengths, computed, query_lens, desc, cpu_ahead=case == "cpu_ahead"
    )


@pytest.mark.parametrize(
    "lengths,computed,query_lens,cpu_ahead,padded_reqs",
    [
        pytest.param([255, 256, 257], [0, 0, 0], [255, 256, 257], False, 4, id="block-boundaries"),
        pytest.param([513, 1024], [255, 257], [258, 767], False, 3, id="cached-long"),
        pytest.param([513, 1024], [0, 1], [256, 513], True, 3, id="partial-cpu-ahead"),
        pytest.param([257, 513], [0, 0], [257, 513], False, None, id="no-request-padding"),
        pytest.param([257], [0], [257], False, 2, id="single-request-padded"),
    ],
)
@pytest.mark.parametrize("write_padding", [False, True])
def test_fused_inputs_match_native_across_token_blocks(
    monkeypatch, lengths, computed, query_lens, cpu_ahead, padded_reqs, write_padding
):
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", str(int(write_padding)))
    # Exercise padding writes in later Triton blocks as well as long prompts.
    padded_tokens = ((sum(query_lens) + 511) // 256) * 256
    desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL if padded_reqs else CUDAGraphMode.PIECEWISE,
        padded_tokens,
        padded_reqs,
    )
    _assert_inputs_match_native(monkeypatch, lengths, computed, query_lens, desc, cpu_ahead)


@pytest.mark.parametrize("unsupported", ["ordinary", "mixed", "decode", "mapping", "spec"])
def test_fused_inputs_decline_before_buffer_mutation(unsupported):
    worker = _registration_worker()
    scheduled = _scheduled(4)
    GPUModelRunner.add_requests(worker, scheduled)
    worker.model_state = DefaultModelState.__new__(DefaultModelState)
    worker.model_state.rope_state = None
    worker.max_num_tokens = 64
    worker.update_requests(scheduled)
    batch_state, _ = worker.gather_batch_req_state(scheduled, dummy_run=False)
    if unsupported == "ordinary":
        scheduled.gr_stage_metadata = {}
    elif unsupported == "mixed":
        scheduled.num_scheduled_tokens["ordinary"] = 1
    elif unsupported == "decode":
        index = batch_state.idx_mapping_np[-1]
        worker.req_states.num_computed_tokens_np[index] = worker.req_states.prompt_len.np[index]
    elif unsupported == "mapping":
        batch_state.idx_mapping_np[:] = 0
    else:
        scheduled.scheduled_spec_decode_tokens = {"member-0": [3]}
    worker.input_buffers.input_ids.fill_(-777)
    assert (
        try_prepare_gr_prefill_inputs_v2(
            worker, scheduled, batch_state, BatchExecutionDescriptor(CUDAGraphMode.FULL, 64, 4)
        )
        is None
    )
    assert worker.input_buffers.input_ids.tolist() == [-777] * 64
