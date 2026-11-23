# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA adapter tests: independent resources, dispatch snapshots and terminal KV."""

from __future__ import annotations

import gc
import subprocess
import sys
from abc import abstractmethod
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import numpy as np
import pytest
import torch
from torch.cuda.streams import Event as CUDAEvent

from tests.helpers.beam_requests import make_batch_request
from tests.helpers.paths import REPO_ROOT
from tests.kernels.beam_stage_test_utils import (
    admission_worker,
    dispatch,
    metadata,
    prepare_binding,
    runtime,
)
from vllm_gr.config import SUPPORTED_BEAM_BATCH_SIZES
from vllm_gr.ops.cuda.beam_kv_cache import (
    cache_unshared_kv_graph,
    copy_suffix_kv_graph,
)
from vllm_gr.v1.beam.dispatch import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRDeviceStateRef,
)
from vllm_gr.v1.worker import gr_context as _gr_context
from vllm_gr.v1.worker.cuda.beam_search_runner import BeamSearchRunner
from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
    LOCAL_TILE_SIZE,
    SLICE_AUTO,
    GPUBeamStageRunner,
    _batch_finished,
    _HeadLogits,
    get_stage_runner,
    head_slice_plan,
    validate_dispatch,
)
from vllm_gr.v1.worker.cuda.gr_inputs import GRInputBuffers, bind_device_inputs
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


def test_batch_finishes_only_when_all_members_finish_or_any_member_errors():
    assert not _batch_finished(torch.tensor([True, False]), torch.tensor([0, 0])).item()
    assert _batch_finished(torch.tensor([True, True]), torch.tensor([0, 0])).item()
    assert _batch_finished(torch.tensor([False, False]), torch.tensor([0, 1])).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_admission_accepts_full_attention(dtype):
    worker = admission_worker(dtype=dtype)
    owner = get_stage_runner(worker, dispatch(metadata()), hooks_type=CudaStageHooksV1)
    assert owner.beam.session.is_active
    assert worker._beam_context.registry.num_active_sessions == 1
    assert worker._beam_context._gr_execution_owner is owner
    assert owner.beam.workspace.execution_owner is owner
    assert worker._beam_context.max_batch == owner.beam.workspace.max_batch
    assert owner.beam.workspace.max_batch == max(SUPPORTED_BEAM_BATCH_SIZES)
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)


@pytest.mark.l2
@pytest.mark.parametrize("failure", ["abstract", "constructor"])
@pytest.mark.parametrize("stale_cache", [False, True])
def test_failed_hooks_construction_preserves_admission(failure, stale_cache):
    class Incomplete(CudaStageHooksV1):  # type: ignore[misc]
        @abstractmethod
        def decode_block_ids(self, session_id):
            raise NotImplementedError

    class BrokenConstructor(CudaStageHooksV1):  # type: ignore[misc]
        def __init__(self, worker):
            raise RuntimeError("hooks construction failed")

    worker = admission_worker()
    cache = {1: object()} if stale_cache else {}
    worker._gr_gpu_stage_runner_cache = cache
    graph = worker._beam_decode_graph = Mock()
    before_cache = dict(cache)
    hooks_type = Incomplete if failure == "abstract" else BrokenConstructor
    error = TypeError if failure == "abstract" else RuntimeError
    with pytest.raises(error):
        get_stage_runner(worker, dispatch(metadata()), hooks_type=hooks_type)

    context = worker._beam_context
    assert context.registry.num_active_sessions == 0
    assert getattr(context, "_gr_execution_owner", None) is None
    assert worker._beam_postprocess_workspaces.require.return_value.execution_owner is None
    assert not getattr(worker, "_beam_sessions", {})
    assert getattr(worker, "_gr_gpu_stage_runner", None) is None
    assert cache == before_cache
    graph.clear.assert_not_called()

    owner = get_stage_runner(worker, dispatch(metadata()), hooks_type=CudaStageHooksV1)
    assert context.registry.num_active_sessions == 1
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)
    assert context.registry.num_active_sessions == 0


@pytest.mark.parametrize("windows", [(4, 4), (None, 4), (4, None), (None, 1)])
def test_admission_rejects_sliding_window_before_resource_creation(windows, monkeypatch):
    worker = admission_worker(windows)
    create_context = Mock(wraps=_gr_context._create_beam_context)
    monkeypatch.setattr(_gr_context, "_create_beam_context", create_context)
    with pytest.raises(ValueError, match="does not support sliding-window attention") as exc:
        get_stage_runner(worker, dispatch(metadata()), hooks_type=CudaStageHooksV1)
    first_window_layer = next(i for i, window in enumerate(windows) if window is not None)
    assert f"layer.{first_window_layer}" in str(exc.value)
    create_context.assert_not_called()
    worker._beam_postprocess_workspaces.require.assert_not_called()
    assert getattr(worker, "_beam_context", None) is None
    assert getattr(worker, "_gr_gpu_stage_runner", None) is None
    assert not getattr(worker, "_beam_sessions", {})


def test_admission_rejects_sliding_window_before_reuse_reset(monkeypatch):
    worker = admission_worker()
    owner = get_stage_runner(worker, dispatch(metadata()), hooks_type=CudaStageHooksV1)
    owner.previous = metadata(2)
    state = owner.beam.state
    state.sequence.fill_(123)
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)
    before = state.sequence.clone()
    reset = Mock(wraps=state.reset)
    monkeypatch.setattr(state, "reset", reset)
    context = worker._beam_context
    allocate = Mock(wraps=context.registry.allocate)
    monkeypatch.setattr(context.registry, "allocate", allocate)
    worker._beam_postprocess_workspaces.require.reset_mock()
    worker.compilation_config.static_forward_context["layer.1"].impl.sliding_window = (3, 0)

    with pytest.raises(ValueError, match="does not support sliding-window attention"):
        get_stage_runner(worker, dispatch(metadata()), hooks_type=CudaStageHooksV1)
    reset.assert_not_called()
    allocate.assert_not_called()
    worker._beam_postprocess_workspaces.require.assert_not_called()
    torch.testing.assert_close(state.sequence, before)
    assert worker._gr_gpu_stage_runner is owner and owner.closed
    assert worker._beam_context is context and context.registry.num_active_sessions == 0
    assert context._gr_execution_owner is None
    assert owner.beam.workspace.execution_owner is None
    assert not worker._beam_sessions


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_continuous_dispatch_snapshots_and_resource_isolation(dtype, n):
    owner = runtime(dtype, n)
    state, ws = owner.beam.state, owner.beam.workspace
    protected = {k: v[1:].clone() for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
    ws.selected_token_ids[1].fill_(987)
    ws.prefix_token_ids[1].fill_(789)
    ws.prefix_token_ids[2].fill_(789)
    for pool in owner.beam.session.pools:
        pool.unshared_key[1:].fill_(123)
        pool.unshared_value[1:].fill_(456)
    receipts = []
    for stage in range(n):
        execution = owner.admit(dispatch(metadata(stage, n=n)))
        if stage:
            prepare_binding(owner, execution.entries[0])
            current = torch.ones((4, 1, 8), device="cuda", dtype=dtype)
            for pool in owner.beam.session.pools:
                cache_unshared_kv_graph(
                    pool.unshared_key,
                    pool.unshared_value,
                    current,
                    current,
                    0,
                    owner.beam.session.context.decode_step_buf,
                    owner.execution_mask,
                )
        owner.logits = torch.zeros((1 if stage == 0 else 4, 6), dtype=dtype, device="cuda")
        receipts.append(owner.sample())
        if stage == 1 and n == 3:
            # Dispatch 1 was enqueued while dispatch 0 remained unconsumed.
            with pytest.raises(RuntimeError, match="both GR dispatch"):
                owner.admit(dispatch(metadata(2, n=3)))
            first = receipts[0].get_output()
            assert getattr(first, GR_WORKER_RESULT_ATTR)["session-0"].finished == (n == 1)
    for stage, receipt in enumerate(receipts):
        result = getattr(receipt.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
        assert result.produced_token_count == 1
        assert result.stage_index == stage
        assert result.finished == (stage == n - 1)
        assert (result.terminal_result is not None) == (stage == n - 1)
        assert result.resources_released == (stage == n - 1)
        if result.terminal_result is not None:
            assert result.terminal_result.output is not None
    assert state.num_generated_tokens.tolist() == [n, 0, 0]
    assert state.num_decode_steps.tolist() == [n - 1, 0, 0]
    for name, value in protected.items():
        torch.testing.assert_close(getattr(state, name)[1:], value)
    assert ws.selected_token_ids[1].tolist() == [987] * 4
    assert state.num_completed[0] > 0
    assert owner.final_output.final_d2h_count == 1
    assert owner.closed
    assert not owner.beam.session.context.registry.has_session("session-0")
    cleanup = owner.worker._update_states.call_args.args[0]
    assert cleanup.total_num_scheduled_tokens == 0
    assert cleanup.finished_req_ids == {"session-0"}
    state.reset(0)
    for name, value in protected.items():
        torch.testing.assert_close(getattr(state, name)[1:], value)
    assert ws.prefix_token_ids[1].eq(789).all()
    assert ws.prefix_token_ids[2].eq(789).all()
    for pool in owner.beam.session.pools:
        assert pool.unshared_key[1:].eq(123).all()
        assert pool.unshared_value[1:].eq(456).all()


@pytest.mark.parametrize("candidate_width", [2, 3, 6, 8])
def test_decode_stages_sample_k_candidates_per_row_when_k_differs_from_w(
    candidate_width, monkeypatch
):
    """W=4 rows with K < W and K > W: every stage asks the backend for K, not W.

    The kernel-level K != W tests stop at the operators; this drives the
    stage runner itself through Prefill and two Decode stages, so the wire K,
    the workspace geometry, the candidate buffers and the completed-result
    capacity all have to agree for the session to finish.
    """

    n = 3
    owner = runtime(n=n, candidate_width=candidate_width)
    state = owner.beam.state
    assert state.candidate_width == candidate_width
    assert state.completed_capacity == candidate_width + (n - 1) * 4 * candidate_width
    assert tuple(owner.tokens.shape) == (4, candidate_width)
    requested = []
    backend_type = type(owner.backend)
    select = backend_type.select_out_unchecked

    def _record(self, logits, metadata, *args, **kwargs):
        requested.append((logits.shape[0], metadata.top_k))
        return select(self, logits, metadata, *args, **kwargs)

    # The backend uses slots, so record at the class rather than the instance.
    monkeypatch.setattr(backend_type, "select_out_unchecked", _record)
    results = []
    for stage in range(n):
        execution = owner.admit(dispatch(metadata(stage, n=n, candidate_width=candidate_width)))
        if stage:
            prepare_binding(owner, execution.entries[0])
        owner.logits = torch.zeros((1 if stage == 0 else 4, 6), dtype=torch.float16, device="cuda")
        # Consume each stage before the next admit: the runner has two slots.
        output = owner.sample().get_output()
        results.append(getattr(output, GR_WORKER_RESULT_ATTR)["session-0"])
    assert requested == [(1, candidate_width), (4, candidate_width), (4, candidate_width)]
    assert [result.stage_index for result in results] == list(range(n))
    final = results[-1]
    assert final.finished
    assert final.terminal_result is not None and final.terminal_result.output is not None
    assert state.num_generated_tokens.tolist()[0] == n
    assert 0 < int(state.num_completed[0]) <= state.completed_capacity


@pytest.mark.parametrize("graph", [False, True])
def test_terminal_successor_preserves_all_layer_kv(graph):
    owner = runtime()
    state = owner.beam.state
    current = torch.ones((4, 1, 8), device="cuda", dtype=torch.float16)
    entry = owner.admit(dispatch(metadata())).entries[0]
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    owner.sample().get_output()
    entry = replace(entry, metadata=metadata(1), row_count=4)
    prepare_binding(owner, entry)
    for pool in owner.beam.session.pools:
        pool.unshared_key.fill_(123)
        pool.unshared_value.fill_(456)

    def step():
        owner.prepare_decode(entry)
        for pool in owner.beam.session.pools:
            cache_unshared_kv_graph(
                pool.unshared_key,
                pool.unshared_value,
                current,
                current,
                0,
                owner.beam.session.context.decode_step_buf,
                owner.execution_mask,
            )
            copy_suffix_kv_graph(
                pool.unshared_key,
                pool.unshared_value,
                owner.beam.session.suffix_k_buf,
                owner.beam.session.suffix_v_buf,
                0,
                owner.beam.session.context.decode_step_buf,
                beam_width=4,
                active_mask=owner.execution_mask,
            )

    step()
    replay = step
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        replay = g.replay
    state.is_finished[0] = True
    before = [(p.unshared_key.clone(), p.unshared_value.clone()) for p in owner.beam.session.pools]
    replay()
    for pool, (key, value) in zip(owner.beam.session.pools, before):
        torch.testing.assert_close(pool.unshared_key, key)
        torch.testing.assert_close(pool.unshared_value, value)
    assert owner.beam.session.suffix_k_buf[:4].count_nonzero() == 0
    assert owner.execution_mask.count_nonzero() == 0
    assert owner.positions.tolist() == [0] * 4


def test_partial_prefill_and_dispatch_identity_rejected_before_writes():
    owner = runtime()
    with pytest.raises(ValueError, match="complete"):
        owner.admit(dispatch(metadata(output=False)))
    assert owner.pending is None and owner.previous is None
    assert not any(owner.control_busy)
    assert owner.beam.state.num_generated_tokens.sum() == 0
    owner.admit(dispatch(metadata()))
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    owner.sample().get_output()
    before = owner.beam.state.sequence.clone()
    with pytest.raises(ValueError, match="dispatch order"):
        owner.admit(dispatch(metadata(stage=1, dispatch=99)))
    # A foreign batch that arrives after a valid stage is rejected either as an
    # identity change (compared against ``previous``) or as a binding change;
    # which one fires first is precedence, not behaviour.
    with pytest.raises(
        ValueError,
        match="logical Beam batch identity changed across stages|shared execution binding",
    ):
        other = replace(
            metadata(stage=1, dispatch=1),
            session_id="other",
            request_ids=("other",),
            output_device_state=GRDeviceStateRef("other", 1),
            input_device_state=GRDeviceStateRef("other", 0),
        )
        owner.admit(dispatch(other))
    mixed = dispatch(metadata(1, dispatch=1))
    mixed.num_scheduled_tokens["ordinary"] = 1
    with pytest.raises(ValueError, match="immutable logical batch"):
        owner.admit(mixed)
    torch.testing.assert_close(owner.beam.state.sequence, before)


def test_native_identity_and_extent_rejected_before_admission():
    owner = runtime()
    invalid = dispatch(metadata())
    invalid.scheduled_cached_reqs = SimpleNamespace(req_ids=[], resumed_req_ids=set())
    with pytest.raises(ValueError, match="no matching native request"):
        owner.admit(invalid)
    assert owner.pending is None and not any(owner.control_busy)
    invalid = dispatch(metadata())
    invalid.total_num_scheduled_tokens = 99
    with pytest.raises(ValueError, match="total disagrees"):
        owner.admit(invalid)
    assert owner.previous is None


@pytest.mark.parametrize("fault", ["protocol", "item", "prompt"])
def test_session_init_binding_rejected_before_writes(fault):
    owner = runtime()
    m = metadata()
    request = make_batch_request()
    request.items[0].request.prompt_token_ids = np.full(m.num_prompt_tokens, 1, dtype=np.int64)
    request = msgspec.structs.replace(request, beam_params=m.beam_params)
    if fault == "protocol":
        request = msgspec.structs.replace(request, protocol_version=request.protocol_version + 1)
    elif fault == "item":
        request = msgspec.structs.replace(
            request, items=(msgspec.structs.replace(request.items[0], item_index=1),)
        )
    else:
        request.items[0].request.prompt_token_ids = np.ones(1, dtype=np.int64)
    before = owner.beam.state.sequence.clone()
    with pytest.raises(ValueError, match="session_init disagrees|protocol_version|item_index"):
        owner.admit(dispatch(replace(m, session_init=request)))
    assert owner.pending is None and owner.previous is None and not any(owner.control_busy)
    torch.testing.assert_close(owner.beam.state.sequence, before)


def test_device_binding_preserves_surrounding_storage_and_strided_prefix():
    owner = runtime()
    s = owner.beam.state
    s.is_initialized[0] = True
    s.num_generated_tokens[0] = 2
    s.num_decode_steps[0] = 1
    s.active_mask[0] = torch.tensor([True, False, True, False], device="cuda")
    s.input_ids[0] = torch.tensor([3, 4, 5, 6], device="cuda")
    s.sequence[0, :, :2] = torch.tensor([[1, 2]] * 4, device="cuda")
    prefix = torch.full((3, 4, 7), 99, device="cuda", dtype=torch.int32)[:, :, :3]
    tokens = torch.full((9,), 77, device="cuda", dtype=torch.int32)
    positions = tokens.clone()
    offsets = owner.metadata.suffix_cu_seqlens_k_device
    buffers = GRInputBuffers(4, 64, 16, "cuda")
    prepared = buffers.prepare_cpu(metadata(2), owner.binding, block_ids=[[1]], slot=0)
    owner.input_ids = tokens[3:7]
    owner.positions = positions[3:7]
    owner.prefixes = prefix[1]
    owner.device_positions = torch.tensor(prepared.layout["positions"], device="cuda")
    bind_device_inputs(prepared, owner)
    assert tokens.tolist() == [77, 77, 77, 3, 0, 5, 0, 77, 77]
    assert positions.tolist() == [77, 77, 77, 8, 0, 8, 0, 77, 77]
    assert prefix[1].tolist() == [[1, 2, -1], [-1, -1, -1], [1, 2, -1], [-1, -1, -1]]
    assert prefix[0].unique().tolist() == [99]
    assert offsets.tolist() == [0, 2, 4, 6, 8]


def test_multiple_entries_rejected_even_with_storage_capacity():
    m = metadata()
    output = dispatch(m)
    getattr(output, GR_STAGE_OUTPUT_ATTR)["other"] = replace(m, session_id="other")
    output.num_scheduled_tokens["other"] = 7
    with pytest.raises(ValueError, match="cannot mix ordinary"):
        validate_dispatch(output)


def test_multi_request_dispatch_keeps_member_token_extents_separate():
    m = replace(
        metadata(),
        num_prompt_tokens=12,
        request_ids=("session-0", "session-0:1"),
        num_prompt_tokens_by_request=(7, 5),
        kv_blocks=((11,), (21,)),
    )
    output = dispatch(m)
    output.num_scheduled_tokens["session-0:1"] = 5
    assert validate_dispatch(output) == (m,)

    output.num_scheduled_tokens["session-0:1"] = 4
    output.scheduled_new_reqs = (
        SimpleNamespace(req_id="session-0", prompt_token_ids=[1] * 7, num_computed_tokens=0),
        SimpleNamespace(req_id="session-0:1", prompt_token_ids=[1] * 5, num_computed_tokens=0),
    )
    with pytest.raises(ValueError, match="partial Prefill"):
        validate_dispatch(output)


def test_all_eos_successor_and_error_preserve_committed_state():
    owner = runtime()
    state = owner.beam.state
    owner.admit(dispatch(metadata()))
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    owner.sample().get_output()
    owner.admit(dispatch(metadata(1)))
    owner.logits = torch.full((4, 6), float("inf"), device="cuda", dtype=torch.float16)
    # Populate valid prefixes for the candidate error to be an invalid score,
    # not merely an empty-table lookup.
    prepare_binding(owner, owner.pending[0].entries[0])
    protected = {
        name: value.clone()
        for name, value in vars(state).items()
        if isinstance(value, torch.Tensor) and name != "error_code"
    }
    result = owner.sample()
    execution = owner.admit(dispatch(metadata(2)))
    prepare_binding(owner, execution.entries[0])
    owner.logits = torch.zeros((4, 6), device="cuda", dtype=torch.float16)
    successor = owner.sample()

    result = result.get_output()
    receipt = getattr(result, GR_WORKER_RESULT_ATTR)["session-0"]
    assert receipt.finished and receipt.produced_token_count == 0
    assert receipt.terminal_result.failure.code == "beam_device_error"
    assert not receipt.resources_released
    for name, value in protected.items():
        torch.testing.assert_close(getattr(state, name), value)
    successor_result = getattr(successor.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
    assert successor_result.produced_token_count == 0
    assert successor_result.resources_released
    assert owner.closed
    assert owner.execution_mask.count_nonzero() == 0


def test_early_terminal_receipt_defers_release_until_the_last_stage():
    """A queued look-ahead dispatch must still be admissible after the receipt.

    The Scheduler may already have issued the successor of a stage that turned
    terminal. Locally drained work therefore cannot release the owner: only a
    receipt from the declared last stage proves that nothing is still in
    transit.
    """
    owner = runtime()
    state = owner.beam.state
    owner.admit(dispatch(metadata()))
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    owner.sample().get_output()
    owner.admit(dispatch(metadata(1)))
    owner.logits = torch.full((4, 6), float("inf"), device="cuda", dtype=torch.float16)
    prepare_binding(owner, owner.pending[0].entries[0])
    protected = {
        name: value.clone()
        for name, value in vars(state).items()
        if isinstance(value, torch.Tensor) and name != "error_code"
    }

    # The successor is admitted only after the terminal receipt was consumed.
    receipt = getattr(owner.sample().get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
    assert receipt.finished and receipt.produced_token_count == 0
    assert receipt.terminal_result.failure.code == "beam_device_error"
    assert not receipt.resources_released
    assert not owner.closed
    for name, value in protected.items():
        torch.testing.assert_close(getattr(state, name), value)

    execution = owner.admit(dispatch(metadata(2)))
    prepare_binding(owner, execution.entries[0])
    owner.logits = torch.zeros((4, 6), device="cuda", dtype=torch.float16)
    successor = getattr(owner.sample().get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
    assert successor.produced_token_count == 0
    assert successor.resources_released
    assert owner.closed
    assert owner.execution_mask.count_nonzero() == 0


def test_release_requires_control_and_completion_consumer_drain():
    owner = runtime()
    owner.admit(dispatch(metadata()))
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    receipt = owner.sample()
    done = torch.Event()
    done.record()
    with pytest.raises(RuntimeError, match="output-consumer drain"):
        owner.release(owner.binding, consumer_done=done)
    receipt.get_output()
    completion = owner.completion_view(owner.binding)
    assert completion["sequence"].data_ptr() == owner.beam.state.sequence[0].data_ptr()
    with pytest.raises(ValueError, match="binding mismatch"):
        owner.release(replace(owner.binding, state_row=1), consumer_done=done)
    owner.cancel(owner.binding)
    owner.release(owner.binding, consumer_done=done)
    with pytest.raises(RuntimeError, match="live matching"):
        owner.completion_view(owner.binding)


@pytest.mark.parametrize("penalty,expected_first", [(0.0, 11), (2.0, 33), (-1.0, 11)])
def test_final_ranking_applies_length_penalty_without_changing_logprobs(penalty, expected_first):
    from vllm_gr.ops.cuda.beam_state import FINISH_LENGTH, FINISH_STOP

    owner = runtime()
    m = metadata()
    m = replace(
        m,
        beam_params=msgspec.structs.replace(m.beam_params, length_penalty=penalty),
        output_options=msgspec.structs.replace(m.output_options, num_return_sequences=2),
    )
    owner.admit(dispatch(m))
    state, row = owner.beam.state, owner.binding.state_row
    state.num_completed[row] = 2
    state.completed_token_ids[row, :2] = torch.tensor([11, 33], device="cuda")
    state.completed_scores[row, :2] = torch.tensor([-0.1, -0.15], device="cuda")
    state.completed_lengths[row, :2] = torch.tensor([1, 3], device="cuda")
    state.completed_finish_reasons[row, :2] = torch.tensor(
        [FINISH_STOP, FINISH_LENGTH], device="cuda"
    )
    state.completed_parent_ids[row, :2] = 0
    state.history_token_ids[row, :2, 0] = torch.tensor([22, 23], device="cuda")
    state.history_parent_ids[row, :2, 0] = 0
    owner.last_work.record()

    result = owner.final_output.consume(owner)

    assert result.failure is None
    sequences = result.output.items[0].sequences
    assert (sequences[0].stop_token_id or sequences[0].token_ids[-1]) == expected_first
    assert sorted(sequence.cumulative_logprob for sequence in sequences) == pytest.approx(
        [-0.15, -0.1]
    )


def test_workspace_bounds_and_exclusive_shared_scratch():
    owner = runtime()
    with pytest.raises(ValueError, match="workspace_index"):
        BeamSearchRunner(
            owner.beam.session,
            owner.beam.state,
            owner.beam.workspace,
            state_index=2,
            workspace_index=3,
        )
    second = BeamSearchRunner(
        owner.beam.session, owner.beam.state, owner.beam.workspace, state_index=1, workspace_index=0
    )
    values = (
        torch.zeros((1, 4), device="cuda", dtype=torch.int32),
        torch.zeros((1, 4), device="cuda"),
        torch.ones((1, 4), device="cuda", dtype=torch.bool),
    )
    with pytest.raises(RuntimeError, match="shared Beam execution scratch"):
        second.initialize_from_prefill(*values)


def test_fresh_metadata_updates_contents_without_changing_graph_addresses():
    owner = runtime()
    owner.metadata = None
    owner.worker.requests = {"session-0": SimpleNamespace(block_ids=([1],))}
    owner.worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer"], kv_cache_spec=SimpleNamespace(block_size=16))
        ]
    )
    owner.worker.parallel_config = SimpleNamespace()
    owner.worker.model_config = SimpleNamespace(
        max_model_len=64, get_num_attention_heads=lambda _: 1, get_num_kv_heads=lambda _: 1
    )
    first = metadata(1)
    owner.prepared = owner.prepare_cpu(first, 0)
    owner._bind_attention(first)
    old = owner.metadata
    graph = owner.graph = object()
    owner.worker.requests["session-0"].block_ids = ([2],)
    changed = replace(metadata(2), num_prompt_tokens=8, num_prompt_tokens_by_request=(8,))
    owner.prepared = owner.prepare_cpu(changed, 1)
    owner._bind_attention(changed)
    assert owner.metadata is not old
    assert owner.graph is graph
    assert owner.metadata.prefix_block_table.data_ptr() == old.prefix_block_table.data_ptr()
    assert owner.metadata.prefix_seqlens_kv_device.tolist() == [8]
    assert owner.metadata.prefix_block_table.tolist() == [[2, 0, 0, 0]]
    assert owner.metadata.suffix_seqlens_kv_cpu == [2] * 4
    assert old.suffix_seqlens_kv_cpu == [1] * 4
    assert owner.metadata.num_reqs == 1
    assert owner.metadata.query_start_loc_cpu == [0, 4]
    assert owner.metadata.prefix_max_seq_len == 64


@pytest.mark.parametrize("fatal", [False, True])
def test_terminal_materialization_preserves_fatal_error_channel(monkeypatch, fatal):
    owner = runtime(torch.float16, 1)
    owner.admit(dispatch(metadata(n=1)))
    owner.logits = torch.zeros((1, 6), dtype=torch.float16, device="cuda")
    receipt = owner.sample()
    error = RuntimeError("CUDA device failure") if fatal else ValueError("invalid output shape")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(owner.final_output, "consume", fail)
    if fatal:
        with pytest.raises(RuntimeError, match="CUDA device failure"):
            receipt.get_output()
        assert not owner.closed
        assert not any(owner.control_busy)
        done = torch.Event()
        done.record()
        owner.release(owner.binding, consumer_done=done)
    else:
        result = getattr(receipt.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
        assert result.terminal_result.failure.code == "beam_final_output_error"
        assert result.resources_released and owner.closed


@pytest.mark.parametrize("n", [1, 3])
def test_prefill_sampling_orders_producer_compute_and_host_copy(monkeypatch, n):
    """Exercise production sampling and both terminal/control D2H routes."""
    # Warm the same kernels before delaying device work so JIT compilation
    # cannot turn a missing stream dependency into an accidental host fence.
    warm = runtime(n=n)
    warm.admit(dispatch(metadata(n=n)))
    warm.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    warm.sample().get_output()
    if not warm.closed:
        done = torch.Event()
        done.record()
        warm.release(warm.binding, consumer_done=done)

    owner = runtime(n=n)
    owner.admit(dispatch(metadata(n=n)))
    logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    producer = torch.cuda.Stream()
    compute = torch.cuda.Stream()
    initialized = torch.Event()
    initialized.record()
    ready = torch.Event()
    observed_streams = []
    select = type(owner.backend).select_out_unchecked

    def record_select(backend, *args, **kwargs):
        observed_streams.append(torch.cuda.current_stream())
        select(backend, *args, **kwargs)

    monkeypatch.setattr(type(owner.backend), "select_out_unchecked", record_select)
    with torch.cuda.stream(producer):
        producer.wait_event(initialized)
        torch.cuda._sleep(20_000_000)
        logits[0, 3] = 9
        ready.record(producer)
    with torch.cuda.stream(compute):
        compute.wait_event(ready)
        owner.logits = logits
        receipt = owner.sample()
    result = getattr(receipt.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]

    assert observed_streams == [compute]
    assert owner.tokens[0, 0].item() == 3
    assert result.produced_token_count == 1
    if n == 1:
        assert result.terminal_result.output.items[0].sequences[0].token_ids == (3,)
        assert owner.final_output.final_d2h_count == 1
    else:
        assert result.terminal_result is None
        assert owner.host_control[receipt.slot, 0].tolist() == [1, 0, 0]
    if not owner.closed:
        done = torch.Event()
        done.record(compute)
        owner.release(owner.binding, consumer_done=done)


def test_head_slice_plan_gates_on_rows_and_clamps_cols() -> None:
    """Only wide selections slice, and never with a block narrower than 16384."""

    vocab, itemsize, budget = 176256, 2, 128 * 2**20

    assert head_slice_plan(4096, vocab, itemsize, 0, 1024, budget) == (False, 0)
    # A forced width is clamped into the measured-useful window and ignores the
    # row gate (manual A/B).
    assert head_slice_plan(1, vocab, itemsize, 1024, 1024, budget) == (True, 16384)
    assert head_slice_plan(1, vocab, itemsize, 100000, 1024, budget) == (True, 32768)
    # ``auto`` derives the width from the vocabulary and gates on the row count.
    assert head_slice_plan(256, vocab, itemsize, SLICE_AUTO, 1024, budget) == (False, 29696)
    assert head_slice_plan(1024, vocab, itemsize, SLICE_AUTO, 1024, budget) == (True, 29696)
    # The budget shrinks the block but never below the measured minimum, and
    # the caller addresses blocks by tile, so the width stays tile-aligned.
    tight = 120 * 2**20
    assert head_slice_plan(4096, vocab, itemsize, SLICE_AUTO, 1024, tight) == (True, 16384)
    # A vocabulary that fits one block is never sliced.
    assert head_slice_plan(64, 8192, itemsize, SLICE_AUTO, 0, budget) == (False, 0)


@pytest.mark.parametrize("rows", [1024, 2048, 4096])  # type: ignore[misc]
def test_head_slice_budget_reduction_keeps_tile_alignment(rows: int) -> None:
    """Every reduced width must address whole tiles (``start // LOCAL_TILE_SIZE``)."""

    for vocab in (16512, 33000, 176256):
        for budget_mb in (1, 16, 64, 128):
            use_slicing, cols = head_slice_plan(rows, vocab, 2, SLICE_AUTO, 1024, budget_mb * 2**20)
            if use_slicing:
                assert cols % LOCAL_TILE_SIZE == 0, (rows, vocab, budget_mb, cols)


def test_head_logits_buffers_are_allocated_lazily_per_mode() -> None:
    """Slicing must not retain the full-vocabulary buffer beside the slice."""

    rows, hidden_size, vocab, cols = 8, 16, 4096, 1024
    device = torch.device("cuda")
    layout = (torch.empty((hidden_size, vocab), dtype=torch.bfloat16, device=device), None)

    buffers = _HeadLogits(layout, device)
    assert buffers.full is None and buffers.slice is None
    full = buffers.full_buffer(rows)
    assert full is not None and tuple(full.shape) == (rows, vocab)
    assert buffers.slice is None

    buffers = _HeadLogits(layout, device)
    sliced = buffers.slice_buffer(rows, cols)
    assert sliced is not None and tuple(sliced.shape) == (rows, cols)
    assert buffers.full is None

    # Growth reallocates; a smaller row count reuses the buffer in place.
    assert buffers.slice_buffer(rows * 2, cols).shape == (rows * 2, cols)
    assert buffers.slice_buffer(rows, cols).shape == (rows, cols)
    assert buffers.full_buffer(0) is None

    assert _HeadLogits(None, device).full_buffer(rows) is None
    assert _HeadLogits(None, device).slice_buffer(rows, cols) is None


@pytest.mark.parametrize("runner_version", ["v1", "v2"])
@pytest.mark.parametrize("slice_setting", [0, "auto", 16384])
def test_transposed_head_allocation_requires_slicing(monkeypatch, runner_version, slice_setting):
    from vllm_gr.v1.worker.cuda import gpu_beam_stage_runner as stage_runner
    from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2

    # Isolate the head-allocation boundary from V2's native input storage.
    monkeypatch.setattr(
        CudaStageHooksV2,
        "initialize_inputs",
        CudaStageHooksV1.initialize_inputs,
    )
    layout = (torch.empty((8, 32768), dtype=torch.bfloat16, device="cuda"), None)
    get_layout = Mock(return_value=layout)
    monkeypatch.setattr(stage_runner, "_get_unconstrained_head_layout", get_layout)
    owner = runtime(
        unconstrained=True,
        gr_options={"unconstrained_head_layout": "transposed", "head_slice": slice_setting},
        hooks_cls=CudaStageHooksV2 if runner_version == "v2" else CudaStageHooksV1,
    )
    if slice_setting == 0:
        get_layout.assert_not_called()
        assert owner._head_layout is None
        assert owner._head_slice_request == 0
    else:
        get_layout.assert_called_once_with(owner.worker)
        assert owner._head_layout is layout
        assert owner._head_slice_request == (SLICE_AUTO if slice_setting == "auto" else 16384)


def _assert_sliced_head_non_divisible_vocab_peak_allocation():
    """Measure only allocations made by this projection in a fresh process."""
    from vllm_gr.v1.worker.cuda import gpu_beam_stage_runner as stage_runner

    rows, hidden_size, vocab_size = 2048, 16, 176256
    budget = 128 * 2**20
    use_slicing, cols = head_slice_plan(rows, vocab_size, 2, SLICE_AUTO, 1024, budget)
    assert use_slicing and vocab_size % cols
    hidden = torch.zeros((rows, hidden_size), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((hidden_size, vocab_size), dtype=hidden.dtype, device="cuda")
    ready = CUDAEvent(external=True)
    ready.record()
    owner = SimpleNamespace(
        _head_layout=(weight, ready),
        _head_buffers=_HeadLogits((weight, ready), hidden.device),
        _active_slice_cols=cols,
        temperatures=torch.ones(rows, device="cuda"),
    )

    # Measure the production projection path independently of Top-K scratch.
    def run():
        GPUBeamStageRunner._select_from_hidden_slices(owner, hidden, rows, 1, None, (), None)

    with (
        patch.object(stage_runner, "stream_slice", lambda *args, **kwargs: None),
        patch.object(stage_runner, "repair_emit", lambda *args, **kwargs: None),
    ):
        run()  # Warm up GEMM, including the tail shape, before measuring allocations.
        torch.accelerator.synchronize()
        owner._head_buffers.release_slice()
        # Reclaim transient cycles before sampling the process-level baseline;
        # none may disappear between that baseline and the peak measurement.
        gc.collect()
        torch.accelerator.synchronize()
        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()
        try:
            baseline = torch.accelerator.memory_allocated()
            torch.accelerator.reset_peak_memory_stats()
            run()
            torch.accelerator.synchronize()
            peak = torch.accelerator.max_memory_allocated() - baseline
        finally:
            if gc_enabled:
                gc.enable()
        assert rows * cols * hidden.element_size() <= peak <= budget
        assert owner._head_buffers.full is None


def test_sliced_head_non_divisible_vocab_peak_allocation():
    # CUDA allocation counters are process-wide. Run this measurement in a
    # fresh process so unrelated tests cannot free allocations after baseline.
    command = (
        "from tests.kernels.test_gpu_beam_stage_runner import "
        "_assert_sliced_head_non_divisible_vocab_peak_allocation; "
        "_assert_sliced_head_non_divisible_vocab_peak_allocation()"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_transposed_head_gemm_only_when_slicing_is_requested():
    rows, hidden_size, vocab_size = 2, 8, 4096
    transposed = torch.randn((hidden_size, vocab_size), dtype=torch.bfloat16, device="cuda")
    ready = CUDAEvent(external=True)
    ready.record()
    owner = SimpleNamespace(
        _head_layout=(transposed, ready),
        _head_buffers=_HeadLogits((transposed, ready), transposed.device),
        _head_slice_request=0,
    )
    hidden = torch.randn((rows, hidden_size), dtype=transposed.dtype, device="cuda")

    # Slicing disabled: the monolithic head stays on the model's compute_logits.
    assert GPUBeamStageRunner._try_transposed_head_logits(owner, hidden) is None

    owner._head_slice_request = 1024
    logits = GPUBeamStageRunner._try_transposed_head_logits(owner, hidden)
    assert logits is not None
    assert tuple(logits.shape) == (rows, vocab_size)


@pytest.mark.parametrize("top_k", [3, 32])
def test_sliced_head_selection_can_capture_projection_and_repair(monkeypatch, top_k):
    from vllm_gr.ops.cuda.unconstrained_topk_mcap import (
        allocate_mcap_topk_scratch,
        unconstrained_topk_mcap_out,
    )

    rows, hidden_size, vocab_size, cols = 3, 16, 8205, 2048
    generator = torch.Generator(device="cuda").manual_seed(77)
    hidden = torch.randn(
        (rows, hidden_size), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    weight = torch.randn(
        (hidden_size, vocab_size), dtype=hidden.dtype, device="cuda", generator=generator
    )
    ready = CUDAEvent(external=True)
    ready.record()
    owner = SimpleNamespace(
        _head_layout=(weight, ready),
        _head_buffers=_HeadLogits((weight, ready), hidden.device),
        _active_slice_cols=cols,
        temperatures=torch.ones(rows, device="cuda"),
    )
    scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, hidden.device)
    reference_scratch = allocate_mcap_topk_scratch(rows, vocab_size, top_k, hidden.device)

    def buffers():
        return (
            torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
            torch.empty(rows, dtype=torch.int32, device="cuda"),
        )

    result, expected = buffers(), buffers()
    mm = torch.mm
    projections = []

    def counted_mm(*args, **kwargs):
        projections.append(args[1].shape[1])
        target = kwargs["out"]
        assert target.is_contiguous()
        assert target.data_ptr() == owner._head_buffers.slice.data_ptr()
        return mm(*args, **kwargs)

    def run():
        GPUBeamStageRunner._select_from_hidden_slices(
            owner, hidden, rows, top_k, scratch, result[:3], result[3]
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch, "mm", counted_mm)
        run()
    assert projections == [cols] * (vocab_size // cols) + [vocab_size % cols]
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration in range(2):
        if iteration:
            hidden.neg_()
        graph.replay()
        # Match the sliced GEMM's shapes and output strides: the monolithic
        # GEMM can legitimately round BF16 values differently.
        logits = torch.empty((rows, vocab_size), dtype=hidden.dtype, device="cuda")
        buffer = torch.empty((rows, cols), dtype=hidden.dtype, device="cuda")
        for start in range(0, vocab_size, cols):
            width = min(cols, vocab_size - start)
            if width == cols:
                mm(hidden, weight[:, start : start + width], out=buffer)
                sliced = buffer
            else:
                sliced = mm(hidden, weight[:, start : start + width])
            logits[:, start : start + width].copy_(sliced)
        unconstrained_topk_mcap_out(logits, owner.temperatures, top_k, reference_scratch, *expected)
        # Streaming and full scans reduce the FP32 normalizer in different
        # orders. Selection stays exact; logprobs may differ in their last bits.
        torch.testing.assert_close(result[1], expected[1], rtol=1e-6, atol=4e-6)
        for index in (0, 2, 3):
            got, want = result[index], expected[index]
            assert torch.equal(got, want)
