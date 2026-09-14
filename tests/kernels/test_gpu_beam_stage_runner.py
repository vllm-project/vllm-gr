# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA adapter tests: independent resources, dispatch snapshots and terminal KV."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import msgspec
import pytest
import torch

from tests.constraints.test_table import make_table
from tests.test_gr_async_scheduler import _batch_request
from vllm_gr.constraints.loader import _upload_cuda_backend
from vllm_gr.ops.cuda import beam_stage as ops
from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl
from vllm_gr.v1.attention.backends.beam_graph_kernels import (
    cache_unshared_kv_graph,
    copy_suffix_kv_graph,
)
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspace
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRStageKind,
    GRStageMetadata,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
)
from vllm_gr.v1.worker.beam_search_runner import BeamSearchRunner
from vllm_gr.v1.worker.gpu_beam_stage_runner import (
    BeamRequestBinding,
    GPUBeamStageRunner,
    get_stage_runner,
    validate_dispatch,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


def metadata(stage=0, dispatch=None, output=True, n=3):
    request = _batch_request(n)
    return GRStageMetadata(
        session_id=request.batch_id,
        dispatch_id=stage if dispatch is None else dispatch,
        stage_index=stage,
        kind=GRStageKind.PREFILL if stage == 0 else GRStageKind.DECODE,
        decode_step=stage - 1 if stage else None,
        is_last_stage=output and stage == n - 1,
        input_device_state=GRDeviceStateRef(request.batch_id, stage - 1) if stage else None,
        output_device_state=GRDeviceStateRef(request.batch_id, stage),
        beam_params=msgspec.structs.replace(request.beam_params, stop_token_ids=()),
        output_options=request.output_options,
        num_prompt_tokens=7,
        produces_output=output,
    )


def dispatch(m):
    result = SimpleNamespace(num_scheduled_tokens={m.session_id: 7 if m.stage_index == 0 else 1})
    setattr(result, GR_STAGE_OUTPUT_ATTR, {m.session_id: m})
    return result


def runtime(dtype=torch.float16, n=3):
    context = BeamAttentionContext(3, 6, 1, 5, 8, 2, torch.device("cuda"), dtype)
    state = BeamSearchState(context.decision, beam_width=4, max_new_tokens=n)
    # State row 2, workspace row 1 and KV slot 0 are intentionally distinct.
    slot = context.registry.allocate("session-0", 4, 7)
    session = BeamSearchSession("session-0", slot, context)
    session.begin_session(4, 7, decision_index=2)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=3, beam_width=4, top_k=4, max_decode_steps=5, device="cuda", logits_dtype=dtype
    )
    beam = BeamSearchRunner(session, state, workspace, state_index=2, workspace_index=1)
    backend, _ = _upload_cuda_backend(
        make_table(), device=torch.device("cuda"), max_top_k=4, max_rows=4
    )
    backend._mark_ready()
    worker = SimpleNamespace(_beam_sessions={"session-0": session}, execute_model_state=None)
    owner = GPUBeamStageRunner(
        worker, beam, BeamRequestBinding("session-0", 0, "session-0", 2, 0, 1), backend
    )
    owner.metadata = SimpleNamespace(
        suffix_seqlens_kv_device=torch.empty(4, device="cuda", dtype=torch.int32),
        suffix_cu_seqlens_k_device=torch.empty(5, device="cuda", dtype=torch.int32),
    )
    owner.temperatures.fill_(1)
    owner.prefixes.fill_(-1)
    return owner


def admission_worker(sliding_windows=(None, None), dtype=torch.float16):
    device = torch.device("cuda", torch.accelerator.current_device_index())
    backend, _ = _upload_cuda_backend(make_table(), device=device, max_top_k=4, max_rows=4)
    backend._mark_ready()
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1, beam_width=4, top_k=4, max_decode_steps=3, device=device, logits_dtype=dtype
    )
    layers = {
        f"layer.{i}": SimpleNamespace(
            impl=BeamAttentionImpl(
                num_heads=1,
                head_size=32,
                scale=32**-0.5,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=window,
                kv_cache_dtype="auto",
            )
        )
        for i, window in enumerate(sliding_windows)
    }
    return SimpleNamespace(
        device=device,
        use_async_scheduling=True,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        model_config=SimpleNamespace(
            dtype=dtype,
            is_encoder_decoder=False,
            is_hybrid=False,
            get_vocab_size=lambda: backend.info.table_metadata.vocab_size,
            get_head_size=lambda: 32,
            get_num_kv_heads=lambda _: 1,
            get_num_attention_heads=lambda _: 1,
            get_num_layers=lambda _: len(layers),
        ),
        supports_mm_inputs=False,
        vllm_config=SimpleNamespace(),
        cache_config=SimpleNamespace(cache_dtype="auto", sliding_window=None),
        _vllm_gr_config=SimpleNamespace(beam_max_width=4, beam_max_decode_steps=3),
        constraint_backend=backend,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=list(layers))]
        ),
        compilation_config=SimpleNamespace(static_forward_context=layers),
        _beam_postprocess_workspaces=SimpleNamespace(require=Mock(return_value=workspace)),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_admission_accepts_full_attention(dtype):
    worker = admission_worker(dtype=dtype)
    owner = get_stage_runner(worker, dispatch(metadata()))
    assert owner.beam.session.is_active
    assert worker._beam_context.registry.num_active_sessions == 1
    assert worker._beam_context._gr_execution_owner is owner
    assert owner.beam.workspace.execution_owner is owner
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)


@pytest.mark.parametrize("windows", [(4, 4), (None, 4), (4, None), (None, 1)])
def test_admission_rejects_sliding_window_before_resource_creation(windows, monkeypatch):
    from vllm_gr.v1.worker import model_runner_common

    worker = admission_worker(windows)
    create_context = Mock(wraps=model_runner_common._create_beam_context)
    monkeypatch.setattr(model_runner_common, "_create_beam_context", create_context)
    with pytest.raises(ValueError, match="does not support sliding-window attention") as exc:
        get_stage_runner(worker, dispatch(metadata()))
    first_window_layer = next(i for i, window in enumerate(windows) if window is not None)
    assert f"layer.{first_window_layer}" in str(exc.value)
    create_context.assert_not_called()
    worker._beam_postprocess_workspaces.require.assert_not_called()
    assert getattr(worker, "_beam_context", None) is None
    assert getattr(worker, "_gr_gpu_stage_runner", None) is None
    assert not getattr(worker, "_beam_sessions", {})


def test_admission_rejects_sliding_window_before_reuse_reset(monkeypatch):
    worker = admission_worker()
    owner = get_stage_runner(worker, dispatch(metadata()))
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
        get_stage_runner(worker, dispatch(metadata()))
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
    protected = {k: v[:2].clone() for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
    ws.selected_token_ids[0].fill_(987)
    ws.prefix_token_ids[0].fill_(789)
    ws.prefix_token_ids[2].fill_(789)
    for pool in owner.beam.session.pools:
        pool.unshared_key[1:].fill_(123)
        pool.unshared_value[1:].fill_(456)
    receipts = []
    for stage in range(n):
        execution = owner.admit(dispatch(metadata(stage, n=n)))
        if stage:
            owner.prepare_decode(execution.entries[0])
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
        if result.terminal_result is not None:
            assert result.terminal_result.output is not None
    assert state.num_generated_tokens.tolist() == [0, 0, n]
    assert state.num_decode_steps.tolist() == [0, 0, n - 1]
    for name, value in protected.items():
        torch.testing.assert_close(getattr(state, name)[:2], value)
    assert ws.selected_token_ids[0].tolist() == [987] * 4
    assert state.num_completed[2] > 0
    assert owner.final_output.final_d2h_count == 1
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)
    owner.release(owner.binding, consumer_done=done)
    assert not owner.beam.session.context.registry.has_session("session-0")
    state.reset(2)
    for name, value in protected.items():
        torch.testing.assert_close(getattr(state, name)[:2], value)
    assert ws.prefix_token_ids[0].eq(789).all()
    assert ws.prefix_token_ids[2].eq(789).all()
    for pool in owner.beam.session.pools:
        assert pool.unshared_key[1:].eq(123).all()
        assert pool.unshared_value[1:].eq(456).all()


@pytest.mark.parametrize("graph", [False, True])
def test_terminal_successor_preserves_all_layer_kv(graph):
    owner = runtime()
    state = owner.beam.state
    current = torch.ones((4, 1, 8), device="cuda", dtype=torch.float16)
    entry = owner.admit(dispatch(metadata())).entries[0]
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    owner.sample().get_output()
    entry = replace(entry, metadata=metadata(1), row_count=4)
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
    state.is_finished[2] = True
    before = [(p.unshared_key.clone(), p.unshared_value.clone()) for p in owner.beam.session.pools]
    replay()
    for pool, (key, value) in zip(owner.beam.session.pools, before):
        torch.testing.assert_close(pool.unshared_key, key)
        torch.testing.assert_close(pool.unshared_value, value)
    assert owner.beam.session.suffix_k_buf[:4].count_nonzero() == 0
    assert owner.execution_mask.count_nonzero() == 0
    assert owner.positions.tolist() == [0] * 4


def test_chunks_identity_and_admission_before_writes():
    owner = runtime()
    m = metadata(output=False)
    owner.admit(dispatch(m))
    result = owner.sample().get_output()
    assert getattr(result, GR_WORKER_RESULT_ATTR)[m.session_id].produced_token_count == 0
    assert owner.beam.state.num_generated_tokens.sum() == 0
    continuation = metadata(dispatch=1)
    owner.admit(dispatch(continuation))
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    owner.sample().get_output()
    before = owner.beam.state.sequence.clone()
    with pytest.raises(ValueError, match="dispatch order"):
        owner.admit(dispatch(metadata(stage=1, dispatch=99)))
    with pytest.raises(ValueError, match="second active"):
        other = replace(
            metadata(stage=1, dispatch=2),
            session_id="other",
            output_device_state=GRDeviceStateRef("other", 1),
            input_device_state=GRDeviceStateRef("other", 0),
        )
        owner.admit(dispatch(other))
    mixed = dispatch(metadata(1, dispatch=2))
    mixed.num_scheduled_tokens["ordinary"] = 1
    with pytest.raises(ValueError, match="mix"):
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
    request = _batch_request()
    request.items[0].request.prompt_token_ids = [1] * m.num_prompt_tokens
    request = msgspec.structs.replace(request, beam_params=m.beam_params)
    if fault == "protocol":
        request = msgspec.structs.replace(request, protocol_version=2)
    elif fault == "item":
        request = msgspec.structs.replace(
            request, items=(msgspec.structs.replace(request.items[0], item_index=1),)
        )
    else:
        request.items[0].request.prompt_token_ids = [1]
    before = owner.beam.state.sequence.clone()
    with pytest.raises(ValueError, match="session_init disagrees"):
        owner.admit(dispatch(replace(m, session_init=request)))
    assert owner.pending is None and owner.previous is None and not any(owner.control_busy)
    torch.testing.assert_close(owner.beam.state.sequence, before)


def test_device_prepare_nonzero_model_offset_and_strided_prefix():
    owner = runtime()
    s = owner.beam.state
    s.is_initialized[2] = True
    s.num_generated_tokens[2] = 2
    s.num_decode_steps[2] = 1
    s.active_mask[2] = torch.tensor([True, False, True, False], device="cuda")
    s.input_ids[2] = torch.tensor([3, 4, 5, 6], device="cuda")
    s.sequence[2, :, :2] = torch.tensor([[1, 2]] * 4, device="cuda")
    prefix = torch.full((3, 4, 7), 99, device="cuda", dtype=torch.int32)[:, :, :3]
    tokens = torch.full((9,), 77, device="cuda", dtype=torch.int32)
    positions = tokens.clone()
    lengths = owner.metadata.suffix_seqlens_kv_device
    offsets = owner.metadata.suffix_cu_seqlens_k_device
    ops.prepare_decode[(1,)](
        s.input_ids,
        s.sequence,
        s.active_mask,
        s.num_generated_tokens,
        s.num_decode_steps,
        s.is_initialized,
        s.is_finished,
        s.error_code,
        tokens,
        positions,
        owner.execution_mask,
        owner.beam.session.context.decode_step_buf,
        prefix,
        lengths,
        offsets,
        STATE_ROW=2,
        WORKSPACE_ROW=1,
        ROW_OFFSET=3,
        WIDTH=4,
        PROMPT=7,
        TOKEN_BATCH_STRIDE=s.input_ids.stride(0),
        SEQ_BATCH_STRIDE=s.sequence.stride(0),
        SEQ_BEAM_STRIDE=s.sequence.stride(1),
        PREFIX_BATCH_STRIDE=prefix.stride(0),
        PREFIX_BEAM_STRIDE=prefix.stride(1),
        MAX_STEPS=2,
        BLOCK=8,
    )
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
    with pytest.raises(ValueError, match="B=1"):
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
    owner.prepare_decode(owner.pending[0].entries[0])
    protected = {
        name: value.clone()
        for name, value in vars(state).items()
        if isinstance(value, torch.Tensor) and name != "error_code"
    }
    result = owner.sample().get_output()
    receipt = getattr(result, GR_WORKER_RESULT_ATTR)["session-0"]
    assert receipt.finished and receipt.produced_token_count == 0
    assert receipt.terminal_result.failure.code == "beam_device_error"
    for name, value in protected.items():
        torch.testing.assert_close(getattr(state, name), value)
    execution = owner.admit(dispatch(metadata(2)))
    owner.prepare_decode(execution.entries[0])
    owner.logits = torch.zeros((4, 6), device="cuda", dtype=torch.float16)
    successor = owner.sample().get_output()
    assert getattr(successor, GR_WORKER_RESULT_ATTR)["session-0"].produced_token_count == 0
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
    assert completion["sequence"].data_ptr() == owner.beam.state.sequence[2].data_ptr()
    with pytest.raises(ValueError, match="binding mismatch"):
        owner.release(replace(owner.binding, state_row=1), consumer_done=done)
    owner.cancel(owner.binding)
    owner.release(owner.binding, consumer_done=done)
    with pytest.raises(RuntimeError, match="live matching"):
        owner.completion_view(owner.binding)


def test_finished_request_defers_release_until_async_output_is_consumed():
    with torch.inference_mode():
        owner = runtime(n=1)
        owner.admit(dispatch(metadata(n=1)))
        owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
        receipt = owner.sample()

    assert not owner.final_output.device.is_inference()
    assert not owner.final_output.host.is_inference()

    owner.request_release(owner.binding)
    assert not owner.closed
    result = getattr(receipt.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]

    assert owner.closed
    assert result.terminal_result is not None
    assert result.terminal_result.output is not None
    assert not owner.beam.session.context.registry.has_session("session-0")
    owner.request_release(owner.binding)


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


@pytest.mark.parametrize("max_model_len", [31, 40961])
def test_graph_geometry_change_requires_drained_binding_refresh(max_model_len):
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
        max_model_len=max_model_len,
        get_num_attention_heads=lambda _: 1,
        get_num_kv_heads=lambda _: 1,
    )
    owner._bind_attention(metadata())
    old = cast(Any, owner.metadata)
    assert old.prefix_max_seq_len == (max_model_len + 15) // 16 * 16
    assert old.prefix_seqlens_kv_device.tolist() == [7]
    owner.graph = object()
    changed = replace(metadata(), num_prompt_tokens=8)
    with pytest.raises(RuntimeError, match="drain and release"):
        owner._bind_attention(changed)
    assert owner.metadata is old
    # This flag is published only after release and re-admission in production.
    owner.attention_needs_refresh = True
    owner._bind_attention(changed)
    assert owner.graph is None and owner.metadata is not old
    assert owner.metadata.prefix_seqlens_kv_device.tolist() == [8]
