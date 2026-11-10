# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-request reuse must preserve reset values, ownership and live admission."""

from dataclasses import replace

import msgspec
import pytest
import torch

from tests.kernels.test_beam_search_runner_cuda import _runtime as request_runtime
from tests.kernels.test_gpu_beam_stage_runner import (
    admission_worker,
    dispatch,
    metadata,
    runtime,
)
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspace
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.engine.beam_search_v1 import GRDeviceStateRef
from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner

pytestmark = pytest.mark.gpu
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


def release(owner, m):
    owner.previous = m
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)


def renamed(m, name):
    return replace(
        m,
        session_id=name,
        request_ids=tuple(
            name if index == 0 else f"{name}:{index}" for index in range(len(m.request_ids))
        ),
        output_device_state=GRDeviceStateRef(name, 0),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@torch.inference_mode()
def test_rebind_preserves_views_and_resets_every_state_field(dtype, steps, batch_size):
    worker = admission_worker(dtype=dtype, max_rows=batch_size * 4)
    m = metadata(n=steps)
    m = renamed(
        replace(
            m,
            request_ids=(m.session_id,) * batch_size,
            num_prompt_tokens=batch_size * 7,
            num_prompt_tokens_by_request=(7,) * batch_size,
        ),
        m.session_id,
    )
    scheduled = dispatch(m)
    scheduled.num_scheduled_tokens = dict(zip(m.request_ids, m.num_prompt_tokens_by_request))
    owner = get_stage_runner(worker, scheduled)
    beam, state = owner.beam, owner.beam.state
    runners = beam.runners
    views = [dict(runner._state) for runner in runners]
    tensors = {k: v for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
    initial = {k: v.clone() for k, v in tensors.items()}
    decision = state.decision
    for iteration in range(4):
        for name, tensor in tensors.items():
            if name not in {
                "stop_token_ids",
                "num_stop_tokens",
                "ignore_stop_tokens",
                "max_new_tokens_tensor",
            }:
                tensor[:batch_size].fill_(17)
        decision.group_token_num[:batch_size].fill_(91)
        release(owner, m)
        m = renamed(m, f"reused-{iteration}")
        scheduled = dispatch(m)
        scheduled.num_scheduled_tokens = dict(zip(m.request_ids, m.num_prompt_tokens_by_request))
        assert get_stage_runner(worker, scheduled) is owner
        assert owner.beam is beam
        assert beam.runners is runners
        for index, (runner, row_views) in enumerate(zip(runners, views)):
            assert all(runner._state[name] is view for name, view in row_views.items())
            assert runner.session is worker._beam_sessions[m.request_ids[index]]
            assert runner.session.is_active
            assert runner.session.slot == runner.state_index == index
        assert beam.session is runners[0].session
        assert owner.binding.session_id == m.session_id
        assert not beam.closed
        assert list(worker._beam_sessions) == list(m.request_ids)
        for name, tensor in tensors.items():
            torch.testing.assert_close(tensor, initial[name])
        assert not decision.group_token_num.any().item()
    release(owner, m)


@torch.inference_mode()
def test_fused_reset_matches_full_reset_for_nonzero_row_and_preserves_other_rows():
    beam = request_runtime(steps=3, stops=(9,), batch=3, state_index=1)
    state = beam.state
    tensors = {k: v for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
    for value in tensors.values():
        value.fill_(7)
    state.decision.group_token_num.fill_(11)
    # The reference remains available on CPU and for standalone/legacy callers.
    state.reset(beam.state_index)
    state.decision.reset(beam.state_index)
    expected = {k: v.clone() for k, v in tensors.items()}
    expected_group = state.decision.group_token_num.clone()
    configuration = {
        "stop_token_ids",
        "num_stop_tokens",
        "ignore_stop_tokens",
        "max_new_tokens_tensor",
    }
    for name, value in tensors.items():
        if name not in configuration:
            value[beam.state_index].fill_(33)
    state.decision.group_token_num[beam.state_index].fill_(44)
    beam.end_session()
    for _ in range(2):
        beam.reset_for_session()
        for name, value in tensors.items():
            torch.testing.assert_close(value, expected[name])
        torch.testing.assert_close(state.decision.group_token_num, expected_group)


@torch.inference_mode()
def test_only_state_configuration_changes_rebuild_state():
    worker = admission_worker()
    m = metadata()
    owner = get_stage_runner(worker, dispatch(m))
    original = owner.beam
    release(owner, m)
    params = msgspec.structs.replace(m.beam_params, temperature=0.5, length_penalty=-1.0)
    m = replace(
        renamed(m, "changed-output-params"),
        beam_params=params,
        num_prompt_tokens=17,
        num_prompt_tokens_by_request=(17,),
    )
    get_stage_runner(worker, dispatch(m))
    assert owner.beam is original
    assert owner.beam.sessions[0]._state.prefix_len == 17
    release(owner, m)
    params = msgspec.structs.replace(params, stop_token_ids=(2,), ignore_stop_tokens=True)
    m = replace(renamed(m, "changed-state-params"), beam_params=params)
    get_stage_runner(worker, dispatch(m))
    assert owner.beam is not original
    assert owner.beam.state.stop_token_ids.tolist() == [[2]] * owner.beam.state.max_batch_size
    assert owner.beam.state.ignore_stop_tokens.all().item()
    release(owner, m)


@torch.inference_mode()
def test_rebind_preserves_request_major_kv_slot():
    worker = admission_worker()
    worker._beam_context = BeamAttentionContext(2, 4, 1, 3, 32, 2, worker.device, torch.float16)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=2,
        beam_width=4,
        top_k=4,
        max_decode_steps=3,
        device=worker.device,
        logits_dtype=torch.float16,
    )
    worker._beam_postprocess_workspaces.require.return_value = workspace
    m = metadata()
    owner = get_stage_runner(worker, dispatch(m))
    beam = owner.beam
    assert owner.binding.kv_slot == 0
    release(owner, m)
    get_stage_runner(worker, dispatch(m))
    assert owner.beam is beam
    assert owner.binding.kv_slot == 0
    first = worker._beam_context.pools[0].unshared_key
    assert (
        beam.runners[0]._kv_args[4]
        == owner.binding.kv_slot * first.stride(0) * first.element_size()
    )
    release(owner, m)


@torch.inference_mode()
def test_request_local_rebind_updates_changed_kv_slot():
    owner = runtime()
    runner = owner.beam.runners[0]
    context = runner.session.context
    release(owner, metadata())
    context.registry.allocate("occupied-row", runner.state.beam_width, 7)
    slot = context.registry.allocate("rebound", runner.state.beam_width, 7)
    assert slot == 1
    session = BeamSearchSession("rebound", slot, context)
    session.begin_session(runner.state.beam_width, 7, decision_index=runner.state_index)
    runner.rebind_session(session)
    first = context.pools[0].unshared_key
    assert runner._kv_args[4] == slot * first.stride(0) * first.element_size()
    runner.end_session()
    context.registry.release("occupied-row")


@pytest.mark.parametrize("captured", [False, True])
@torch.inference_mode()
def test_changed_resource_address_rebuilds_views_or_rejects_captured_binding(captured):
    worker = admission_worker()
    m = metadata()
    owner = get_stage_runner(worker, dispatch(m))
    old = owner.beam
    if captured:
        owner.graph = object()
    release(owner, m)
    old.state.sequence.fill_(321)
    old.workspace.selected_token_ids = old.workspace.selected_token_ids.clone()
    if captured:
        with pytest.raises(ValueError, match="resource addresses or layouts changed"):
            get_stage_runner(worker, dispatch(m))
        assert (old.state.sequence == 321).all().item()
        assert not worker._beam_sessions
        assert owner.closed
    else:
        owner = get_stage_runner(worker, dispatch(m))
        assert owner.beam is not old
        assert (
            owner.beam.runners[0]._selected_tokens.data_ptr()
            == old.workspace.selected_token_ids.data_ptr()
        )
        release(owner, m)


@torch.inference_mode()
def test_invalid_pool_layout_is_rejected_before_reuse_reset():
    worker = admission_worker()
    m = metadata()
    owner = get_stage_runner(worker, dispatch(m))
    release(owner, m)
    owner.beam.state.sequence.fill_(123)
    pool = worker._beam_context.pools[1]
    pool.unshared_value = pool.unshared_value[..., :16]
    with pytest.raises(ValueError, match="identical layouts"):
        get_stage_runner(worker, dispatch(m))
    assert (owner.beam.state.sequence == 123).all().item()
    assert owner.closed
    assert not worker._beam_sessions


@torch.inference_mode()
def test_replaced_decision_storage_rebuilds_state_aliases_before_reset():
    worker = admission_worker()
    m = metadata()
    owner = get_stage_runner(worker, dispatch(m))
    old_state = owner.beam.state
    release(owner, m)
    decision = worker._beam_context.decision
    decision.sequence = torch.full_like(decision.sequence, 45)
    decision.beam_scores = torch.full_like(decision.beam_scores, 67)
    owner = get_stage_runner(worker, dispatch(m))
    state = owner.beam.state
    assert state is not old_state
    assert state.sequence.data_ptr() == decision.sequence.data_ptr()
    assert state.beam_scores.data_ptr() == decision.beam_scores.data_ptr()
    assert not state.sequence.any().item()
    assert not state.beam_scores[0].any().item()
    assert torch.isneginf(state.beam_scores[1:]).all().item()
    release(owner, m)
