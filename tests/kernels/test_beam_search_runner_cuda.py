# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-step acceptance tests using real CUDA ops and an independent CPU oracle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from numpy.typing import NDArray

from vllm_gr.ops.cuda.beam_kv_reorder import (
    _select_kv_gather_kernel,
    _select_kv_scatter_kernel,
)
from vllm_gr.ops.cuda.beam_selection import (
    _hierarchical_topk_out,
    allocate_beam_selection_scratch,
)
from vllm_gr.ops.cuda.beam_state import (
    ERROR_ALREADY_INITIALIZED,
    ERROR_CAPACITY,
    ERROR_INVALID_CANDIDATE,
    ERROR_NO_CANDIDATES,
    ERROR_NOT_INITIALIZED,
    FINISH_LENGTH,
    FINISH_STOP,
    _pack_final_beams_kernel,
    _rank_final_beams_kernel,
    _validate_step_kernel,
)
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspace
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState
from vllm_gr.v1.beam.cpu_reference import (
    normalize_constraint_candidates,
    select_initial_beams,
    select_next_beams,
)
from vllm_gr.v1.worker.beam_final_output import _PayloadLayout
from vllm_gr.v1.worker.beam_search_runner import BeamSearchRunner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
# Test fixtures and Triton DSL functions intentionally use untyped values.
# mypy: disable-error-code="misc,no-untyped-def,no-untyped-call"


def _runtime(
    width=4,
    steps=7,
    *,
    dtype=torch.float16,
    stops=(),
    ignore=False,
    batch=1,
    state_index=0,
    physical_steps=None,
    candidate_width=None,
):
    context = BeamAttentionContext(
        batch, width, 1, physical_steps or steps, 8, 2, torch.device("cuda"), dtype
    )
    state = BeamSearchState(
        context.decision,
        beam_width=width,
        max_new_tokens=steps,
        stop_token_ids=stops,
        ignore_stop_tokens=ignore,
        candidate_width=candidate_width,
    )
    slot = context.registry.allocate("s", width, 0)
    session = BeamSearchSession("s", slot, context)
    session.begin_session(width, 0, decision_index=state_index)
    state.reset(state_index)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=width,
        top_k=width if candidate_width is None else candidate_width,
        max_decode_steps=physical_steps or steps,
        device="cuda",
        logits_dtype=dtype,
    )
    return BeamSearchRunner(session, state, workspace, state_index=state_index)


def _gpu(tokens, scores, mask=None):
    tokens = np.asarray(tokens, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    mask = np.ones(tokens.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return tuple(torch.tensor(x, device="cuda") for x in (tokens, scores, mask))


def _cpu_candidates(tokens, scores, mask, active):
    # CPU reference accepts only active physical rows. No GPU output is read.
    return (
        normalize_constraint_candidates(
            item_index=0,
            parent_beam_indices=np.arange(active, dtype=np.int64),
            token_ids=np.asarray(tokens, dtype=np.int64)[:active],
            logprobs=np.asarray(scores, dtype=np.float32)[:active],
            valid_mask=np.asarray(mask, dtype=bool)[:active],
        ),
    )


def _completed(state, index=0):
    count = int(state.num_completed[index].cpu())
    names = ("token_ids", "scores", "parent_ids", "lengths", "finish_reasons", "tie_break_indices")
    columns = [getattr(state, "completed_" + name)[index, :count].cpu().tolist() for name in names]
    history_tokens = state.history_token_ids[index].cpu().tolist()
    history_parents = state.history_parent_ids[index].cpu().tolist()
    result = []
    for token, score, parent, length, reason, tie in zip(*columns):
        tokens, parents = [token], [parent]
        row = parent
        for depth in range(length - 2, -1, -1):
            tokens.append(history_tokens[depth][row])
            parent = history_parents[depth][row]
            parents.append(parent)
            row = parent
        result.append((tuple(reversed(tokens)), score, tuple(reversed(parents)), reason, tie))
    return result


@triton.jit
def _write_test_kv(
    key,
    value,
    step_ptr,
    active,
    finished,
    WIDTH: tl.constexpr,
    STEPS: tl.constexpr,
    LAYER: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 8)
    step = tl.load(step_ptr)
    live = tl.load(active + row) & ~tl.load(finished) & (step < STEPS)
    offset = row * STEPS * 8 + step * 8 + d
    number = (LAYER * 100 + step * 20 + row * 8 + d).to(tl.float32)
    tl.store(key + offset, number, mask=live)
    tl.store(value + offset, -number, mask=live)


def _write_kv(runner):
    s = runner._state
    for layer, pool in enumerate(runner.session.pools):
        _write_test_kv[(runner.state.beam_width,)](
            pool.unshared_key[runner.session.slot],
            pool.unshared_value[runner.session.slot],
            s["num_decode_steps"],
            s["active_mask"],
            s["is_finished"],
            WIDTH=runner.state.beam_width,
            STEPS=pool.max_decode_steps,
            LAYER=layer,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_continuous_decode_without_intermediate_readback(dtype):
    runner = _runtime(dtype=dtype, stops=(99,), physical_steps=9)
    width = 4
    rng = np.random.default_rng(812)
    initial = ([[10, 99, 12, 13]], [[-0.4, -0.1, -0.2, -0.3]], [[True] * width])
    data = []
    for step in range(4):
        tokens: NDArray[np.int64] = np.arange(16).reshape(4, 4) + 100 + 20 * step
        tokens[0, 1] = 99
        scores = rng.integers(-10, 0, size=(4, 4)).astype(np.float32) / 8
        mask: NDArray[np.bool_] = np.ones((4, 4), dtype=bool)
        if step == 0:
            mask[:, 2:] = False
        data.append((tokens, scores, mask))
    # Inputs are allocated before any execution, and the CPU oracle uses only
    # its own state. All GPU results are compared after the entire chain.
    device_inputs = [_gpu(*entry) for entry in (initial, *data)]
    reference = select_initial_beams(
        _cpu_candidates(*initial, 1), beam_width=4, stop_token_ids=(99,)
    )
    expected_kv: list[NDArray[np.float32]] = [
        np.zeros((4, 9, 8), dtype=np.float32) for _ in range(2)
    ]
    runner.initialize_from_prefill(*device_inputs[0])
    for step, (host_input, device_input) in enumerate(zip(data, device_inputs[1:])):
        active = reference.items[0].num_active_beams
        for layer in range(2):
            expected_kv[layer][:active, step] = (
                layer * 100 + step * 20 + np.arange(active)[:, None] * 8 + np.arange(8)
            )
        reference = select_next_beams(
            reference, _cpu_candidates(*host_input, active), stop_token_ids=(99,)
        )
        parents = [
            beam.parent_beam_indices[-1] for beam in reference.items[0].beams if beam.is_active
        ]
        for layer in range(2):
            expected_kv[layer][: len(parents), : step + 1] = expected_kv[layer][
                parents, : step + 1
            ].copy()
        _write_kv(runner)
        runner.advance(*device_input)
    state = runner.state
    assert state.error_code.cpu().item() == 0
    assert state.num_generated_tokens.cpu().item() == 5
    assert state.num_decode_steps.cpu().item() == 4
    beams = reference.items[0].beams
    for row, beam in enumerate(beams):
        assert state.active_mask[0, row].cpu().item() == beam.is_active
        if beam.is_active:
            assert tuple(state.sequence[0, row, :5].cpu().tolist()) == beam.token_ids
            assert state.beam_scores[0, row].cpu().item() == beam.cumulative_logprob
    completed = _completed(state)
    assert len(completed) == len(reference.items[0].completed_beams)
    for actual, expected in zip(completed, reference.items[0].completed_beams):
        assert actual == (
            expected.token_ids,
            expected.cumulative_logprob,
            expected.parent_beam_indices,
            FINISH_STOP,
            expected.tie_break_index,
        )
    for layer, pool in enumerate(runner.session.pools):
        np.testing.assert_array_equal(
            pool.unshared_key[0, :, 0, :4].cpu().float(), expected_kv[layer][:, :4]
        )
        np.testing.assert_array_equal(
            pool.unshared_value[0, :, 0, :4].cpu().float(), -expected_kv[layer][:, :4]
        )


@pytest.mark.parametrize("ignore", [False, True])
def test_fixed_width_masks_can_shrink_grow_and_finish(ignore):
    runner = _runtime(stops=(99,), ignore=ignore, steps=4)
    state = runner.state
    addresses = [t.data_ptr() for t in state.owned_tensors()]
    runner.initialize_from_prefill(
        *_gpu([[10, 99, -1, -1]], [[0, -1, float("nan"), 0]], [[True, True, False, False]])
    )
    assert state.num_active_beams.cpu().item() == (2 if ignore else 1)
    tokens = [[20, 21, 22, 23]] * 4
    runner.advance(*_gpu(tokens, [[-0.1, -0.2, -0.3, -0.4]] * 4))
    assert state.active_mask.shape == (1, 4)
    assert state.active_mask.cpu().all()
    runner.advance(*_gpu([[99] * 4] * 4, [[-1] * 4] * 4))
    if ignore:
        assert not state.is_finished.cpu().item()
        runner.advance(*_gpu(tokens, [[-1] * 4] * 4))
    assert state.is_finished.cpu().item()
    assert not state.active_mask.cpu().any()
    assert state.error_code.cpu().item() == 0
    completed = _completed(state)
    assert len(completed) == (16 if ignore else 17)
    assert all(c[3] == (FINISH_LENGTH if ignore else FINISH_STOP) for c in completed)
    snapshots = [t.clone() for t in (*state.owned_tensors(), state.sequence, state.beam_scores)]
    kv = [p.unshared_key.clone() for p in runner.session.pools]
    runner.advance(*_gpu([[-1] * 4] * 4, [[float("nan")] * 4] * 4))
    for actual, expected in zip(
        (*state.owned_tensors(), state.sequence, state.beam_scores), snapshots
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for pool, expected in zip(runner.session.pools, kv):
        torch.testing.assert_close(pool.unshared_key, expected)
    assert [t.data_ptr() for t in state.owned_tensors()] == addresses


def test_single_token_and_initialization_errors():
    runner = _runtime(steps=1, stops=(99,))
    initial = _gpu([[10, 99, 12, 13]], [[-0.2, -0.1, -0.3, -0.4]])
    runner.initialize_from_prefill(*initial)
    assert runner.state.is_finished.cpu().item()
    assert runner.state.num_generated_tokens.cpu().item() == 1
    assert runner.state.num_decode_steps.cpu().item() == 0
    assert [c[3] for c in _completed(runner.state)] == [
        FINISH_LENGTH,
        FINISH_STOP,
        FINISH_LENGTH,
        FINISH_LENGTH,
    ]
    snapshot = runner.state.completed_scores.clone()
    runner.initialize_from_prefill(*initial)
    assert runner.state.error_code.cpu().item() == ERROR_ALREADY_INITIALIZED
    torch.testing.assert_close(snapshot, runner.state.completed_scores)
    runner.reset()
    runner.advance(*_gpu([[1] * 4] * 4, [[0] * 4] * 4))
    assert runner.state.error_code.cpu().item() == ERROR_NOT_INITIALIZED


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_finished_uninitialized_row_ignores_prefill(dtype, graph, invalid):
    runner = _runtime(steps=3, dtype=dtype, batch=2, state_index=1)
    state = runner.state
    valid = _gpu([[10, 11, 12, 13]], [[0, -1, -2, -3]])
    candidates = _gpu([[10, 11, 12, 13]], [[float("nan")] * 4]) if invalid else valid
    runner.initialize_from_prefill(*candidates)
    runner.reset()

    def initialize():
        runner.initialize_from_prefill(*candidates)

    replay = initialize
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            initialize()
        runner.reset()
        replay = captured.replay
    state.is_finished[1] = True
    snapshot = {
        name: value.clone()
        for name, value in vars(state).items()
        if isinstance(value, torch.Tensor)
    }
    replay()
    for name, expected in snapshot.items():
        torch.testing.assert_close(getattr(state, name), expected, rtol=0, atol=0)
    runner.reset()
    runner.initialize_from_prefill(*valid)
    assert state.num_generated_tokens[1].cpu().item() == 1
    assert state.error_code[1].cpu().item() == 0
    assert state.is_initialized[1].cpu().item()


def test_invalid_values_do_not_partially_commit():
    runner = _runtime(stops=(99,))
    runner.initialize_from_prefill(*_gpu([[1, 2, 3, 4]], [[0] * 4]))
    state = runner.state
    before = [
        state.sequence.clone(),
        state.beam_scores.clone(),
        state.num_generated_tokens.clone(),
        state.completed_token_ids.clone(),
        state.num_completed.clone(),
    ]
    runner.advance(*_gpu([[99, 2, 3, 4]] * 4, [[0, 0, float("nan"), 0]] * 4))
    assert state.error_code.cpu().item() == ERROR_INVALID_CANDIDATE
    for tensor, expected in zip(
        (
            state.sequence,
            state.beam_scores,
            state.num_generated_tokens,
            state.completed_token_ids,
            state.num_completed,
        ),
        before,
    ):
        torch.testing.assert_close(tensor, expected)


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize(
    "fault", ["completed_overflow", "generated_zero", "generated_limit", "decode_mismatch"]
)
def test_capacity_errors_do_not_partially_commit(fault, graph_mode):
    runner = _runtime(stops=(99,))
    state = runner.state
    prefill = _gpu([[1, 2, 3, 99]], [[-0.4, -0.2, -0.3, -0.1]])
    # A successful step must append three EOS completions and select the
    # nonidentity, duplicate parent mapping [1, 1, 1, 2] for the next KV rows.
    decode = _gpu(
        [[10, 11, 12, 99], [20, 21, 22, 99], [30, 31, 32, 99], [40, 41, 42, 99]],
        [[-5, -6, -7, 0], [0, -0.125, -0.25, 0], [-0.5, -1, -2, 0], [0, 0, 0, 0]],
    )
    kv_sources = [
        torch.arange(pool.unshared_key.numel(), dtype=pool.unshared_key.dtype, device="cuda")
        .reshape(pool.unshared_key.shape)
        .add_(layer * 512)
        for layer, pool in enumerate(runner.session.pools)
    ]

    def seed():
        runner.reset()
        runner.initialize_from_prefill(*prefill)
        # Model-side KV writes precede advance. The graph under test contains
        # only postprocessing, so its failure must preserve this entire pool.
        for pool, source in zip(runner.session.pools, kv_sources):
            pool.unshared_key.copy_(source)
            pool.unshared_value.copy_(-source)

    graph = None
    if graph_mode:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            seed()
            runner.advance(*decode)
            seed()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner.advance(*decode)
        torch.cuda.current_stream().wait_stream(stream)

    seed()
    # Faults are injected only AFTER normal capture, with no change of tensor
    # addresses. Keep the other capacity conditions valid to isolate each one.
    if fault == "completed_overflow":
        state.num_completed.fill_(state.completed_capacity)
    elif fault == "generated_zero":
        state.num_generated_tokens.zero_()
        state.num_decode_steps.fill_(-1)  # Preserve decode == generated - 1.
    elif fault == "generated_limit":
        state.num_generated_tokens.fill_(state.max_new_tokens)
        state.num_decode_steps.fill_(state.max_new_tokens - 1)
    else:
        state.num_decode_steps.fill_(1)  # Prefill requires zero, still within KV capacity.

    protected = {
        name: tensor
        for name, tensor in vars(state).items()
        if isinstance(tensor, torch.Tensor) and name != "error_code"
    }
    for layer, pool in enumerate(runner.session.pools):
        protected[f"layer_{layer}_key"] = pool.unshared_key
        protected[f"layer_{layer}_value"] = pool.unshared_value
    # Snapshot after fault injection: the runner must not repair counters or
    # partially modify any persistent row. Workspace scratch may change.
    before = {name: tensor.clone() for name, tensor in protected.items()}
    for _ in range(2):
        if graph is None:
            runner.advance(*decode)
        else:
            graph.replay()
    # No intermediate result readback between the failing calls/replays.
    assert state.error_code.cpu().item() == ERROR_CAPACITY
    assert not runner._scratch["execute"].cpu().item()
    assert not runner._scratch["reorder"].cpu().item()
    for name, tensor in protected.items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0, msg=name)

    # Reuse the same captured graph after reset to prove this was a runtime
    # error guard, rather than a permanently disabled graph or inert fixture.
    seed()
    if graph is None:
        runner.advance(*decode)
    else:
        graph.replay()
    assert state.error_code.cpu().item() == 0
    assert state.num_generated_tokens.cpu().item() == 2
    assert state.num_decode_steps.cpu().item() == 1
    assert state.num_completed.cpu().item() == 4
    assert state.input_ids.cpu().tolist() == [[20, 21, 22, 30]]
    assert state.parent_ids.cpu().tolist() == [[1, 1, 1, 2]]
    for pool, source in zip(runner.session.pools, kv_sources):
        expected = source.clone()
        expected[0, :, :, :1] = source[0, [1, 1, 1, 2], :, :1]
        torch.testing.assert_close(pool.unshared_key, expected, rtol=0, atol=0)
        torch.testing.assert_close(pool.unshared_value, -expected, rtol=0, atol=0)


@pytest.mark.parametrize("kv_capacity", [1, 2], ids=["exhausted", "within_capacity"])
def test_capacity_guard_checks_kv_limit_independently(kv_capacity):
    runner = _runtime(stops=(99,))
    runner.initialize_from_prefill(*_gpu([[1, 2, 3, 99]], [[0, 0, 0, 0]]))
    s, w = runner._state, runner._scratch
    # Generated/Decode counters agree and generation/completed capacities are
    # not exhausted. The constructor's T <= KV capacity + 1 normally makes
    # the KV bound redundant; invoke the guard with its own independent limit.
    s["num_generated_tokens"].fill_(2)
    s["num_decode_steps"].fill_(1)
    w["row_counts"].copy_(torch.tensor([1, 1, 1, 0], dtype=torch.int32, device="cuda"))
    w["row_valid_counts"].copy_(torch.tensor([2, 2, 2, 0], dtype=torch.int32, device="cuda"))
    w["row_errors"].zero_()
    _validate_step_kernel[(1,)](
        s["is_initialized"],
        s["is_finished"],
        s["error_code"],
        s["num_generated_tokens"],
        s["num_decode_steps"],
        s["max_new_tokens_tensor"],
        s["num_completed"],
        w["row_counts"],
        w["row_errors"],
        w["row_valid_counts"],
        w["row_offsets"],
        w["execute"],
        w["reorder"],
        w["next_completed"],
        ROWS=4,
        BLOCK=4,
        CAPACITY=runner.state.completed_capacity,
        KV_CAPACITY=kv_capacity,
        PREFILL=False,
    )
    allowed = kv_capacity == 2
    assert s["error_code"].cpu().item() == (0 if allowed else ERROR_CAPACITY)
    assert w["execute"].cpu().item() == allowed
    assert w["reorder"].cpu().item() == allowed
    assert s["num_generated_tokens"].cpu().item() == 2
    assert s["num_decode_steps"].cpu().item() == 1
    assert s["num_completed"].cpu().item() == 1


def test_same_graph_replay_advances_device_steps_and_resets():
    runner = _runtime(steps=6, batch=2, state_index=1)
    state = runner.state
    state.sequence[0].fill_(321)
    state.history_token_ids[0].fill_(654)
    prefill = _gpu([[10, 11, 12, 13]], [[0, -1, -2, -3]])
    decode = _gpu([[20, 21, 22, 23]] * 4, [[0, -1, -2, -3]] * 4)
    # Warmup on a side stream, then reset before capture. Capture records one
    # update; reset again before replay so semantics do not depend on whether
    # capture itself executes the recorded kernels.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner.initialize_from_prefill(*prefill)
        _write_kv(runner)
        runner.advance(*decode)
        runner.reset()
        runner.initialize_from_prefill(*prefill)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _write_kv(runner)
        runner.advance(*decode)
    torch.cuda.current_stream().wait_stream(stream)
    runner.reset()
    runner.initialize_from_prefill(*prefill)
    addresses = [t.data_ptr() for t in (*state.owned_tensors(), *runner.workspace.device_tensors())]
    for _ in range(5):
        graph.replay()
    # First result read follows all replays.
    assert state.num_generated_tokens[1].cpu().item() == 6
    assert state.num_decode_steps[1].cpu().item() == 5
    assert state.is_finished[1].cpu().item()
    assert state.error_code[1].cpu().item() == 0
    final = _completed(state, 1)
    assert len(final) == 16
    assert final[0][0] == (10, 20, 20, 20, 20, 20)
    assert torch.all(state.sequence[0] == 321)
    assert torch.all(state.history_token_ids[0] == 654)
    runner.reset()
    prefill[0].fill_(77)
    runner.initialize_from_prefill(*prefill)
    for _ in range(5):
        graph.replay()
    assert all(c[0][0] == 77 for c in _completed(state, 1))
    assert [
        t.data_ptr() for t in (*state.owned_tensors(), *runner.workspace.device_tensors())
    ] == addresses


def test_graph_replay_changes_masks_and_eos_without_intermediate_readback():
    runner = _runtime(steps=7, stops=(99,))
    state = runner.state
    initial = ([[10, 11, 99, 12]], [[-0.4, -0.2, -0.1, -0.3]], [[True] * 4])
    data = [
        (
            [[20, 99, -1, -1]] * 4,
            [[0, -0.25, float("nan"), float("nan")]] * 4,
            [[True, True, False, False]] + [[False] * 4] * 3,
        ),
        ([[30, 31, 32, 33]] * 4, [[0, -0.1, -0.2, -0.3]] * 4, [[True] * 4] * 4),
        ([[99, 40, 99, 41]] * 4, [[0, -0.1, -0.2, -0.3]] * 4, [[True] * 4] * 4),
        ([[99] * 4] * 4, [[0, -0.1, -0.2, -0.3]] * 4, [[True] * 4] * 4),
    ]
    prefill = _gpu(*initial)
    sources = [_gpu(*entry) for entry in data]
    inputs = tuple(t.clone() for t in sources[0])
    invalid = _gpu([[-1] * 4] * 4, [[float("nan")] * 4] * 4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner.initialize_from_prefill(*prefill)
        _write_kv(runner)
        runner.advance(*inputs)
        runner.reset()
        runner.initialize_from_prefill(*prefill)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _write_kv(runner)
        runner.advance(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    # Clear capture/warmup KV only for independent reference comparison.
    # Production reset intentionally leaves invalid KV ranges unspecified.
    for pool in runner.session.pools:
        pool.unshared_key.zero_()
        pool.unshared_value.zero_()
    runner.reset()
    runner.initialize_from_prefill(*prefill)
    reference = select_initial_beams(
        _cpu_candidates(*initial, 1), beam_width=4, stop_token_ids=(99,)
    )
    expected_kv: list[NDArray[np.float32]] = [
        np.zeros((4, 7, 8), dtype=np.float32) for _ in range(2)
    ]
    for step, (host_input, source) in enumerate(zip(data, sources)):
        active = reference.items[0].num_active_beams
        for layer in range(2):
            expected_kv[layer][:active, step] = (
                layer * 100 + step * 20 + np.arange(active)[:, None] * 8 + np.arange(8)
            )
        reference = select_next_beams(
            reference, _cpu_candidates(*host_input, active), stop_token_ids=(99,)
        )
        assert reference.items[0].num_active_beams == (1, 4, 4, 0)[step]
        parents = [
            beam.parent_beam_indices[-1] for beam in reference.items[0].beams if beam.is_active
        ]
        for layer in range(2):
            expected_kv[layer][: len(parents), : step + 1] = expected_kv[layer][
                parents, : step + 1
            ].copy()
        for dest, source_tensor in zip(inputs, source):
            dest.copy_(source_tensor)
        graph.replay()
    # Even invalid candidate values cannot alter an already finished state.
    for dest, source_tensor in zip(inputs, invalid):
        dest.copy_(source_tensor)
    graph.replay()
    graph.replay()

    # First readback occurs only after every replay, including terminal no-ops.
    assert state.error_code.cpu().item() == 0
    assert state.num_generated_tokens.cpu().item() == 5
    assert state.num_decode_steps.cpu().item() == 4
    assert state.is_finished.cpu().item()
    assert not state.active_mask.cpu().any()
    assert _completed(state) == [
        (c.token_ids, c.cumulative_logprob, c.parent_beam_indices, FINISH_STOP, c.tie_break_index)
        for c in reference.items[0].completed_beams
    ]
    for layer, pool in enumerate(runner.session.pools):
        np.testing.assert_array_equal(pool.unshared_key[0, :, 0].cpu(), expected_kv[layer])
        np.testing.assert_array_equal(pool.unshared_value[0, :, 0].cpu(), -expected_kv[layer])


@pytest.mark.parametrize("graph_mode", [False, True])
def test_steady_state_profiler_has_no_readback_or_allocation(graph_mode):
    width = 32
    runner = _runtime(width=width, steps=40)
    ranks: NDArray[np.int32] = np.arange(width, dtype=np.int32)
    prefill = _gpu((ranks + 1)[None], -ranks[None])
    decode = _gpu(np.tile(ranks + 100, (width, 1)), np.tile(-ranks, (width, 1)))
    runner.initialize_from_prefill(*prefill)
    runner.advance(*decode)
    runner.reset()
    runner.initialize_from_prefill(*prefill)
    graph = None
    if graph_mode:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            runner.advance(*decode)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner.advance(*decode)
        torch.cuda.current_stream().wait_stream(stream)
        runner.reset()
        runner.initialize_from_prefill(*prefill)
    torch.accelerator.synchronize()
    allocated = torch.accelerator.memory_allocated()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    ) as profile:
        with torch.profiler.record_function("beam_state_steady"):
            for _ in range(16):
                if graph is None:
                    runner.advance(*decode)
                else:
                    graph.replay()
    assert torch.accelerator.memory_allocated() == allocated
    events = profile.events()
    assert not any(e.device_memory_usage > 0 for e in events)
    forbidden = ("dtoh", "device to host", "_local_scalar_dense", "aten::item", "cudamalloc")
    assert not any(any(word in e.name.lower() for word in forbidden) for e in events)
    steady = next(e for e in events if e.name == "beam_state_steady")
    synchronizations = [
        e
        for e in events
        if "synchronize" in e.name.lower()
        and steady.time_range.start <= e.time_range.start < steady.time_range.end
    ]
    assert not synchronizations
    trace_dir = os.environ.get("BEAM_STATE_TRACE_DIR")
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        mode = "graph" if graph_mode else "eager"
        profile.export_chrome_trace(str(Path(trace_dir) / f"beam_state_{mode}.json"))
        summary = {
            "mode": mode,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "steps": 16,
            "beam_width": runner.state.beam_width,
            "max_new_tokens": runner.state.max_new_tokens,
            "allocated_before": allocated,
            "allocated_after": torch.accelerator.memory_allocated(),
            "positive_allocation_events": sum(e.device_memory_usage > 0 for e in events),
            "steady_synchronizations": [e.name for e in synchronizations],
            "forbidden_events": [
                e.name for e in events if any(w in e.name.lower() for w in forbidden)
            ],
            "state_bytes": runner.state.device_memory_bytes,
            "state_scratch_bytes": sum(
                t.numel() * t.element_size()
                for t in runner.workspace.state_scratch.device_tensors()
            ),
            "event_names": sorted({e.name for e in events}),
        }
        (Path(trace_dir) / f"summary_{mode}.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert runner.state.num_generated_tokens.cpu().item() == 17


@pytest.mark.parametrize("width", [1, 3, 32, 256])
def test_width_boundaries_and_final_candidate_history(width):
    runner = _runtime(width=width, steps=3)
    prefill_tokens: NDArray[np.int32] = np.arange(width, dtype=np.int32)[None] + 100
    prefill_scores = np.zeros((1, width), dtype=np.float32)
    decode_tokens: NDArray[np.int32] = (
        np.arange(width**2, dtype=np.int32).reshape(width, width) + 1000
    )
    decode_scores = np.zeros((width, width), dtype=np.float32)
    initial = _gpu(prefill_tokens, prefill_scores)
    decode = _gpu(decode_tokens, decode_scores)
    runner.initialize_from_prefill(*initial)
    runner.advance(*decode)
    runner.advance(*decode)
    state = runner.state
    assert state.error_code.cpu().item() == 0
    assert state.is_finished.cpu().item()
    completed = _completed(state)
    assert len(completed) == width**2
    for index, (tokens, score, parents, reason, tie) in enumerate(completed):
        parent, rank = divmod(index, width)
        assert tokens == (100, 1000 + parent, 1000 + parent * width + rank)
        assert parents == (0, 0, parent)
        assert score == 0
        assert reason == FINISH_LENGTH
        assert tie == index


@pytest.mark.parametrize("width", [1, 2, 128, 256])
def test_full_vocab_candidate_width_boundaries_preserve_model_beam_width(width):
    candidate_width = 2 * width
    runner = _runtime(width=width, steps=3, candidate_width=candidate_width)
    initial = _gpu(
        np.arange(candidate_width, dtype=np.int32)[None] + 100,
        np.zeros((1, candidate_width), dtype=np.float32),
    )
    tokens = np.arange(width * candidate_width, dtype=np.int32).reshape(width, candidate_width)
    decode = _gpu(tokens + 1000, np.zeros(tokens.shape, dtype=np.float32))
    runner.initialize_from_prefill(*initial)
    runner.advance(*decode)
    runner.advance(*decode)
    state = runner.state
    assert state.error_code.cpu().item() == 0
    assert state.is_finished.cpu().item()
    assert state.num_completed.cpu().item() == width * candidate_width
    assert state.active_mask.shape == (1, width)
    torch.testing.assert_close(
        state.completed_token_ids[0, : width * candidate_width].cpu(),
        torch.from_numpy((tokens + 1000).reshape(-1)),
    )
    torch.testing.assert_close(
        state.completed_parent_ids[0, : width * candidate_width].cpu(),
        torch.arange(width, dtype=torch.int32).repeat_interleave(candidate_width),
    )


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
def test_full_vocab_keeps_eos_outside_first_w_and_reorders_wide_candidate_parents(graph_mode):
    runner = _runtime(width=2, steps=3, candidate_width=4, stops=(99,))
    initial = _gpu([[10, 11, 99, 12]], [[-0.01, -0.02, -0.03, -0.04]])
    decode = _gpu(
        [[20, 99, 21, 22], [30, 31, 99, 32]], [[-2, -0.8, -3, -4], [-0.1, -0.2, -0.9, -0.3]]
    )
    final = _gpu([[40, 41, 42, 43], [50, 51, 52, 53]], [[-2, -3, -4, -5]] * 2)
    runner.initialize_from_prefill(*initial)
    if graph_mode:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _write_kv(runner)
            runner.advance(*decode)
            runner.reset()
            runner.initialize_from_prefill(*initial)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _write_kv(runner)
            runner.advance(*decode)
        torch.cuda.current_stream().wait_stream(stream)
        runner.reset()
        runner.initialize_from_prefill(*initial)
        graph.replay()
    else:
        _write_kv(runner)
        runner.advance(*decode)
    # Winners both came from parent one, whose flat candidate indices are 4/5.
    # Dividing by W instead of K corrupts the parent/KV mapping here.
    assert runner.state.parent_ids[0].cpu().tolist() == [1, 1]
    assert runner.state.sequence[0, :, :2].cpu().tolist() == [[11, 30], [11, 31]]
    for target, source in zip(decode, final):
        target.copy_(source)
    if graph_mode:
        graph.replay()
    else:
        _write_kv(runner)
        runner.advance(*decode)
    state = runner.state
    assert state.error_code.cpu().item() == 0
    completed = _completed(state)
    assert len(completed) == 11
    assert [row[0] for row in completed[:3]] == [(99,), (10, 99), (11, 99)]
    assert all(row[0][:2] in ((11, 30), (11, 31)) for row in completed[3:])
    assert completed[0][1] == pytest.approx(-0.03)
    # The saved prefill EOS can outrank every later continuation. Its length
    # includes the prompt (which contains SID begin), but excludes EOS itself.
    ranking = torch.empty_like(state.completed_scores[0])
    _rank_final_beams_kernel[(triton.cdiv(state.completed_capacity, 256),)](
        state.completed_scores[0],
        state.completed_lengths[0],
        state.completed_finish_reasons[0],
        ranking,
        CAPACITY=state.completed_capacity,
        PROMPT_LENGTH=5,
        MAX_TOKENS=3,
        LENGTH_PENALTY=1.0,
        BLOCK=256,
    )
    assert ranking.argmax().cpu().item() == 0
    assert ranking[0].cpu().item() == pytest.approx(-0.03 * 8 / 5)


def test_final_compaction_preserves_rank_and_parent_lineage():
    runner = _runtime(width=4, steps=3)
    runner.initialize_from_prefill(*_gpu([[100, 101, 102, 103]], [[0, -1, -2, -3]]))
    tokens = np.arange(16, dtype=np.int32).reshape(4, 4) + 1000
    runner.advance(*_gpu(tokens, np.zeros((4, 4), dtype=np.float32)))
    runner.advance(*_gpu(tokens, np.zeros((4, 4), dtype=np.float32)))
    state = runner.state
    scratch = allocate_beam_selection_scratch(
        1, 1, state.completed_capacity, state.beam_width, state.device
    )
    _, indices, _ = _hierarchical_topk_out(
        torch.zeros((1, 1), device="cuda"),
        state.completed_scores,
        None,
        1,
        1,
        state.completed_capacity,
        state.beam_width,
        scratch,
    )
    layout = _PayloadLayout.create(state.beam_width, state.max_new_tokens)
    payload = torch.zeros(layout.size, dtype=torch.int32, device="cuda")
    _pack_final_beams_kernel[(state.beam_width,)](
        indices,
        state.num_completed,
        state.completed_token_ids[0],
        state.completed_scores[0],
        state.completed_parent_ids[0],
        state.completed_lengths[0],
        state.completed_finish_reasons[0],
        state.history_token_ids[0],
        state.history_parent_ids[0],
        payload[3:],
        WIDTH=state.beam_width,
        STEPS=state.max_new_tokens,
        RESULT_WIDTH=state.beam_width,
        ROW_STRIDE=layout.row_stride,
        ROW_BLOCK=triton.next_power_of_2(2 * state.max_new_tokens),
    )
    compact = payload.cpu()
    assert compact[3].item() == 4
    rows = compact[4:].view(4, layout.row_stride)
    assert rows[:, :3].tolist() == [
        [100, 1000, 1000],
        [100, 1000, 1001],
        [100, 1000, 1002],
        [100, 1000, 1003],
    ]
    assert rows[:, 3:6].tolist() == [[0, 0, 0]] * 4
    assert rows[:, 7].tolist() == [3] * 4
    assert rows[:, 8].tolist() == [FINISH_LENGTH] * 4


def test_no_candidates_and_masked_kv_rows():
    runner = _runtime(stops=(99,))
    initial = _gpu(
        [[99, 10, -1, -1]], [[0, 0, float("nan"), float("nan")]], [[True, True, False, False]]
    )
    runner.initialize_from_prefill(*initial)
    for pool in runner.session.pools:
        pool.unshared_key.fill_(123)
        pool.unshared_value.fill_(-123)
    runner.advance(
        *_gpu(
            [[20, -1, -1, -1]] * 4, [[0, float("nan"), 0, 0]] * 4, [[True, False, False, False]] * 4
        )
    )
    assert runner.state.active_mask.cpu().tolist() == [[True, False, False, False]]
    for pool in runner.session.pools:
        assert torch.all(pool.unshared_key[0, 1:] == 123)
        assert torch.all(pool.unshared_value[0, 1:] == -123)
    empty = _gpu([[-1] * 4] * 4, [[float("nan")] * 4] * 4, [[False] * 4] * 4)
    runner.advance(*empty)
    assert runner.state.is_finished.cpu().item()
    assert runner.state.error_code.cpu().item() == 0
    assert len(_completed(runner.state)) == 1
    runner.reset()
    runner.initialize_from_prefill(*_gpu([[-1] * 4], [[float("nan")] * 4], [[False] * 4]))
    assert runner.state.error_code.cpu().item() == ERROR_NO_CANDIDATES
    assert runner.state.num_generated_tokens.cpu().item() == 0


def test_events_order_inputs_consumption_and_slot_reuse():
    runner = _runtime(steps=2)
    state, workspace = runner.state, runner.workspace
    addresses = [t.data_ptr() for t in (*state.owned_tensors(), *workspace.device_tensors())]
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    ready = torch.Event()
    with torch.cuda.stream(producer):
        initial = _gpu([[1, 2, 3, 4]], [[0] * 4])
        decode = _gpu([[10, 11, 12, 13]] * 4, [[0] * 4] * 4)
        ready.record()
    torch.cuda.current_stream().wait_event(ready)
    runner.initialize_from_prefill(*initial)
    runner.advance(*decode)
    beam_done = torch.Event()
    beam_done.record()
    consumer = torch.cuda.Stream()
    consumer.wait_event(beam_done)
    consumed = torch.Event()
    with torch.cuda.stream(consumer):
        snapshot = state.completed_token_ids.clone()
        consumed.record()
    context = runner.session.context
    runner.end_session(consumed)
    assert not context.registry.has_session("s")
    with pytest.raises(RuntimeError, match="released"):
        runner.advance(*decode)
    with pytest.raises(RuntimeError, match="released"):
        runner.reset()
    slot = context.registry.allocate("b", 4, 0)
    assert slot == runner.session.slot
    session_b = BeamSearchSession("b", slot, context)
    session_b.begin_session(4, 0, decision_index=0)
    runner_b = BeamSearchRunner(session_b, state, workspace)
    runner_b.reset()
    runner_b.initialize_from_prefill(*_gpu([[77, 78, 79, 80]], [[0] * 4]))
    runner_b.advance(*decode)
    assert [
        t.data_ptr() for t in (*state.owned_tensors(), *workspace.device_tensors())
    ] == addresses
    assert all(c[0][0] >= 77 for c in _completed(state))
    assert snapshot[0, :16].cpu().tolist() == [10, 11, 12, 13] * 4
    runner_b.end_session()


def test_device_kv_reorder_handles_unsorted_duplicate_parents():
    runner = _runtime(steps=5)
    state = runner.state
    state.parent_ids.copy_(torch.tensor([[3, 1, 1, 0]], device="cuda", dtype=torch.int32))
    state.active_mask.copy_(torch.tensor([[True, True, True, False]], device="cuda"))
    state.num_decode_steps.fill_(2)
    runner._scratch["reorder"].fill_(True)
    expected = []
    for layer, pool in enumerate(runner.session.pools):
        source = (
            torch.arange(
                pool.unshared_key.numel(), dtype=pool.unshared_key.dtype, device="cpu"
            ).reshape(pool.unshared_key.shape)
            + layer * 100
        )
        pool.unshared_key.copy_(source)
        pool.unshared_value.copy_(-source)
        reordered = source.clone()
        reordered[0, :3, :, :3] = source[0, [3, 1, 1], :, :3]
        expected.append(reordered)
    _select_kv_gather_kernel[runner._kv_grid](
        *runner._kv_args, runner._state["parent_ids"], **runner._kv_kwargs
    )
    _select_kv_scatter_kernel[runner._kv_grid](*runner._kv_args, **runner._kv_kwargs)
    for pool, reference in zip(runner.session.pools, expected):
        torch.testing.assert_close(pool.unshared_key.cpu(), reference, rtol=0, atol=0)
        torch.testing.assert_close(pool.unshared_value.cpu(), -reference, rtol=0, atol=0)


def test_prefill_all_eos_and_layout_validation():
    runner = _runtime(stops=(99,))
    prefill = _gpu([[99, 99, 99, 99]], [[0, -1, -2, -3]])
    with pytest.raises(ValueError, match="token_ids"):
        runner.initialize_from_prefill(prefill[0].to(torch.int64), *prefill[1:])
    with pytest.raises(ValueError, match="token_ids"):
        runner.advance(*prefill)
    runner.initialize_from_prefill(*prefill)
    assert runner.state.is_finished.cpu().item()
    assert runner.state.num_generated_tokens.cpu().item() == 1
    assert runner.state.num_decode_steps.cpu().item() == 0
    assert len(_completed(runner.state)) == 4
