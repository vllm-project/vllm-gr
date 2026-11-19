# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts for NPU stage preparation and sampling."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests._npu_model_runner_patch_helpers import (
    _CountingPositionTuple,
    _CUDAPositionTensor,
    _eager_stage_output,
    _NPUPositionTensor,
    _position_plan,
)
from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.v1.beam.execution_batch import BeamExecutionContractError
from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.npu_beam_stage_adapter import (
    NPUEagerStagePlan,
    validate_npu_eager_capabilities,
)
from vllm_gr.v1.worker.npu_beam_stage_runner import NPUEagerBeamStageRunner
from vllm_gr.v1.worker.npu_model_runner_inputs import InputBatchProxy


@pytest.mark.parametrize(  # type: ignore[misc]
    ("decode_step", "expected_positions"),
    [(0, [2, 2, 2, 3, 3, 3, -1, -1]), (1, [3, 3, 3, 4, 4, 4, -1, -1])],
)
def test_npu_eager_decode_binds_previous_winners(
    decode_step: int, expected_positions: list[int]
) -> None:
    plan = NPUEagerStagePlan.from_scheduler_output(
        _eager_stage_output(kind="decode", decode_step=decode_step)
    )
    input_ids = torch.full((8,), -1, dtype=torch.int32)
    positions = torch.full((8,), -1, dtype=torch.int64)
    winners = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)

    plan.prepare_decode_inputs(input_ids, positions, winners)

    assert input_ids.tolist() == [11, 12, 13, 21, 22, 23, -1, -1]
    assert positions.tolist() == expected_positions


@pytest.mark.parametrize("beam_width", [1, 3, 128])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "prompt_lengths",
    [
        (128,),
        (128, 128, 128, 128),
        (1, 127, 4096, 8192),
        (128, 128, 4096, 4096),
        (128, 4096, 128, 4096),
    ],
)
def test_npu_decode_positions_read_only_request_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    beam_width: int,
    prompt_lengths: tuple[int, ...],
) -> None:
    plan = _position_plan(prompt_lengths, 1, beam_width)
    assert plan.decode_positions is not None
    expected = torch.tensor(plan.decode_positions, dtype=torch.int64)
    counted = _CountingPositionTuple(plan.decode_positions)
    plan = replace(plan, decode_positions=counted)
    rows = expected.numel()
    input_ids = torch.full((rows + 2,), -7, dtype=torch.int32)
    positions = torch.full((rows + 2,), -9, dtype=torch.int64).as_subclass(_NPUPositionTensor)
    winners = torch.arange(rows, dtype=torch.int32).reshape(len(prompt_lengths), beam_width)
    calls: list[tuple[int, int, int]] = []

    def fill(tensor: torch.Tensor, value: int) -> torch.Tensor:
        calls.append((tensor.storage_offset(), tensor.numel(), value))
        return torch.Tensor.fill_(tensor, value)

    monkeypatch.setattr(_NPUPositionTensor, "fill_", fill)
    plan.prepare_decode_inputs(input_ids, positions, winners)

    # This is independent of width: the former per-row scan reads B * W
    # positions, including interior rows, even when every request is equal.
    assert counted.reads == [index * beam_width for index in range(len(prompt_lengths))]
    expected_runs = sum(
        index == 0 or length != prompt_lengths[index - 1]
        for index, length in enumerate(prompt_lengths)
    )
    assert len(calls) == expected_runs
    assert sum(count for _offset, count, _value in calls) == rows
    torch.testing.assert_close(positions[:rows].as_subclass(torch.Tensor), expected)
    torch.testing.assert_close(input_ids[:rows], winners.flatten())
    assert positions[rows:].as_subclass(torch.Tensor).tolist() == [-9, -9]
    assert input_ids[rows:].tolist() == [-7, -7]


@pytest.mark.parametrize(  # type: ignore[misc]
    "dynamic_beam_widths", [(128, 64, 32), (32, 128, 256)]
)
@pytest.mark.parametrize("step", [0, 1])  # type: ignore[misc]
@pytest.mark.parametrize("prompt_lengths", [(128,), (128, 128, 4096, 4096)])  # type: ignore[misc]
def test_npu_decode_positions_dynamic_boundaries_use_active_input_width(
    monkeypatch: pytest.MonkeyPatch,
    dynamic_beam_widths: tuple[int, ...],
    step: int,
    prompt_lengths: tuple[int, ...],
) -> None:
    capacity = 256
    plan = _position_plan(prompt_lengths, step, capacity, dynamic_beam_widths)
    width = dynamic_beam_widths[step]
    assert plan.execution.input_width == width < capacity
    assert plan.decode_positions is not None
    expected = torch.tensor(plan.decode_positions, dtype=torch.int64)
    counted = _CountingPositionTuple(plan.decode_positions)
    plan = replace(plan, decode_positions=counted)
    rows = len(prompt_lengths) * width
    # Capacity is larger than the active input: never fill inactive Beam rows.
    input_ids = torch.full((len(prompt_lengths) * capacity + 2,), -7, dtype=torch.int32)
    positions = torch.full_like(input_ids, -9, dtype=torch.int64).as_subclass(_NPUPositionTensor)
    winners = torch.arange(rows, dtype=torch.int32).reshape(len(prompt_lengths), width)
    calls: list[tuple[int, int, int]] = []

    def fill(tensor: torch.Tensor, value: int) -> torch.Tensor:
        calls.append((tensor.storage_offset(), tensor.numel(), value))
        return torch.Tensor.fill_(tensor, value)

    monkeypatch.setattr(_NPUPositionTensor, "fill_", fill)
    plan.prepare_decode_inputs(input_ids, positions, winners)

    assert counted.reads == [index * width for index in range(len(prompt_lengths))]
    expected_calls = (
        [(0, width, prompt_lengths[0] + step)]
        if len(prompt_lengths) == 1
        else [(0, 2 * width, 128 + step), (2 * width, 2 * width, 4096 + step)]
    )
    assert calls == expected_calls
    torch.testing.assert_close(positions[:rows].as_subclass(torch.Tensor), expected)
    torch.testing.assert_close(input_ids[:rows], winners.flatten())
    assert torch.all(positions[rows:].as_subclass(torch.Tensor) == -9)
    assert torch.all(input_ids[rows:] == -7)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("prompt_lengths", "runs"),
    [
        ((128,), 1),
        ((128, 128, 128, 128), 1),
        ((1, 127, 4096, 8192), 4),
        ((128, 128, 4096, 4096), 2),
        ((128, 4096, 128, 4096), 4),
    ],
)
def test_npu_decode_positions_fill_only_adjacent_equal_rows_without_host_transfer(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    prompt_lengths: tuple[int, ...],
    runs: int,
) -> None:
    rows = len(prompt_lengths) * 3
    input_ids = torch.full((rows + 2,), -7, dtype=torch.int32)
    positions = torch.full((rows + 2,), -9, dtype=dtype).as_subclass(_NPUPositionTensor)
    pointers = (input_ids.data_ptr(), positions.data_ptr())
    calls: list[tuple[int, int, int]] = []

    def fill(tensor: torch.Tensor, value: int) -> torch.Tensor:
        calls.append((tensor.storage_offset(), tensor.numel(), value))
        return torch.Tensor.fill_(tensor, value)

    def forbid_host_transfer(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("NPU position binding must not create or read a host tensor")

    monkeypatch.setattr(_NPUPositionTensor, "fill_", fill)
    for step in (0, 1, 7):
        plan = _position_plan(prompt_lengths, step)
        winners = torch.arange(rows, dtype=torch.int32) + step * 100
        expected_positions = torch.tensor(plan.decode_positions, dtype=dtype)
        calls.clear()
        with monkeypatch.context() as guard:
            guard.setattr(torch, "as_tensor", forbid_host_transfer)
            guard.setattr(torch.Tensor, "cpu", forbid_host_transfer)
            guard.setattr(torch.Tensor, "item", forbid_host_transfer)
            guard.setattr(torch.Tensor, "tolist", forbid_host_transfer)
            plan.prepare_decode_inputs(
                input_ids,
                positions,
                winners if step == 1 else winners.reshape(len(prompt_lengths), 3),
            )
        assert len(calls) == runs
        assert sum(count for _offset, count, _value in calls) == rows
        assert pointers == (input_ids.data_ptr(), positions.data_ptr())
        torch.testing.assert_close(positions[:rows].as_subclass(torch.Tensor), expected_positions)
        torch.testing.assert_close(input_ids[:rows], winners)
        assert positions[rows:].as_subclass(torch.Tensor).tolist() == [-9, -9]
        assert input_ids[rows:].tolist() == [-7, -7]


@pytest.mark.parametrize("mock_cuda", [False, True])  # type: ignore[misc]
def test_non_npu_decode_positions_keep_original_tensor_copy(
    monkeypatch: pytest.MonkeyPatch, mock_cuda: bool
) -> None:
    plan = _position_plan((128, 4096), 1)
    positions = torch.full((8,), -9, dtype=torch.int64)
    if mock_cuda:
        positions = positions.as_subclass(_CUDAPositionTensor)
    input_ids = torch.full((8,), -7, dtype=torch.int32)
    winners = torch.arange(6, dtype=torch.int32)
    original = torch.as_tensor
    calls: list[tuple[Any, torch.dtype, str]] = []

    def as_tensor(data: Any, *, dtype: torch.dtype, device: Any) -> torch.Tensor:
        calls.append((data, dtype, device.type))
        # Fake CUDA storage is still CPU storage, so keep the real operation on CPU.
        return original(data, dtype=dtype, device="cpu")

    monkeypatch.setattr(torch, "as_tensor", as_tensor)
    plan.prepare_decode_inputs(input_ids, positions, winners)
    assert calls == [(plan.decode_positions, torch.int64, "cuda" if mock_cuda else "cpu")]
    torch.testing.assert_close(
        positions.as_subclass(torch.Tensor),
        torch.tensor([129, 129, 129, 4097, 4097, 4097, -9, -9]),
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    ("invalid", "message"),
    [
        ("prefill", "requires a Decode plan"),
        ("input-rank", "input_ids has insufficient"),
        ("input-capacity", "input_ids has insufficient"),
        ("position-rank", "positions has insufficient"),
        ("position-capacity", "positions has insufficient"),
        ("winner-shape", "winner_token_ids must have shape"),
        ("winner-device", "must be on the same device"),
    ],
)
def test_npu_decode_positions_retain_input_validation(invalid: str, message: str) -> None:
    plan = _position_plan((128,), 1)
    input_ids = torch.full((5,), -7, dtype=torch.int32)
    positions = torch.full((5,), -9, dtype=torch.int64).as_subclass(_NPUPositionTensor)
    winners = torch.arange(3, dtype=torch.int32)
    if invalid == "prefill":
        plan = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="prefill"))
    elif invalid == "input-rank":
        input_ids = input_ids.unsqueeze(0)
    elif invalid == "input-capacity":
        input_ids = input_ids[:2]
    elif invalid == "position-rank":
        positions = positions.unsqueeze(0)
    elif invalid == "position-capacity":
        positions = positions[:2]
    elif invalid == "winner-shape":
        winners = winners.unsqueeze(1)
    elif invalid == "winner-device":
        winners = winners.as_subclass(_CUDAPositionTensor)
    with pytest.raises(BeamExecutionContractError, match=message):
        plan.prepare_decode_inputs(input_ids, positions, winners)
    assert torch.all(input_ids == -7)
    assert torch.all(positions.as_subclass(torch.Tensor) == -9)


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "prompt_lengths",
    [(128,), (128, 128, 128, 128), (1, 127, 4096, 8192), (128, 4096, 128, 4096)],
)
def test_npu_decode_positions_match_original_construction_across_async_steps(
    dtype: torch.dtype, prompt_lengths: tuple[int, ...]
) -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    plans = [_position_plan(prompt_lengths, step, beam_width=128) for step in (0, 1, 7)]
    rows = len(prompt_lengths) * 128
    input_ids = torch.full((rows + 2,), -7, dtype=torch.int32, device="npu")
    positions = torch.full((rows + 2,), -9, dtype=dtype, device="npu")
    winners = torch.empty((len(prompt_lengths), 128), dtype=torch.int32, device="npu")
    original_positions = [
        torch.as_tensor(plan.decode_positions, dtype=dtype, device=positions.device)
        for plan in plans
    ]
    pointers = (input_ids.data_ptr(), positions.data_ptr(), winners.data_ptr())
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    snapshots = []
    with torch.npu.stream(stream):
        for index, plan in enumerate(plans):
            winners.fill_(100 + index)
            plan.prepare_decode_inputs(input_ids, positions, winners)
            snapshots.append((input_ids.clone(), positions.clone()))
    # No host reads or synchronization between successive device updates.
    stream.synchronize()
    assert pointers == (input_ids.data_ptr(), positions.data_ptr(), winners.data_ptr())
    for index, ((ids, actual), expected) in enumerate(zip(snapshots, original_positions)):
        torch.testing.assert_close(actual[:rows], expected, rtol=0, atol=0)
        assert ids[:rows].cpu().tolist() == [100 + index] * rows
        assert ids[rows:].cpu().tolist() == [-7, -7]
        assert actual[rows:].cpu().tolist() == [-9, -9]


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "dynamic_beam_widths", [(128, 64, 32), (32, 128, 256)]
)
@pytest.mark.parametrize("prompt_lengths", [(128,), (128, 128, 4096, 4096)])  # type: ignore[misc]
def test_npu_decode_positions_dynamic_widths_match_original_across_async_steps(
    dtype: torch.dtype,
    dynamic_beam_widths: tuple[int, ...],
    prompt_lengths: tuple[int, ...],
) -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    capacity = 256
    batch = len(prompt_lengths)
    plans = [_position_plan(prompt_lengths, step, capacity, dynamic_beam_widths) for step in (0, 1)]
    input_ids = torch.full((batch * capacity + 2,), -7, dtype=torch.int32, device="npu")
    positions = torch.full((batch * capacity + 2,), -9, dtype=dtype, device="npu")
    winners = torch.empty((batch * capacity,), dtype=torch.int32, device="npu")
    expected_positions = [
        torch.as_tensor(plan.decode_positions, dtype=dtype, device="npu") for plan in plans
    ]
    pointers = (input_ids.data_ptr(), positions.data_ptr(), winners.data_ptr())
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    snapshots = []
    with torch.npu.stream(stream):
        for step, plan in enumerate(plans):
            width = dynamic_beam_widths[step]
            rows = batch * width
            input_ids.fill_(-7)
            positions.fill_(-9)
            winners.fill_(100 + step)
            plan.prepare_decode_inputs(input_ids, positions, winners[:rows].view(batch, width))
            snapshots.append((input_ids.clone(), positions.clone()))
    stream.synchronize()
    assert pointers == (input_ids.data_ptr(), positions.data_ptr(), winners.data_ptr())
    for step, ((ids, actual), expected) in enumerate(zip(snapshots, expected_positions)):
        rows = batch * dynamic_beam_widths[step]
        torch.testing.assert_close(actual[:rows], expected, rtol=0, atol=0)
        assert ids[:rows].cpu().tolist() == [100 + step] * rows
        assert torch.all(ids[rows:] == -7)
        assert torch.all(actual[rows:] == -9)


def test_npu_capability_gate_allows_graph_mode_and_remains_npu_only() -> None:
    worker = SimpleNamespace(
        device="npu",
        model_config=SimpleNamespace(enforce_eager=True, dtype=torch.bfloat16),
        use_async_scheduling=True,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        speculative_config=None,
    )
    validate_npu_eager_capabilities(worker)
    worker.model_config.enforce_eager = False
    validate_npu_eager_capabilities(worker)

    worker.device = "cuda"
    with pytest.raises(ValueError, match="NPU worker"):
        validate_npu_eager_capabilities(worker)


@pytest.mark.parametrize("decode_step", [0, 1])  # type: ignore[misc]
def test_npu_stage_runner_binds_physical_decode_without_mutating_scheduler(
    decode_step: int,
) -> None:
    scheduler_output = _eager_stage_output(kind="decode", decode_step=decode_step)
    plan = NPUEagerStagePlan.from_scheduler_output(scheduler_output)
    winners = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([0.5, 0.9]),
        top_p=torch.tensor([0.8, 0.7]),
        top_k=torch.tensor([5, 9], dtype=torch.int32),
    )
    input_batch = SimpleNamespace(
        req_ids=list(plan.execution.request_ids), sampling_metadata=sampling_metadata
    )
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        input_ids=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
        positions=torch.zeros(8, dtype=torch.int64),
        input_batch=input_batch,
        _current_scheduler_output=scheduler_output,
        _current_beam_info={"req_ids": []},
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker = worker
    stage.pending_plan = plan
    stage.latest_token_ids = winners
    runtime_output = stage._runtime_scheduler_output(scheduler_output, plan)
    worker._current_scheduler_output = runtime_output

    def original(self: Any, physical_output: Any, counts: Any) -> tuple[Any, ...]:
        assert self is worker
        assert physical_output is runtime_output
        assert physical_output is not scheduler_output
        assert physical_output.num_scheduled_tokens == {
            "request-0": 3,
            "request-1": 3,
        }
        assert counts.dtype == np.int32
        assert torch.from_numpy(counts).tolist() == [3, 3]
        return object(), object(), 6

    counts = np.asarray([1, 1], dtype=np.int32)
    logits, _, total = stage.prepare_inputs(original, runtime_output, counts)

    assert counts.tolist() == [1, 1]
    assert scheduler_output.num_scheduled_tokens == {
        "request-0": 1,
        "request-1": 1,
    }
    assert worker.input_ids.gpu[:6].tolist() == [11, 12, 13, 21, 22, 23]
    assert worker.positions[:6].tolist() == [value + decode_step for value in (2, 2, 2, 3, 3, 3)]
    assert logits.tolist() == list(range(6))
    assert total == 6
    assert isinstance(worker.input_batch, InputBatchProxy)
    # Bookkeeping retains B logical ids; indexed/copy access maps B*W rows.
    assert len(worker.input_batch.req_ids) == 2
    assert worker.input_batch.req_ids.copy() == ["request-0"] * 3 + ["request-1"] * 3
    assert worker.input_batch.req_ids[3] == "request-1"
    assert input_batch.req_ids == ["request-0", "request-1"]
    assert worker._current_beam_info["req_ids"] == input_batch.req_ids
    remapped = worker.input_batch.sampling_metadata
    assert remapped is sampling_metadata
    assert remapped.temperature.tolist() == pytest.approx([0.5] * 3 + [0.9] * 3)
    assert remapped.top_p.tolist() == pytest.approx([0.8] * 3 + [0.7] * 3)
    assert remapped.top_k.tolist() == [5] * 3 + [9] * 3
    assert remapped.top_k.dtype == torch.int32
    assert remapped.temperature.is_contiguous() and remapped.top_p.is_contiguous()


@pytest.mark.parametrize("device", ["cuda", "npu", "cpu"])  # type: ignore[misc]
def test_shared_stage_dispatch_routes_devices(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    from unittest.mock import Mock

    import vllm_gr.v1.worker.gr_execution as gr_execution
    from vllm_gr.v1.worker import gpu_beam_stage_runner, npu_beam_stage_runner

    output = object()
    stage = SimpleNamespace(execute=Mock(return_value=output))
    cuda = Mock(return_value=stage)
    npu = Mock(return_value=stage)
    monkeypatch.setattr(gpu_beam_stage_runner, "get_stage_runner", cuda)
    monkeypatch.setattr(npu_beam_stage_runner, "get_npu_stage_runner", npu)
    hooks = object()
    worker = SimpleNamespace(device=device)
    scheduled = _eager_stage_output(kind="prefill")
    if device == "cpu":
        with pytest.raises(ValueError, match="no execution adapter"):
            gr_execution._dispatch_stage_frame(worker, scheduled, object(), hooks_type=hooks)
        cuda.assert_not_called()
        npu.assert_not_called()
    else:
        assert (
            gr_execution._dispatch_stage_frame(worker, scheduled, object(), hooks_type=hooks)
            is output
        )
        if device == "cuda":
            assert cuda.call_args.kwargs["hooks_type"] is hooks
            npu.assert_not_called()
        else:
            npu.assert_called_once()
            cuda.assert_not_called()


def test_npu_graph_context_survives_shape_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
    from vllm_gr.v1.worker import gr_context

    def context(batch_size: int) -> BeamAttentionContext:
        return BeamAttentionContext(batch_size, 4, 1, 3, 2, 1, torch.device("cpu"), torch.float16)

    graph_context = context(2)

    def pointers() -> tuple[int, ...]:
        return tuple(
            tensor.data_ptr()
            for tensor in (
                graph_context.pools[0].unshared_key,
                graph_context.pools[0].unshared_value,
                graph_context.decision.sequence,
                graph_context.decision.beam_scores,
                graph_context.decode_step_buf,
                graph_context.decode_step_1based_buf,
                graph_context.xllm_block_table_buf,
            )
        )

    captured_pointers = pointers()
    worker = SimpleNamespace(
        _gr_npu_graph_context=graph_context, _gr_npu_beam_graph_shapes={(2, 4)}
    )
    created: list[int] = []

    def create(worker: Any, config: Any, *, max_batch: int) -> BeamAttentionContext:
        created.append(max_batch)
        return context(max_batch)

    monkeypatch.setattr(gr_context, "_create_beam_context", create)
    monkeypatch.setattr(gr_context, "_get_gr_config", lambda worker: object())
    for batch_size in (2, 1, 2):
        scheduled = _eager_stage_output(kind="decode", batch_size=batch_size, beam_width=4)
        next(iter(scheduled.gr_stage_metadata.values())).beam_params.max_tokens = 3
        worker._current_scheduler_output = scheduled
        stage = object.__new__(NPUEagerBeamStageRunner)
        stage.worker, stage.attention_batch = worker, None
        stage.prefill_candidates = ConstraintOutput(
            token_ids=torch.arange(batch_size * 4, dtype=torch.int32).view(batch_size, 4),
            logprobs=torch.zeros(batch_size, 4),
            valid_mask=torch.ones(batch_size, 4, dtype=torch.bool),
        )
        stage._ensure_attention_batch(NPUEagerStagePlan.from_scheduler_output(scheduled))
        assert stage.attention_batch is not None
        expected = graph_context if batch_size == 2 else worker._gr_npu_eager_context
        assert stage.attention_batch.context is expected
        assert worker._gr_npu_graph_context is graph_context
        assert pointers() == captured_pointers
        assert expected.registry.num_active_sessions == batch_size
        stage._release()
        assert expected.registry.num_active_sessions == 0 and not worker._beam_sessions
    assert created == [1]


def test_npu_first_decode_native_contract_keeps_six_forward_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror the 0.30.0rc2 first-decode call order through the real GR hooks."""
    from vllm.v1.cudagraph_dispatcher import CUDAGraphMode

    from vllm_gr.v1.worker import (
        gr_execution,
        npu_beam_stage_runner,
        npu_model_runner_inputs,
    )

    logical = _eager_stage_output(kind="decode")
    metadata = next(iter(logical.gr_stage_metadata.values()))
    metadata.is_last_stage = False
    metadata.beam_params.max_tokens = 3
    prefill = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="prefill"))
    winners = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    with torch.inference_mode():
        context = SimpleNamespace(
            decode_step_buf=torch.zeros(1, dtype=torch.int32),
            decode_step_1based_buf=torch.zeros(1, dtype=torch.int32),
            xllm_block_table_buf=torch.zeros(2, dtype=torch.int32),
        )
    worker = SimpleNamespace(
        device="npu",
        input_ids=SimpleNamespace(gpu=torch.zeros(6, dtype=torch.int32)),
        positions=torch.zeros(6, dtype=torch.int64),
        input_batch=SimpleNamespace(req_ids=list(logical.num_scheduled_tokens)),
        execute_model_state=None,
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=False),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.previous = worker, prefill.execution
    stage.closed, stage.pending_plan, stage.execution_error = False, None, None
    stage.latest_token_ids = winners
    stage.session_request = None
    stage.attention_batch = SimpleNamespace(
        execution=NPUEagerStagePlan.from_scheduler_output(logical).execution,
        is_active=True,
        slot=0,
        sessions=tuple(
            SimpleNamespace(_state=SimpleNamespace(is_last_step=False)) for _ in winners
        ),
        context=context,
    )
    monkeypatch.setattr(npu_beam_stage_runner, "get_npu_stage_runner", lambda *args: stage)
    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *args, **kwargs: arange(*args, dtype=torch.int32))
    monkeypatch.setattr(
        npu_model_runner_inputs, "_materialize_logits_indices", lambda _, rows: torch.tensor(rows)
    )
    seen: list[str] = []

    def update(self: Any, scheduled: Any) -> None:
        assert scheduled.total_num_scheduled_tokens == 2
        assert tuple(scheduled.num_scheduled_tokens.values()) == (1, 1)
        seen.append("logical_state")

    worker._update_states = npu_patch.make_patched_update_states(update).__get__(worker)

    def prepare(self: Any, scheduled: Any, counts: np.ndarray) -> tuple[Any, ...]:
        assert counts.dtype == np.int32
        assert torch.from_numpy(counts).tolist() == [3, 3]
        assert scheduled.total_num_scheduled_tokens == 6
        seen.append("physical_inputs")
        return object(), None, int(counts.sum())

    worker._prepare_inputs = npu_patch.make_patched_prepare_inputs(prepare).__get__(worker)

    def determine(self: Any, **kwargs: Any) -> tuple[Any, ...]:
        assert kwargs["num_tokens"] == 6
        assert kwargs["max_num_scheduled_tokens"] == 3
        assert kwargs["num_scheduled_tokens_np"].tolist() == [3, 3]
        seen.append("physical_padding")
        return CUDAGraphMode.NONE, SimpleNamespace(num_tokens=6), False, None, None

    worker._determine = npu_patch.make_patched_determine_batch_execution_and_padding(
        determine
    ).__get__(worker)

    def native(self: Any, runtime: Any) -> Any:
        assert torch.is_inference_mode_enabled()
        # Native execute reads this scalar before updating request state.
        assert runtime.total_num_scheduled_tokens == 6
        self._update_states(runtime)
        counts = np.asarray(list(runtime.num_scheduled_tokens.values()), dtype=np.int32)
        maximum = int(counts.max())
        logits_indices, _, total = self._prepare_inputs(runtime, counts)
        assert total == 6 and logits_indices.tolist() == list(range(6))
        _, descriptor, *_ = self._determine(
            num_tokens=runtime.total_num_scheduled_tokens,
            num_scheduled_tokens_np=counts,
            max_num_scheduled_tokens=maximum,
        )
        # Native preprocess slices by the descriptor, and forward consumes that slice.
        input_ids = self.input_ids.gpu[: descriptor.num_tokens]
        assert input_ids.tolist() == winners.reshape(-1).tolist()
        assert self.positions[: descriptor.num_tokens].tolist() == [2, 2, 2, 3, 3, 3]
        seen.append("six_token_forward")
        return SimpleNamespace(logits=input_ids[:, None].float())

    result = gr_execution._dispatch_stage_frame(worker, logical, native)
    assert result.logits.shape == (6, 1)
    assert logical.total_num_scheduled_tokens == 2
    assert tuple(logical.num_scheduled_tokens.values()) == (1, 1)
    assert seen == ["logical_state", "physical_inputs", "physical_padding", "six_token_forward"]
    # Ordinary requests bypass the logical GR view after sampling is complete.
    stage.pending_plan = None
    worker._update_states(logical)


@pytest.mark.parametrize("failure_point", ["plan", "forward"])  # type: ignore[misc]
def test_npu_failed_execute_reaches_sample_without_native_fallback(
    monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    from unittest.mock import Mock

    from vllm_gr.v1.worker import gr_execution, npu_beam_stage_runner

    failure = ValueError("injected execute failure")
    worker = SimpleNamespace(device="npu", execute_model_state=None)
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.previous = worker, None
    stage.closed, stage.pending_plan, stage.execution_error = False, None, None
    worker._gr_npu_stage_runner = stage
    monkeypatch.setattr(npu_beam_stage_runner, "get_npu_stage_runner", lambda *args: stage)
    scheduled = _eager_stage_output(kind="prefill")
    next(iter(scheduled.gr_stage_metadata.values())).is_last_stage = False
    fail = Mock(side_effect=failure)
    if failure_point == "plan":
        monkeypatch.setattr(stage, "_validate_sequence", fail)
    with pytest.raises(ValueError) as execute_error:
        gr_execution._dispatch_stage_frame(worker, scheduled, fail)
    assert execute_error.value is failure
    native_sample = Mock(side_effect=AssertionError("native sampler must not run"))

    def sample_original(*args: Any, **kwargs: Any) -> Any:
        return native_sample(*args, **kwargs)

    with pytest.raises(RuntimeError, match="execute_model failed") as sample_error:
        npu_patch.make_patched_sample_tokens(sample_original)(worker, None)
    assert sample_error.value.__cause__ is failure
    native_sample.assert_not_called()
    with pytest.raises(RuntimeError, match="owner failed"):
        stage.execute(scheduled, fail)


def test_npu_retirement_clears_failed_dispatch_state() -> None:
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.attention_batch = None
    stage.closed = False
    stage.logical_scheduler_output = object()
    stage.execution_error = ValueError("failed dispatch")
    stage._release()
    assert stage.closed
    assert stage.logical_scheduler_output is None
    assert stage.execution_error is None


def test_npu_stage_runner_receipt_preserves_owner_generation() -> None:
    scheduler_output = _eager_stage_output(
        kind="prefill",
        batch_size=1,
        beam_width=2,
    )
    metadata = next(iter(scheduler_output.gr_stage_metadata.values()))
    metadata.beam_params.temperature = 0.0
    metadata.is_last_stage = False
    metadata.output_device_state = object()
    plan = NPUEagerStagePlan.from_scheduler_output(scheduler_output)

    def select(logits: torch.Tensor, constraint: ConstraintMetadata) -> ConstraintOutput:
        assert torch.is_inference_mode_enabled()
        assert logits.shape[0] == 1
        assert constraint.top_k == 2
        return ConstraintOutput(
            token_ids=torch.tensor([[11, 12]], dtype=torch.int32),
            logprobs=torch.tensor([[-0.1, -0.2]], dtype=torch.float32),
            valid_mask=torch.ones((1, 2), dtype=torch.bool),
        )

    worker = SimpleNamespace(
        _current_scheduler_output=scheduler_output,
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(max_rows=1),
            select=select,
        ),
        execute_model_state=object(),
        kv_connector_output=object(),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker = worker
    stage.execution_error = None
    stage.pending_plan = plan
    stage.logits = torch.zeros((1, 16), dtype=torch.float32)
    stage.prefill_candidates = None
    stage.latest_token_ids = None
    stage.attention_batch = None
    stage.closed = False

    output = stage.sample()

    result = output.gr_worker_results[plan.execution.session_id]
    assert result.owner_generation == plan.execution.owner_generation == 1


pytestmark = pytest.mark.cpu_test
