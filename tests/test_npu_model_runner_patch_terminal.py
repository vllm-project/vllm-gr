# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts for NPU terminal result and candidate handling."""

import struct
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests._npu_model_runner_patch_helpers import (
    _eager_stage_output,
    _parent_history_request,
    _terminal_result_runner,
)
from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.v1.beam.execution_batch import BeamExecutionContractError
from vllm_gr.v1.beam.npu_execution_batch import (
    NPUBeamExecutionBatch,
    restore_npu_cache_pool_layout,
)
from vllm_gr.v1.worker.npu_beam_stage_adapter import (
    NPUEagerStagePlan,
)
from vllm_gr.v1.worker.npu_beam_stage_runner import NPUEagerBeamStageRunner


@pytest.mark.parametrize(  # type: ignore[misc]
    ("candidate_top_k", "candidate_shape", "error_type", "message"),
    [
        (3, (2, 3), BeamExecutionContractError, "Prefill winner_token_ids must have shape (2, 2)"),
        (1, (2, 1), BeamExecutionContractError, "Prefill winner_token_ids must have shape (2, 2)"),
        (3, (2, 2), RuntimeError, "NPU candidate selection returned an invalid candidate shape"),
        (2, (1, 2), RuntimeError, "NPU candidate selection returned an invalid candidate shape"),
    ],
)
def test_npu_prefill_sample_preserves_candidate_and_winner_shape_errors(
    candidate_top_k: int,
    candidate_shape: tuple[int, int],
    error_type: type[Exception],
    message: str,
) -> None:
    scheduled = _eager_stage_output(kind="prefill", beam_width=2 if candidate_top_k == 1 else 4)
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    metadata.beam_params.temperature = 0.0
    # A dynamic schedule rejects K < W before sampling; exercise that case
    # through fixed-width execution so it reaches the winner-shape check.
    if candidate_top_k != 1:
        metadata.beam_params.dynamic_beam_widths = (2, 2, 2)
    metadata.beam_params.candidate_width = candidate_top_k
    metadata.is_last_stage = False
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    candidates = ConstraintOutput(
        token_ids=torch.ones(candidate_shape, dtype=torch.int32),
        logprobs=torch.zeros(candidate_shape),
        valid_mask=torch.ones(candidate_shape, dtype=torch.bool),
    )

    def select(logits: torch.Tensor, constraint: ConstraintMetadata) -> ConstraintOutput:
        assert constraint.current_step == 0 and constraint.top_k == candidate_top_k
        return candidates

    worker = SimpleNamespace(
        _current_scheduler_output=scheduled,
        constraint_backend=SimpleNamespace(info=SimpleNamespace(max_rows=2), select=select),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.pending_plan, stage.execution_error = worker, plan, None
    previous_candidates = object()
    previous_token_ids = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    logits = torch.zeros((2, 16))
    stage.prefill_candidates = previous_candidates
    stage.latest_token_ids = previous_token_ids
    stage.logits = logits

    with pytest.raises(error_type) as error:
        stage.sample()

    assert error.type is error_type
    assert str(error.value) == message

    if error_type is BeamExecutionContractError:
        assert stage.prefill_candidates is candidates
    else:
        assert stage.prefill_candidates is previous_candidates
    assert stage.latest_token_ids is previous_token_ids
    assert stage.pending_plan is plan
    assert stage.logits is logits


def test_npu_decode_sample_accepts_candidate_width_distinct_from_winner_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import npu_beam_stage_runner

    scheduled = _eager_stage_output(kind="decode", batch_size=1, beam_width=2)
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    metadata.beam_params.temperature = 0.0
    metadata.beam_params.candidate_width = 3
    metadata.is_last_stage = False
    metadata.output_device_state = None
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    candidates = ConstraintOutput(
        token_ids=torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32),
        logprobs=torch.zeros((2, 3)),
        valid_mask=torch.ones((2, 3), dtype=torch.bool),
    )
    winners = torch.tensor([[11, 21]], dtype=torch.int32)

    def select(logits: torch.Tensor, constraint: ConstraintMetadata) -> ConstraintOutput:
        assert constraint.current_step == 1 and constraint.top_k == 3
        return candidates

    def decide(
        worker: Any,
        attention_batch: Any,
        token_ids: torch.Tensor,
        logprobs: torch.Tensor,
        *,
        is_last_step: bool,
        result_width: int,
    ) -> dict[str, torch.Tensor]:
        assert token_ids is candidates.token_ids
        assert logprobs.shape == (2, 3)
        assert not is_last_step and result_width == 2
        return {"out_token_ids": winners}

    monkeypatch.setattr(npu_beam_stage_runner, "run_beam_decision_candidates", decide)
    worker = SimpleNamespace(
        _current_scheduler_output=scheduled,
        constraint_backend=SimpleNamespace(info=SimpleNamespace(max_rows=2), select=select),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.pending_plan, stage.execution_error = worker, plan, None
    stage.logits = torch.zeros((2, 16))
    stage.session_request = SimpleNamespace(
        output_options=SimpleNamespace(include_parent_beam_indices=False)
    )
    stage.attention_batch = SimpleNamespace(
        context=SimpleNamespace(decision=SimpleNamespace(sequence=torch.zeros((1, 2, 1))))
    )

    output = stage.sample()

    assert stage.latest_token_ids is winners
    assert output.gr_worker_results[plan.execution.session_id].stage_index == 1


def test_npu_execution_batch_flattens_only_the_xllm_attention_view() -> None:
    execution = NPUEagerStagePlan.from_scheduler_output(
        _eager_stage_output(kind="decode")
    ).execution
    source = SimpleNamespace(
        unshared_key=torch.arange(6).view(2, 3, 1),
        unshared_value=torch.arange(6, 12).view(2, 3, 1),
        max_batch=2,
        max_beams=3,
    )
    context = SimpleNamespace(
        pools=[source],
        max_beams=3,
        max_decode_steps=2,
        suffix_k_buf=torch.zeros(12, 1),
        suffix_v_buf=torch.zeros(12, 1),
    )
    sessions = [
        SimpleNamespace(
            session_id=request_id,
            slot=index,
            context=context,
            beam_width=3,
            decode_step=0,
            is_active=True,
            _state=object(),
        )
        for index, request_id in enumerate(execution.request_ids)
    ]

    batch = NPUBeamExecutionBatch(execution, sessions)
    flat = batch.pools[0].unshared_key

    assert flat.shape == (1, 6, 1)
    assert flat.data_ptr() == source.unshared_key.data_ptr()
    restored = restore_npu_cache_pool_layout(flat, request_width=3)
    assert restored.shape == (2, 3, 1)
    assert restored.data_ptr() == source.unshared_key.data_ptr()


def test_npu_terminal_result_keeps_batch_identity_and_output_options() -> None:
    request_ids = ("request-0", "request-1")
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.pending_plan = SimpleNamespace(execution=SimpleNamespace(request_ids=request_ids))
    stage.session_request = SimpleNamespace(
        batch_id="batch-0",
        items=tuple(
            SimpleNamespace(
                item_index=index,
                request=SimpleNamespace(request_id=request_id, prompt_token_ids=[1, 2]),
            )
            for index, request_id in enumerate(request_ids)
        ),
        beam_params=SimpleNamespace(
            stop_token_ids=(99,), ignore_stop_tokens=False, length_penalty=1.0
        ),
        output_options=SimpleNamespace(
            num_return_sequences=1,
            include_stop_token=False,
            include_cumulative_logprob=True,
            include_parent_beam_indices=True,
        ),
    )
    result = stage._build_terminal_result(
        torch.tensor([[[10, 99], [20, 21]], [[30, 31], [40, 41]]], dtype=torch.int32),
        torch.tensor([[-0.1, -1.0], [-0.2, -2.0]], dtype=torch.float32),
        torch.tensor([[[0, 0], [0, 1]], [[0, 0], [0, 1]]], dtype=torch.int32),
    )

    assert result.failure is None and result.output is not None
    assert tuple(item.request_id for item in result.output.items) == request_ids
    first = result.output.items[0].sequences[0]
    assert (first.token_ids, first.parent_beam_indices) == ((10,), (0,))
    assert (first.finish_reason, first.stop_token_id) == ("stop", 99)


@pytest.mark.parametrize(  # type: ignore[misc]
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.npu_test)]
)
@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("result_width", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "case", ["nan", "posinf", "neginf", "masked", "finite", "unsorted", "all_invalid"]
)
def test_npu_single_step_filters_candidates_before_return_limit(
    device: str, batch_size: int, result_width: int, case: str
) -> None:
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("Ascend NPU is unavailable")
    scheduled = _eager_stage_output(kind="prefill", batch_size=batch_size, beam_width=2)
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    metadata.beam_params.temperature = 1.0
    metadata.beam_params.dynamic_beam_widths = (2,)
    metadata.beam_params.max_tokens = 1
    metadata.is_last_stage = True
    metadata.output_device_state = None
    metadata.output_options = SimpleNamespace(
        num_return_sequences=result_width,
        include_stop_token=False,
        include_cumulative_logprob=True,
        include_parent_beam_indices=True,
    )
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    first_score = {
        "nan": float("nan"),
        "posinf": float("inf"),
        "neginf": -float("inf"),
        "masked": 0.0,
        "finite": -0.1,
        "unsorted": -0.9,
        "all_invalid": -0.1,
    }[case]
    token_ids = torch.tensor(
        [[11 + row * 10, 12 + row * 10] for row in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )
    scores = torch.tensor([[first_score, -0.2]] * batch_size, device=device)
    valid_mask = torch.ones_like(scores, dtype=torch.bool)
    if case == "masked":
        valid_mask[:, 0] = False
    elif case == "all_invalid":
        scores[-1] = torch.tensor([float("nan"), float("inf")], device=device)
    candidates = ConstraintOutput(token_ids=token_ids, logprobs=scores, valid_mask=valid_mask)

    def select(logits: torch.Tensor, constraint: ConstraintMetadata) -> ConstraintOutput:
        assert logits.device.type == device
        assert constraint.current_step == 0 and constraint.top_k == 2
        return candidates

    worker = SimpleNamespace(
        _current_scheduler_output=scheduled,
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(max_rows=batch_size), select=select
        ),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.pending_plan, stage.execution_error = worker, plan, None
    stage.logits = torch.zeros((batch_size, 16), device=device)
    stage.attention_batch = None
    stage.session_request = SimpleNamespace(
        batch_id="batch-0",
        items=tuple(
            SimpleNamespace(
                item_index=row,
                request=SimpleNamespace(request_id=request_id, prompt_token_ids=[1, 2]),
            )
            for row, request_id in enumerate(plan.execution.request_ids)
        ),
        beam_params=SimpleNamespace(stop_token_ids=(), ignore_stop_tokens=True, length_penalty=0.0),
        output_options=metadata.output_options,
    )
    output = stage.sample()
    receipt = output.gr_worker_results[plan.execution.session_id]
    result = receipt.terminal_result
    assert receipt.finished and receipt.resources_released
    assert stage.closed and stage.pending_plan is None
    if case == "all_invalid":
        assert result.output is None and result.failure.code == "no_valid_beams"
        assert result.failure.item_index == batch_size - 1
        return
    assert result.failure is None
    assert tuple(item.request_id for item in result.output.items) == plan.execution.request_ids
    expected = [0, 1] if case == "finite" else [1, 0] if case == "unsorted" else [1]
    for row, item in enumerate(result.output.items):
        selected = expected[:result_width]
        assert [seq.token_ids for seq in item.sequences] == [
            (11 + row * 10 + index,) for index in selected
        ]
        for sequence, index in zip(item.sequences, selected):
            assert sequence.cumulative_logprob == pytest.approx(first_score if index == 0 else -0.2)
            assert sequence.parent_beam_indices == (0,)
            assert sequence.finish_reason == "length" and sequence.stop_token_id is None


@pytest.mark.parametrize(  # type: ignore[misc]
    "batch,steps,include_score,include_parents",
    [
        (1, 1, True, False),
        (1, 2, False, True),
        (1, 3, True, True),
        (4, 1, False, False),
        (4, 2, True, False),
        (4, 3, True, True),
    ],
)
def test_npu_terminal_fixed_lengths_preserve_rows_scores_and_output_flags(
    batch: int, steps: int, include_score: bool, include_parents: bool
) -> None:
    stage = _terminal_result_runner(
        batch, include_score=include_score, include_parents=include_parents
    )
    tokens = torch.tensor(
        [
            [[100 * row + 10 * beam + step for step in range(steps)] for beam in range(3)]
            for row in range(batch)
        ],
        dtype=torch.int32,
    )
    parents = torch.tensor([[[0, 1, 2][:steps]] * 3] * batch, dtype=torch.int32)
    scores = torch.tensor([[-0.25, -0.1, -0.2]] * batch, dtype=torch.float32)
    original_tokens, original_parents = tokens.clone(), parents.clone()

    result = stage._build_terminal_result(tokens, scores, parents)

    assert result.batch_id == "terminal-batch" and result.failure is None
    assert result.output is not None and len(result.output.items) == batch
    for row, output in enumerate(result.output.items):
        assert (output.item_index, output.request_id) == (row, f"request-{row}")
        assert [sequence.token_ids for sequence in output.sequences] == [
            tuple(100 * row + 10 * beam + step for step in range(steps)) for beam in (1, 2)
        ]
        for sequence, score_bits in zip(output.sequences, ("bdcccccd", "be4ccccd"), strict=True):
            if include_score:
                assert struct.pack(">f", sequence.cumulative_logprob).hex() == score_bits
            else:
                assert sequence.cumulative_logprob is None
            assert sequence.parent_beam_indices == ((0, 1, 2)[:steps] if include_parents else ())
            assert sequence.finish_reason == "length" and sequence.stop_token_id is None
    assert torch.equal(tokens, original_tokens) and torch.equal(parents, original_parents)


def test_npu_terminal_filters_nonfinite_before_limits_independently_per_request() -> None:
    stage = _terminal_result_runner(4, include_score=False, include_parents=False)
    tokens = torch.arange(20, dtype=torch.int32).view(4, 5, 1)
    scores = torch.tensor(
        [
            [float("nan"), float("inf"), -float("inf"), -0.25, -0.125],
            [-0.1, float("nan"), float("inf"), -float("inf"), float("nan")],
            [-float("inf"), -float("inf"), -float("inf"), -0.125, -0.125],
            [-0.3, float("nan"), float("inf"), -float("inf"), -0.2],
        ],
        dtype=torch.float32,
    )

    result = stage._build_terminal_result(tokens, scores, torch.zeros_like(tokens))

    assert result.failure is None and result.output is not None
    assert [[seq.token_ids for seq in item.sequences] for item in result.output.items] == [
        [(4,), (3,)],
        [(5,)],
        [(13,), (14,)],
        [(19,), (15,)],
    ]
    assert all(
        seq.cumulative_logprob is None for item in result.output.items for seq in item.sequences
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "penalty,include_stop,order",
    [
        (1.0, True, [2, 0, 1]),
        (1.0, False, [2, 0, 1]),
        (0.0, False, [2, 0, 1]),
        (-1.0, True, [0, 2, 1]),
    ],
)
def test_npu_terminal_stop_lengths_rank_with_prompt_and_keep_parent_alignment(
    penalty: float, include_stop: bool, order: list[int]
) -> None:
    stage = _terminal_result_runner(
        1, returned=3, stop_ids=(99, 98), penalty=penalty, include_stop=include_stop
    )
    tokens = torch.tensor([[[99, 97, 99], [20, 21, 22], [30, 99, 98]]], dtype=torch.int32)
    parents = torch.tensor([[[0, 1, 2], [0, 2, 1], [0, 1, 0]]], dtype=torch.int32)
    result = stage._build_terminal_result(
        tokens, torch.tensor([[-0.5, -1.0, -0.375]], dtype=torch.float32), parents
    )

    assert result.failure is None and result.output is not None
    expected_tokens = [
        (99,) if include_stop else (),
        (20, 21, 22),
        (30, 99) if include_stop else (30,),
    ]
    expected_parents = [(0,) if include_stop else (), (0, 2, 1), (0, 1) if include_stop else (0,)]
    expected_scores = [-0.5, -1.0, -0.375]
    for sequence, beam in zip(result.output.items[0].sequences, order, strict=True):
        assert sequence.token_ids == expected_tokens[beam]
        assert sequence.parent_beam_indices == expected_parents[beam]
        assert sequence.cumulative_logprob == expected_scores[beam]
        assert sequence.finish_reason == ("length" if beam == 1 else "stop")
        assert sequence.stop_token_id == (None if beam == 1 else 99)


def test_npu_terminal_ignored_stops_keep_full_history_and_same_parent_payload() -> None:
    stage = _terminal_result_runner(1, stop_ids=(99, 98), ignore_stops=True, include_stop=False)
    tokens = torch.tensor([[[99, 97, 99], [20, 21, 22], [30, 99, 98]]], dtype=torch.int32)
    parents = torch.tensor([[[0, 1, 2], [0, 2, 1], [0, 1, 0]]], dtype=torch.int32)
    scores = torch.tensor([[-0.5, -1.0, -0.375]], dtype=torch.float32)
    included = stage._build_terminal_result(tokens, scores, parents)
    stage.session_request.output_options.include_parent_beam_indices = False
    omitted = stage._build_terminal_result(tokens, scores, parents)

    assert included.output is not None and omitted.output is not None
    for left, right, expected_tokens, expected_parents in zip(
        included.output.items[0].sequences,
        omitted.output.items[0].sequences,
        [(30, 99, 98), (99, 97, 99)],
        [(0, 1, 0), (0, 1, 2)],
        strict=True,
    ):
        assert left.token_ids == right.token_ids == expected_tokens
        assert left.parent_beam_indices == expected_parents and right.parent_beam_indices == ()
        assert left.cumulative_logprob == right.cumulative_logprob
        assert left.finish_reason == right.finish_reason == "length"
        assert left.stop_token_id is right.stop_token_id is None


@pytest.mark.parametrize(  # type: ignore[misc]
    "raw_scores,penalty,expected_tokens,expected_bits",
    [
        ([-0.0, 0.0, -0.5], 1.0, [10, 20, 30], ["80000000", "00000000", "bf000000"]),
        (
            [-8 * 2.0**-149, -(2.0**-149), -(2.0**-149)],
            1023.0,
            [20, 30, 10],
            ["80000001", "80000001", "80000008"],
        ),
    ],
)
def test_npu_terminal_ties_keep_raw_score_then_original_beam_order(
    raw_scores: list[float], penalty: float, expected_tokens: list[int], expected_bits: list[str]
) -> None:
    stage = _terminal_result_runner(1, returned=3, penalty=penalty)
    tokens = torch.tensor([[[10], [20], [30]]], dtype=torch.int32)
    result = stage._build_terminal_result(
        tokens, torch.tensor([raw_scores], dtype=torch.float32), torch.zeros_like(tokens)
    )

    assert result.output is not None and result.failure is None
    sequences = result.output.items[0].sequences
    assert [sequence.token_ids for sequence in sequences] == [(token,) for token in expected_tokens]
    assert [
        struct.pack(">f", sequence.cumulative_logprob).hex() for sequence in sequences
    ] == expected_bits


@pytest.mark.parametrize("width", [0, 3])  # type: ignore[misc]
def test_npu_terminal_empty_or_nonfinite_rows_do_not_evaluate_overflowing_penalty(
    width: int,
) -> None:
    stage = _terminal_result_runner(1, penalty=1e6)
    tokens = torch.zeros((1, width, 3), dtype=torch.int32)
    scores = torch.tensor([[float("nan"), float("inf"), -float("inf")]], dtype=torch.float32)[
        :, :width
    ]

    result = stage._build_terminal_result(tokens, scores, torch.zeros_like(tokens))

    assert result.output is None and result.failure is not None
    assert (result.failure.code, result.failure.item_index) == ("no_valid_beams", 0)


def test_npu_terminal_invalid_second_request_returns_batch_failure_without_partial_output() -> None:
    stage = _terminal_result_runner(4)
    tokens = torch.arange(8, dtype=torch.int32).view(4, 2, 1)
    scores = torch.tensor(
        [[-0.1, -0.2], [float("nan"), float("inf")], [-0.3, -0.4], [-0.5, -0.6]],
        dtype=torch.float32,
    )

    result = stage._build_terminal_result(tokens, scores, torch.zeros_like(tokens))

    assert result.batch_id == "terminal-batch" and result.output is None
    assert result.failure is not None
    assert (result.failure.code, result.failure.message, result.failure.item_index) == (
        "no_valid_beams",
        "No valid final Beam candidates",
        1,
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "invalid,message",
    [
        ("session", "missing session_init"),
        ("order", "request order changed"),
        ("scores", "inconsistent shapes"),
        ("parents", "inconsistent shapes"),
        ("parent_depth", "shape"),
    ],
)
def test_npu_terminal_rejects_existing_session_order_and_tensor_errors(
    invalid: str, message: str
) -> None:
    stage = _terminal_result_runner(1, include_parents=True)
    tokens = torch.zeros((1, 2, 3), dtype=torch.int32)
    scores = torch.zeros((1, 2), dtype=torch.float32)
    parents = torch.zeros_like(tokens)
    if invalid == "session":
        stage.session_request = None
    elif invalid == "order":
        stage.pending_plan.execution.request_ids = ("different-request",)
    elif invalid == "scores":
        scores = scores[:, :1]
    elif invalid == "parents":
        parents = parents[:, :1]
    else:
        parents = parents[:, :, :2]

    with pytest.raises(RuntimeError, match=message):
        stage._build_terminal_result(tokens, scores, parents)


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("steps", [1, 2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("include_parents", [False, True])  # type: ignore[misc]
def test_npu_parent_history_follows_output_options_through_session(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, steps: int, include_parents: bool
) -> None:
    """Exercise the real decision, including duplicate parents and request offsets."""
    from unittest.mock import Mock

    from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext

    context = BeamAttentionContext(batch_size, 3, 1, 3, 2, 1, torch.device("cpu"), torch.float16)
    sequence_pointer = context.decision.sequence.data_ptr()
    scores_pointer = context.decision.beam_scores.data_ptr()
    request = _parent_history_request(batch_size, include_parents)
    request.beam_params.max_tokens = steps
    token_tables = (
        torch.tensor([[10, 20, 30]], dtype=torch.int32),
        torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=torch.int32),
        torch.tensor([[41, 42, 43], [51, 52, 53], [61, 62, 63]], dtype=torch.int32),
    )
    score_tables = (
        torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32),
        torch.tensor([[-0.8, -1.0, -1.2], [-0.01, -0.02, -2.0], [-0.03, -2.0, -3.0]]),
        torch.tensor([[-0.5, -2.0, -3.0], [-0.01, -0.02, -3.0], [-0.01, -2.0, -3.0]]),
    )

    def select(logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        index = metadata.current_step
        tokens = torch.cat([token_tables[index] + row * 100 for row in range(batch_size)])
        return ConstraintOutput(
            token_ids=tokens,
            logprobs=score_tables[index].repeat(batch_size, 1),
            valid_mask=torch.ones_like(tokens, dtype=torch.bool),
        )

    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _vllm_gr_config=SimpleNamespace(constrained_generation=True),
        _gr_npu_graph_context=context,
        _gr_npu_beam_graph_shapes={(batch_size, 3)},
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(max_rows=batch_size * 3), select=select
        ),
        execute_model_state=None,
        kv_connector_output=None,
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.session_request = worker, request
    stage.execution_error, stage.attention_batch, stage.parent_history = None, None, None
    stage.closed = False
    advance = Mock(wraps=stage._advance_parent_history)
    if not include_parents:
        advance.side_effect = AssertionError("unrequested parent history must not be advanced")
    monkeypatch.setattr(stage, "_advance_parent_history", advance)
    zeros_like = Mock(wraps=torch.zeros_like)
    monkeypatch.setattr(torch, "zeros_like", zeros_like)
    zeros = Mock(wraps=torch.zeros)
    monkeypatch.setattr(torch, "zeros", zeros)
    terminal = None
    for index in range(steps):
        scheduled = _eager_stage_output(
            kind="prefill" if index == 0 else "decode",
            batch_size=batch_size,
            beam_width=3,
            decode_step=max(index - 1, 0),
        )
        metadata = next(iter(scheduled.gr_stage_metadata.values()))
        metadata.beam_params = request.beam_params
        metadata.is_last_stage = index == steps - 1
        metadata.output_device_state = None
        worker._current_scheduler_output = scheduled
        stage.pending_plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
        if index:
            stage._ensure_attention_batch(stage.pending_plan)
            assert (stage.parent_history is not None) is include_parents
            assert stage.attention_batch is not None
            for session in stage.attention_batch.sessions:
                session._state.is_last_step = metadata.is_last_stage
        rows = batch_size if index == 0 else batch_size * 3
        stage.logits = torch.zeros(rows, 128)
        output = stage.sample()
        terminal = output.gr_worker_results["batch-0"].terminal_result
        assert stage.closed is (index == steps - 1)

    assert terminal is not None and terminal.output is not None and terminal.failure is None
    expected_tokens = (
        ((10,), (20,), (30,)),
        ((20, 21), (20, 22), (30, 31)),
        ((20, 22, 51), (20, 22, 52), (30, 31, 61)),
    )[steps - 1]
    expected_parents = (
        ((0,), (0,), (0,)),
        ((0, 1), (0, 1), (0, 2)),
        ((0, 1, 1), (0, 1, 1), (0, 2, 2)),
    )[steps - 1]
    expected_scores = (
        score_tables[0][0],
        torch.stack(
            (
                score_tables[0][0, 1] + score_tables[1][1, 0],
                score_tables[0][0, 1] + score_tables[1][1, 1],
                score_tables[0][0, 2] + score_tables[1][2, 0],
            )
        ),
        torch.tensor([-0.2, -0.2, -0.3])
        + torch.tensor([-0.02, -0.02, -0.03])
        + torch.tensor([-0.01, -0.02, -0.01]),
    )[steps - 1]
    for row, item in enumerate(terminal.output.items):
        assert item.request_id == f"request-{row}"
        assert tuple(sequence.token_ids for sequence in item.sequences) == tuple(
            tuple(token + row * 100 for token in tokens) for tokens in expected_tokens
        )
        assert tuple(sequence.parent_beam_indices for sequence in item.sequences) == (
            expected_parents if include_parents else ((), (), ())
        )
        assert tuple(sequence.cumulative_logprob for sequence in item.sequences) == tuple(
            expected_scores.tolist()
        )
        assert all(sequence.finish_reason == "length" for sequence in item.sequences)
    assert advance.call_count == (steps - 1 if include_parents else 0)
    history_allocations = [
        call for call in zeros_like.call_args_list if call.args[0].data_ptr() == sequence_pointer
    ]
    assert len(history_allocations) == int(include_parents and steps > 1)
    prefill_history_allocations = [
        call for call in zeros.call_args_list if call.args == ((batch_size, 3, 1),)
    ]
    assert len(prefill_history_allocations) == int(include_parents and steps == 1)
    assert stage.parent_history is None
    assert context.registry.num_active_sessions == 0
    assert not getattr(worker, "_beam_sessions", {})
    assert context.decision.sequence.data_ptr() == sequence_pointer
    assert context.decision.beam_scores.data_ptr() == scores_pointer


@pytest.mark.parametrize("include_parents", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("include_stop", [False, True])  # type: ignore[misc]
def test_npu_terminal_parent_transfer_and_stop_alignment(
    monkeypatch: pytest.MonkeyPatch, include_parents: bool, include_stop: bool
) -> None:
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.session_request = _parent_history_request(1, include_parents)
    stage.session_request.beam_params.stop_token_ids = (99,)
    stage.session_request.output_options.include_stop_token = include_stop
    stage.pending_plan = SimpleNamespace(execution=SimpleNamespace(request_ids=("request-0",)))
    cat, tolist = torch.cat, torch.Tensor.tolist
    pieces: list[tuple[int, ...]] = []
    converted: list[tuple[int, ...]] = []

    def record_cat(tensors: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        pieces.append(tuple(tensor.numel() for tensor in tensors))
        return cat(tensors, *args, **kwargs)

    def record_tolist(tensor: torch.Tensor) -> Any:
        converted.append(tuple(tensor.shape))
        return tolist(tensor)

    monkeypatch.setattr(torch, "cat", record_cat)
    monkeypatch.setattr(torch.Tensor, "tolist", record_tolist)
    result = stage._build_terminal_result(
        torch.tensor([[[10, 99, 12], [20, 21, 22], [30, 31, 32]]], dtype=torch.int32),
        torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32),
        torch.tensor([[[0, 2, 1], [0, 0, 2], [0, 1, 0]]], dtype=torch.int32)
        if include_parents
        else None,
    )
    assert result.output is not None
    first = result.output.items[0].sequences[0]
    assert first.token_ids == ((10, 99) if include_stop else (10,))
    assert first.parent_beam_indices == (
        ((0, 2) if include_stop else (0,)) if include_parents else ()
    )
    assert (first.finish_reason, first.stop_token_id) == ("stop", 99)
    expected_pieces: list[tuple[int, ...]] = [(9, 9, 3)] if include_parents else [(9, 3)]
    expected_converted: list[tuple[int, ...]] = (
        [(1, 3, 3), (1, 3, 3), (1, 3)] if include_parents else [(1, 3, 3), (1, 3)]
    )
    assert pieces == expected_pieces
    assert converted == expected_converted


def test_npu_missing_parent_history_is_rejected_only_when_requested() -> None:
    stage = object.__new__(NPUEagerBeamStageRunner)
    assert stage._needs_parent_history()
    stage.session_request = None
    assert stage._needs_parent_history()
    stage.session_request = _parent_history_request(1, True)
    stage.pending_plan = SimpleNamespace(execution=SimpleNamespace(request_ids=("request-0",)))
    sequences = torch.ones(1, 3, 2, dtype=torch.int32)
    scores = torch.zeros(1, 3)
    with pytest.raises(RuntimeError, match="missing requested parent history"):
        stage._build_terminal_result(sequences, scores, None)
    with pytest.raises(RuntimeError, match="inconsistent shapes"):
        stage._build_terminal_result(sequences, scores, torch.zeros(1, 3, dtype=torch.int32))
    stage.session_request.output_options.include_parent_beam_indices = False
    # A supplied legacy history is ignored entirely when not requested.
    result = stage._build_terminal_result(sequences, scores, object())
    assert result.output is not None
    assert all(not sequence.parent_beam_indices for sequence in result.output.items[0].sequences)


@pytest.mark.parametrize("widths", [(3, 2, 1), (1, 2, 3), None])  # type: ignore[misc]
@pytest.mark.parametrize("include_parents", [False, True])  # type: ignore[misc]
def test_npu_dynamic_parent_history_is_not_advanced_twice(
    monkeypatch: pytest.MonkeyPatch,
    widths: tuple[int, ...] | None,
    include_parents: bool,
) -> None:
    """Dynamic decisions own lineage even when only the return width differs."""
    from unittest.mock import Mock

    request = _parent_history_request(2, include_parents)
    request.beam_params.dynamic_beam_widths = widths
    request.beam_params.ignore_stop_tokens = True
    result_width = 1 if widths == (3, 2, 1) else 2
    request.output_options.num_return_sequences = result_width
    scheduled = _eager_stage_output(kind="decode", batch_size=2, beam_width=3, decode_step=1)
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    metadata.beam_params = request.beam_params
    metadata.output_options = request.output_options
    metadata.is_last_stage, metadata.output_device_state = True, None
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    execution = plan.execution
    candidate_shape = (execution.num_execution_rows, execution.candidate_top_k)
    candidates = ConstraintOutput(
        token_ids=torch.zeros(candidate_shape, dtype=torch.int32),
        logprobs=torch.zeros(candidate_shape),
        valid_mask=torch.ones(candidate_shape, dtype=torch.bool),
    )
    sequences = torch.tensor(
        [[[10, 11, 12], [20, 21, 22]], [[110, 111, 112], [120, 121, 122]]],
        dtype=torch.int32,
    )[:, :result_width]
    parents = torch.tensor([[[0, 0, 1], [0, 0, 0]]] * 2, dtype=torch.int32)[:, :result_width]
    scores = torch.tensor([[-0.25, -0.5]] * 2)[:, :result_width]
    decision = {
        "out_token_ids": sequences[:, :, -1],
        "out_sequence": sequences,
        "out_log_probs": scores,
        "out_parent_history": parents,
    }
    dynamic_select = Mock(return_value=decision)
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.session_request, stage.pending_plan, stage.execution_error = request, plan, None
    stage.parent_history = None
    stage.dynamic_state = SimpleNamespace(
        select=dynamic_select,
        sequence=torch.zeros(2, execution.input_width, 3, dtype=torch.int32),
    )
    stage.attention_batch = SimpleNamespace(
        sessions=tuple(SimpleNamespace(session_id=f"request-{row}") for row in range(2)),
        end_session=Mock(),
    )
    stage.worker = SimpleNamespace(
        _current_scheduler_output=scheduled,
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(max_rows=execution.num_execution_rows),
            select=Mock(return_value=candidates),
        ),
    )
    stage.logits = torch.zeros(execution.num_execution_rows, 128)
    native = Mock(side_effect=AssertionError("dynamic lineage must not use the native decision"))
    advance = Mock(side_effect=AssertionError("dynamic lineage must not be advanced twice"))
    monkeypatch.setattr(
        "vllm_gr.v1.worker.npu_beam_stage_runner.run_beam_decision_candidates", native
    )
    monkeypatch.setattr(stage, "_advance_parent_history", advance)
    build_terminal = stage._build_terminal_result

    def check_terminal_history(*args: Any) -> Any:
        # Inspect before terminal cleanup clears the session's history.
        assert (stage.parent_history is not None) is include_parents
        if include_parents:
            assert stage.parent_history is parents
        return build_terminal(*args)

    terminal = Mock(side_effect=check_terminal_history)
    monkeypatch.setattr(stage, "_build_terminal_result", terminal)

    assert stage._uses_dynamic_state(plan)
    result = stage.sample().gr_worker_results["batch-0"].terminal_result

    dynamic_select.assert_called_once_with(
        candidates,
        execution.input_width,
        execution.output_width,
        2,
        is_last=True,
        result_width=result_width,
    )
    native.assert_not_called()
    advance.assert_not_called()
    terminal.assert_called_once()
    if include_parents:
        torch.testing.assert_close(terminal.call_args.args[2], parents, rtol=0, atol=0)
    else:
        assert terminal.call_args.args[2] is None
    assert result is not None and result.failure is None and result.output is not None
    for row, item in enumerate(result.output.items):
        assert item.request_id == f"request-{row}"
        assert tuple(sequence.token_ids for sequence in item.sequences) == tuple(
            tuple(tokens) for tokens in sequences[row].tolist()
        )
        assert tuple(sequence.cumulative_logprob for sequence in item.sequences) == tuple(
            scores[row].tolist()
        )
        assert tuple(sequence.parent_beam_indices for sequence in item.sequences) == (
            tuple(tuple(history) for history in parents[row].tolist())
            if include_parents
            else ((),) * result_width
        )
        assert all(
            (sequence.finish_reason, sequence.stop_token_id) == ("length", None)
            for sequence in item.sequences
        )
    assert stage.closed and stage.parent_history is None
    stage.attention_batch.end_session.assert_called_once()


@pytest.mark.parametrize(  # type: ignore[misc]
    ("rows", "top_k"), [(4, 128), (128, 128), (512, 128), (512, 64)]
)
@pytest.mark.parametrize("scalar_temperature", [False, True])  # type: ignore[misc]
def test_npu_constraint_candidates_single_call_preserves_tensors(
    rows: int, top_k: int, scalar_temperature: bool
) -> None:
    """B4 Decode and smaller stage widths use one call without repacking."""
    logits = torch.arange(rows, dtype=torch.float32)[:, None].expand(-1, 128).contiguous()
    metadata = ConstraintMetadata(
        prefix_token_ids=torch.arange(rows * 3, dtype=torch.int32).view(rows, 3),
        temperatures=torch.ones(1 if scalar_temperature else rows, dtype=torch.float32),
        current_step=0 if rows == 4 else 2,
        top_k=top_k,
    )
    expected = ConstraintOutput(
        token_ids=torch.arange(rows * top_k, dtype=torch.int32).view(rows, top_k),
        logprobs=torch.zeros(rows, top_k),
        valid_mask=torch.ones(rows, top_k, dtype=torch.bool),
    )
    calls = 0

    def select(
        actual_logits: torch.Tensor, actual_metadata: ConstraintMetadata
    ) -> ConstraintOutput:
        nonlocal calls
        calls += 1
        assert actual_logits is logits
        assert actual_metadata is metadata
        return expected

    stage = object.__new__(NPUEagerBeamStageRunner)
    result = stage._select_constraint_candidates(
        SimpleNamespace(info=SimpleNamespace(max_rows=512), select=select), logits, metadata
    )

    assert calls == 1
    assert result is expected


@pytest.mark.parametrize("rows,max_rows", [(6, 3), (512, 128), (512, 192)])  # type: ignore[misc]
@pytest.mark.parametrize("scalar_temperature", [False, True])  # type: ignore[misc]
def test_npu_constraint_candidates_chunk_multi_batch_rows(
    rows: int, max_rows: int, scalar_temperature: bool
) -> None:
    calls: list[ConstraintMetadata] = []

    def select(logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        calls.append(metadata)
        row_ids = logits[:, 0].to(torch.int32)
        return ConstraintOutput(
            token_ids=row_ids[:, None].expand(-1, metadata.top_k).contiguous(),
            logprobs=logits[:, : metadata.top_k].contiguous(),
            valid_mask=torch.ones((logits.shape[0], metadata.top_k), dtype=torch.bool),
        )

    top_k = 2 if rows == 6 else 128
    prefixes = torch.arange(rows * 3, dtype=torch.int32).view(rows, 3)
    temperatures = torch.arange(1 if scalar_temperature else rows, dtype=torch.float32)
    metadata = ConstraintMetadata(
        prefix_token_ids=prefixes,
        temperatures=temperatures,
        current_step=1,
        top_k=top_k,
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    result = stage._select_constraint_candidates(
        SimpleNamespace(info=SimpleNamespace(max_rows=max_rows), select=select),
        torch.arange(rows, dtype=torch.float32)[:, None].expand(-1, max(top_k, 4)),
        metadata,
    )

    assert [item.prefix_token_ids.shape[0] for item in calls] == [
        min(max_rows, rows - start) for start in range(0, rows, max_rows)
    ]
    torch.testing.assert_close(torch.cat([item.prefix_token_ids for item in calls]), prefixes)
    if scalar_temperature:
        assert all(item.temperatures is temperatures for item in calls)
    else:
        torch.testing.assert_close(torch.cat([item.temperatures for item in calls]), temperatures)
    assert all(item.current_step == 1 and item.top_k == top_k for item in calls)
    torch.testing.assert_close(result.token_ids[:, 0], torch.arange(rows, dtype=torch.int32))
    assert result.logprobs.is_contiguous() and result.valid_mask.all()


@pytest.mark.parametrize("ignore_stop_tokens", [False, True])  # type: ignore[misc]
def test_npu_dynamic_beam_admission_requires_fixed_horizon(ignore_stop_tokens: bool) -> None:
    from vllm_gr.v1.beam.batch_contracts import BeamBatchParams
    from vllm_gr.v1.engine.beam_search_v1 import _validate_gr_generation_params

    params = BeamBatchParams(
        beam_width=4,
        max_tokens=3,
        temperature=0.0,
        stop_token_ids=(99,),
        ignore_stop_tokens=ignore_stop_tokens,
        dynamic_beam_widths=(4, 2, 1),
    )
    if ignore_stop_tokens:
        _validate_gr_generation_params(params)
    else:
        with pytest.raises(ValueError, match="requires ignore_eos=True"):
            _validate_gr_generation_params(params)


pytestmark = pytest.mark.cpu_test
