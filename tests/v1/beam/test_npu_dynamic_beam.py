# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-call"
"""CPU regressions for NPU schedule validation, selection, and KV ownership."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.exceptions import VLLMValidationError

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.sampling_params import BeamSearchParams
from vllm_gr.v1.beam.attention_pool import BeamAttentionPool
from vllm_gr.v1.beam.dynamic_width import (
    DynamicBeamState,
    get_stage_pools,
    initialize_packed_stage_pools,
    prune_stage_pools,
)
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView


@pytest.mark.parametrize(
    "widths,max_tokens,begin,end",
    [([2], 1, None, None), ([4, 2], 3, "begin", None), ([1, 2, 4], 5, "begin", "end")],
)
def test_stage_schedule_counts_only_generated_tokens(
    widths: list[int], max_tokens: int, begin: str | None, end: str | None
) -> None:
    params = BeamSearchParams(
        beam_width=4,
        max_tokens=max_tokens,
        begin_token=begin,
        end_token=end,
        dynamic_beam_widths=widths,
        result_width=1,
    )
    assert params.dynamic_beam_widths == widths
    assert params.result_width == 1


@pytest.mark.parametrize(
    "widths,result_width",
    [
        ([], None),
        ([1, 2], None),
        ([1, 2, 5], None),
        ([1, 0, 4], None),
        ([True, 2, 4], None),
        ([1, 2.0, 4], None),
        ((1, 2, 4), None),
        ([4, 2, 1], 2),
        ([1, 2, 4], 0),
        ([1, 2, 4], True),
    ],
)
def test_invalid_stage_or_result_width_is_rejected(widths: Any, result_width: Any) -> None:
    with pytest.raises(VLLMValidationError, match="dynamic_beam_widths|result_width"):
        BeamSearchParams(
            beam_width=4,
            max_tokens=3,
            dynamic_beam_widths=widths,
            result_width=result_width,
        )


def test_per_parent_beam_widths_is_validated_against_retained_widths() -> None:
    params = BeamSearchParams(
        beam_width=4,
        max_tokens=3,
        dynamic_beam_widths=[2, 4, 4],
        per_parent_beam_widths=[2, 2, 1],
        result_width=2,
    )
    assert params.per_parent_beam_widths == [2, 2, 1]

    with pytest.raises(VLLMValidationError, match="first per-parent beam width"):
        BeamSearchParams(
            beam_width=4,
            max_tokens=3,
            dynamic_beam_widths=[2, 4, 4],
            per_parent_beam_widths=[1, 2, 1],
        )

    with pytest.raises(VLLMValidationError, match=r"input_width \* per_parent_beam_width"):
        BeamSearchParams(
            beam_width=4,
            max_tokens=3,
            dynamic_beam_widths=[2, 4, 4],
            per_parent_beam_widths=[2, 1, 1],
        )

    with pytest.raises(VLLMValidationError, match="per_parent_beam_widths"):
        BeamSearchParams(
            beam_width=1,
            max_tokens=4,
            per_parent_beam_widths=[1, 1, 1, 1],
        )


@pytest.mark.parametrize("widths", [(4, 2, 1), (1, 2, 4), (4, 4, 4), None])
def test_schedule_geometry_uses_previous_width_for_each_request(
    widths: tuple[int, ...] | None,
) -> None:
    schedule = widths or (4, 4, 4)
    previous = None
    for stage in range(3):
        view = BeamBatchExecutionView(
            session_id="session",
            dispatch_id=stage,
            stage_index=stage,
            kind="prefill" if stage == 0 else "decode",
            decode_step=None if stage == 0 else stage - 1,
            owner_generation=0,
            request_ids=("first", "second"),
            num_prompt_tokens_by_request=(128, 128),
            beam_width=4,
            produces_output=True,
            dynamic_beam_widths=widths,
        )
        input_width = 1 if stage == 0 else schedule[stage - 1]
        assert view.input_width == input_width
        assert view.output_width == view.candidate_top_k == schedule[stage]
        assert view.num_execution_rows == 2 * input_width
        assert view.row_request_ids == ("first",) * input_width + ("second",) * input_width
        if previous is not None:
            previous.assert_same_logical_batch(view)
            parents = (0,) * view.output_width + (input_width,) * view.output_width
            assert view.localize_global_parent_rows(parents) == (0,) * (2 * view.output_width)
            with pytest.raises(ValueError, match="crosses"):
                view.localize_global_parent_rows((input_width,) + parents[1:])
        previous = view


def test_per_parent_beam_widths_controls_per_parent_candidate_width() -> None:
    for stage, expected in enumerate((2, 2, 1)):
        view = BeamBatchExecutionView(
            session_id="session",
            dispatch_id=stage,
            stage_index=stage,
            kind="prefill" if stage == 0 else "decode",
            decode_step=None if stage == 0 else stage - 1,
            owner_generation=0,
            request_ids=("request",),
            num_prompt_tokens_by_request=(128,),
            beam_width=4,
            produces_output=True,
            dynamic_beam_widths=(2, 4, 4),
            per_parent_beam_widths=(2, 2, 1),
        )
        assert view.candidate_top_k == expected


def test_candidate_top_k_clamps_to_output_width() -> None:
    """A parent cannot contribute more survivors than the stage keeps."""

    view = BeamBatchExecutionView(
        session_id="session",
        dispatch_id=1,
        stage_index=1,
        kind="decode",
        decode_step=0,
        owner_generation=0,
        request_ids=("request",),
        num_prompt_tokens_by_request=(128,),
        beam_width=4,
        produces_output=True,
        dynamic_beam_widths=(4, 1, 1),
        per_parent_beam_widths=(4, 4, 4),
    )
    assert view.output_width == 1
    assert view.candidate_top_k == 1


def _context(packed: bool) -> Any:
    context = SimpleNamespace(
        max_batch=3,
        max_beams=4,
        max_decode_steps=3,
        registry=SimpleNamespace(num_active_sessions=0),
        pools=[
            BeamAttentionPool(3, 4, 1, 3, 2, torch.device("cpu"), torch.float16) for _ in range(2)
        ],
    )
    if packed:
        initialize_packed_stage_pools(context)
    return context


def _candidates(scores: list[list[float]], *, first_token: int = 11) -> ConstraintOutput:
    logprobs = torch.tensor(scores, dtype=torch.float32)
    return ConstraintOutput(
        torch.arange(logprobs.numel(), dtype=torch.int32).reshape(logprobs.shape) + first_token,
        logprobs,
        torch.ones_like(logprobs, dtype=torch.bool),
    )


@pytest.mark.parametrize("packed", [False, True])
def test_idle_pruning_releases_unused_banks_and_retains_captured_addresses(packed: bool) -> None:
    context = _context(packed)
    get_stage_pools(context, 1)
    captured = get_stage_pools(context, 2)
    context._npu_beam_graph_captured = True
    retained = [*captured, *context.pools]
    addresses = [
        (pool.unshared_key.data_ptr(), pool.unshared_value.data_ptr()) for pool in retained
    ]
    if packed:
        for width in (1, 2):
            context._npu_packed_reorder_staging[width] = torch.empty_like(
                context._npu_packed_width_pools[width]
            )

    # The caller keeps width 2 because a captured graph references its bank.
    assert prune_stage_pools(context, keep_widths=(2,)) == (1,)
    assert set(context._npu_beam_width_pools) == {2}
    assert get_stage_pools(context, 2) is captured
    assert addresses == [
        (pool.unshared_key.data_ptr(), pool.unshared_value.data_ptr())
        for pool in [*get_stage_pools(context, 2), *context.pools]
    ]
    if packed:
        assert set(context._npu_packed_width_pools) == {2, 4}
        assert set(context._npu_packed_reorder_staging) == {2}


def test_stage_banks_cannot_be_pruned_during_active_session() -> None:
    context = _context(True)
    bank = get_stage_pools(context, 1)
    context.registry.num_active_sessions = 1

    with pytest.raises(RuntimeError, match="while Beam sessions are active"):
        prune_stage_pools(context, keep_widths=())

    assert get_stage_pools(context, 1) is bank
    assert 1 in context._npu_packed_width_pools


def test_late_packing_is_rejected_after_graph_capture() -> None:
    context = _context(False)
    addresses = [
        (pool.unshared_key.data_ptr(), pool.unshared_value.data_ptr()) for pool in context.pools
    ]
    context._npu_beam_graph_captured = True

    with pytest.raises(RuntimeError, match="after Beam graph capture"):
        initialize_packed_stage_pools(context)

    assert not hasattr(context, "_npu_packed_width_pools")
    assert addresses == [
        (pool.unshared_key.data_ptr(), pool.unshared_value.data_ptr()) for pool in context.pools
    ]


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -float("inf")])
def test_prefill_nonfinite_scores_and_masked_candidates_are_invalid(invalid_score: float) -> None:
    state = DynamicBeamState(_context(False), batch_size=1, max_decode_steps=3)
    prefill = _candidates([[invalid_score, -2.0, 99.0]])
    prefill.valid_mask[0, 2] = False
    state.initialize(prefill, width=3)

    assert state.scores is not None and state.sequence is not None
    assert state.scores.tolist() == [[-float("inf"), -2.0, -float("inf")]]
    assert state.sequence[:, :, 0].tolist() == [[0, 12, 0]]


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_candidate_cannot_displace_finite_winner(invalid_score: float) -> None:
    state = DynamicBeamState(_context(False), batch_size=1, max_decode_steps=3)
    state.initialize(_candidates([[-1.0, -2.0]]), width=2)
    actual = state.select(
        _candidates([[invalid_score], [-0.5]], first_token=21),
        input_width=2,
        output_width=1,
        stage_index=1,
        is_last=True,
    )

    assert actual["out_log_probs"].tolist() == [[-2.5]]
    assert actual["out_sequence"].tolist() == [[[12, 22]]]
    assert actual["local_parent_ids"].tolist() == [[1]]


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_parent_is_rejected_even_after_initialization(invalid_score: float) -> None:
    state = DynamicBeamState(_context(False), batch_size=1, max_decode_steps=3)
    state.initialize(_candidates([[-1.0, -2.0]]), width=2)
    assert state.scores is not None
    # Exercise select's own guard independently of Prefill sanitization.
    state.scores[0, 0] = invalid_score
    actual = state.select(
        _candidates([[-0.25], [-0.5]], first_token=21),
        input_width=2,
        output_width=1,
        stage_index=1,
        is_last=True,
    )

    assert actual["out_log_probs"].tolist() == [[-2.5]]
    assert actual["out_sequence"].tolist() == [[[12, 22]]]


@pytest.mark.parametrize("large_score", [3e38, -3e38])
def test_finite_fp32_accumulation_overflow_cannot_displace_finite_winner(
    large_score: float,
) -> None:
    state = DynamicBeamState(_context(False), batch_size=1, max_decode_steps=3)
    state.initialize(_candidates([[large_score, -2.0]]), width=2)
    actual = state.select(
        _candidates([[large_score], [-0.5]], first_token=21),
        input_width=2,
        output_width=1,
        stage_index=1,
        is_last=True,
    )

    assert actual["out_log_probs"].tolist() == [[-2.5]]
    assert actual["out_sequence"].tolist() == [[[12, 22]]]


@pytest.mark.parametrize("is_last", [False, True])
def test_all_invalid_candidates_produce_negative_infinity_and_zero_tokens(is_last: bool) -> None:
    state = DynamicBeamState(_context(False), batch_size=1, max_decode_steps=3)
    state.initialize(_candidates([[-1.0, -2.0]]), width=2)
    candidates = _candidates([[float("nan"), float("inf")], [-float("inf"), 99.0]])
    candidates.valid_mask[1, 1] = False
    actual = state.select(candidates, 2, 2, 1, is_last=is_last)

    assert torch.isneginf(actual["out_log_probs"]).all()
    assert actual["out_token_ids"].tolist() == [[0, 0]]
    assert actual["out_sequence"][:, :, -1].tolist() == [[0, 0]]
    assert state.scores is not None and torch.isneginf(state.scores).all()
    if not is_last:
        # Subsequent finite children cannot revive an invalid parent.
        final = state.select(_candidates([[-0.1], [-0.2]]), 2, 1, 2, is_last=True)
        assert torch.isneginf(final["out_log_probs"]).all()
        assert final["out_token_ids"].tolist() == [[0]]


def test_nonfinite_selection_remains_request_local() -> None:
    state = DynamicBeamState(_context(False), batch_size=2, max_decode_steps=3)
    state.initialize(_candidates([[-1.0, -2.0], [-3.0, -4.0]]), width=2)
    actual = state.select(
        _candidates([[float("nan")], [-0.5], [-0.25], [float("inf")]], first_token=21),
        input_width=2,
        output_width=1,
        stage_index=1,
        is_last=True,
    )

    assert actual["out_log_probs"].tolist() == [[-2.5], [-3.25]]
    assert actual["out_sequence"].tolist() == [[[12, 22]], [[13, 23]]]
    assert actual["out_token_index"].tolist() == [[1], [2]]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("widths", [(4, 2, 1), (1, 2, 4), (4, 4, 4)])
def test_stage_selection_matches_independent_beam_reference_and_preserves_request_kv(
    packed: bool, widths: tuple[int, int, int]
) -> None:
    context = _context(packed)
    state = DynamicBeamState(context, batch_size=2, max_decode_steps=3)
    ids = torch.arange(2 * widths[0], dtype=torch.int32).reshape(2, widths[0])
    scores = -torch.arange(widths[0], dtype=torch.float32).expand(2, -1) * 0.7
    valid = torch.ones_like(ids, dtype=torch.bool)
    # An invalid parent with a high nominal score must never win later.
    if widths[0] > 1:
        valid[:, -1] = False
    state.initialize(ConstraintOutput(ids, scores, valid), widths[0])
    reference = [
        [
            ([int(ids[row, beam])], float(scores[row, beam]) if valid[row, beam] else -float("inf"))
            for beam in range(widths[0])
        ]
        for row in range(2)
    ]
    for stage, output_width in enumerate(widths[1:], start=1):
        input_width = widths[stage - 1]
        source = get_stage_pools(context, input_width)
        target = get_stage_pools(context, output_width)
        for layer, pool in enumerate(source):
            for row in range(3):
                for beam in range(input_width):
                    pool.unshared_key[row, beam].fill_(100 * layer + 10 * row + beam)
                    pool.unshared_value[row, beam].fill_(-100 * layer - 10 * row - beam)
        inactive = [
            (pool.unshared_key[2].clone(), pool.unshared_value[2].clone()) for pool in target
        ]
        tokens = (
            torch.arange(2 * input_width * output_width, dtype=torch.int32).reshape(
                2 * input_width, output_width
            )
            + stage * 100
        )
        local = (
            -torch.arange(input_width * output_width, dtype=torch.float32).reshape(
                input_width, output_width
            )
            * 0.013
        )
        local = local.repeat(2, 1)
        candidate_valid = torch.ones_like(tokens, dtype=torch.bool)
        if output_width > 1 and input_width > 1:
            candidate_valid[-1, -1] = False
        candidates = ConstraintOutput(tokens, local, candidate_valid)
        expected = []
        expected_parents = []
        for row, beams in enumerate(reference):
            choices = []
            for parent, (history, previous_score) in enumerate(beams):
                index = row * input_width + parent
                for candidate in range(output_width):
                    score = previous_score + float(local[index, candidate])
                    if not candidate_valid[index, candidate]:
                        score = -float("inf")
                    choices.append((history + [int(tokens[index, candidate])], score, parent))
            selected = sorted(choices, key=lambda item: item[1], reverse=True)[:output_width]
            if stage != 2:
                selected.sort(key=lambda item: item[2])
            expected.append([(history, score) for history, score, _ in selected])
            expected_parents.append([parent for _, _, parent in selected])
        returned_width = 1 if stage == 2 else output_width
        actual = state.select(
            candidates,
            input_width,
            output_width,
            stage,
            is_last=stage == 2,
            result_width=returned_width,
        )
        assert state.current_width == output_width
        assert state.sequence is not None and state.sequence.shape == (2, output_width, 3)
        assert state.sequence[:, :, : stage + 1].tolist() == [
            [history for history, _ in row] for row in expected
        ]
        for row in range(2):
            assert actual["out_sequence"][row].tolist() == [
                history for history, _ in expected[row][:returned_width]
            ]
            assert actual["out_log_probs"][row].tolist() == pytest.approx(
                [score for _, score in expected[row][:returned_width]], abs=1e-6
            )
            assert actual["out_token_index"][row].tolist() == [
                parent + row * input_width for parent in expected_parents[row][:returned_width]
            ]
            if stage != 2:
                for layer, pool in enumerate(target):
                    for beam, parent in enumerate(expected_parents[row]):
                        assert torch.all(
                            pool.unshared_key[row, beam, :, :stage]
                            == 100 * layer + 10 * row + parent
                        )
                        assert torch.all(
                            pool.unshared_value[row, beam, :, :stage]
                            == -100 * layer - 10 * row - parent
                        )
        for pool, (key, value) in zip(target, inactive, strict=True):
            assert torch.equal(pool.unshared_key[2], key)
            assert torch.equal(pool.unshared_value[2], value)
        reference = expected


pytestmark = pytest.mark.cpu_test
