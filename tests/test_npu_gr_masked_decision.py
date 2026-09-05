# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the explicit mask-aware NPU device decision stage."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import torch

from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _beam_search_rec_final_select_torch,
)
from vllm_gr.v1.beam.beam_attention_context import BeamDecisionBuffers
from vllm_gr.v1.worker.npu_gr_masked_decision import NPUMaskedBeamDecision
from vllm_gr.v1.worker.npu_gr_workspace import NPUWorkspace


def _session(width: int = 4, steps: int = 3):
    decision = BeamDecisionBuffers(
        max_batch=1,
        max_beams=width,
        max_decode_steps=steps,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    decision.sequence[0, :, 0].copy_(torch.arange(10, 10 + width, dtype=torch.int32))
    decision.beam_scores.zero_()
    decision.worker_decode_steps = 1
    return SimpleNamespace(
        is_active=True,
        beam_width=width,
        slot=0,
        decode_step=0,
        context=SimpleNamespace(decision=decision),
    )


def _selector(width: int = 4, steps: int = 3) -> NPUMaskedBeamDecision:
    return NPUMaskedBeamDecision(
        beam_width=width,
        max_steps=steps,
        device="cpu",
        cpu_test_backend=True,
    )


def test_legal_negative_infinity_beats_equal_invalid_padding() -> None:
    width = 4
    session = _session(width)
    tokens = torch.zeros((width, width), dtype=torch.int32)
    tokens[:, 0].copy_(torch.arange(20, 20 + width, dtype=torch.int32))
    scores = torch.full((width, width), float("-inf"), dtype=torch.float32)
    valid = torch.zeros((width, width), dtype=torch.bool)
    valid[:, 0] = True

    result = _selector(width).select(
        session,
        tokens,
        scores,
        valid,
        expected_device_step=1,
        is_last_step=False,
        result_width=width,
    )

    assert result.token_ids.tolist() == [[20, 21, 22, 23]]
    assert result.parent_ids.tolist() == [[0, 1, 2, 3]]
    assert torch.isneginf(result.scores).all()
    assert result.parent_prefix is not None
    assert result.parent_prefix.tolist() == [[1, 2, 3, 4]]
    assert result.status.tolist() == [0]
    assert not result.active_mask.any()


def test_finite_group_matches_reference_with_stable_parent_grouping() -> None:
    width = 4
    session = _session(width)
    session.context.decision.beam_scores.copy_(
        torch.tensor([[0.0, -0.1, -0.2, -0.3]], dtype=torch.float32)
    )
    tokens = torch.arange(100, 100 + width * width, dtype=torch.int32).view(width, width)
    scores = -torch.tensor(
        [
            [0.7, 0.8, 0.9, 1.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.05, 0.6, 0.7, 0.8],
            [0.01, 0.5, 0.6, 0.7],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones_like(tokens, dtype=torch.bool)
    expected = _beam_search_group_torch(
        session.context.decision.beam_scores.view(width, 1),
        tokens,
        scores,
        session.context.decision.sequence.clone(),
        1,
        width,
    )

    result = _selector(width).select(
        session,
        tokens,
        scores,
        valid,
        expected_device_step=1,
        is_last_step=False,
        result_width=width,
    )

    torch.testing.assert_close(result.token_ids, expected[0])
    torch.testing.assert_close(result.parent_ids, expected[1])
    torch.testing.assert_close(result.scores, expected[2])
    torch.testing.assert_close(result.parent_prefix, expected[3])
    torch.testing.assert_close(result.sequence, expected[4])


def test_final_keeps_global_score_order_and_supports_unaligned_result_width() -> None:
    width = 4
    result_width = 2
    session = _session(width)
    tokens = torch.arange(100, 100 + width * width, dtype=torch.int32).view(width, width)
    scores = -torch.arange(width * width, dtype=torch.float32).view(width, width)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    expected = _beam_search_rec_final_select_torch(
        session.context.decision.beam_scores.view(width, 1),
        tokens,
        scores,
        session.context.decision.sequence.clone(),
        1,
        result_width,
    )

    result = _selector(width).select(
        session,
        tokens,
        scores,
        valid,
        expected_device_step=1,
        is_last_step=True,
        result_width=result_width,
    )

    assert result.parent_prefix is None
    torch.testing.assert_close(result.token_ids, expected[0].view(1, result_width))
    torch.testing.assert_close(result.parent_ids, expected[1].view(1, result_width))
    torch.testing.assert_close(result.scores, expected[2].view(1, result_width))
    torch.testing.assert_close(result.sequence, expected[3])


def test_insufficient_valid_candidates_emit_safe_rows_and_deferred_status() -> None:
    width = 4
    session = _session(width)
    tokens = torch.arange(100, 100 + width * width, dtype=torch.int32).view(width, width)
    scores = torch.zeros((width, width), dtype=torch.float32)
    valid = torch.zeros((width, width), dtype=torch.bool)
    valid[2, 1] = True

    result = _selector(width).select(
        session,
        tokens,
        scores,
        valid,
        expected_device_step=1,
        is_last_step=False,
        result_width=width,
    )

    assert result.status.tolist() == [1]
    assert result.token_ids.tolist() == [[0, 0, 0, 109]]
    assert result.parent_ids.tolist() == [[0, 0, 0, 2]]
    assert result.parent_prefix is not None
    assert result.parent_prefix.tolist() == [[3, 3, 4, 4]]
    assert result.active_mask.tolist() == [[False, False, False, True]]


def test_workspace_records_legal_negative_infinity_without_false_error() -> None:
    width = 4
    session = _session(width, steps=2)
    selector = _selector(width, steps=2)
    workspace = NPUWorkspace.allocate(beam_width=width, max_steps=2, device="cpu")
    lease = workspace.acquire(
        "gr",
        generation_steps=2,
        result_width=width,
        canonical_sequence=session.context.decision.sequence,
        canonical_scores=session.context.decision.beam_scores,
    )
    workspace.reset(lease)
    assert not workspace.record_bootstrap(
        lease,
        session.context.decision.beam_scores,
        torch.ones((1, width), dtype=torch.bool),
        candidate_valid_mask=torch.ones((1, width), dtype=torch.bool),
    )
    workspace.before_decision(
        lease,
        session.context.decision.beam_scores,
        expected_device_step=1,
    )
    tokens = torch.zeros((width, width), dtype=torch.int32)
    tokens[:, 0].copy_(torch.arange(20, 20 + width, dtype=torch.int32))
    local_scores = torch.full((width, width), float("-inf"), dtype=torch.float32)
    valid = torch.zeros((width, width), dtype=torch.bool)
    valid[:, 0] = True
    result = selector.select(
        session,
        tokens,
        local_scores,
        valid,
        expected_device_step=1,
        is_last_step=True,
        result_width=width,
    )

    assert workspace.record_decision(
        lease,
        result.parent_ids,
        session.context.decision.beam_scores,
        result.active_mask,
        expected_device_step=1,
        selected_valid_mask=result.valid_mask,
        selected_local_scores=result.local_scores,
    )
    assert workspace.status.tolist() == [0]
    assert torch.isneginf(workspace.cumulative_delta_history[0, :, 1]).all()
    assert not workspace.active_mask.any()


def test_outputs_reuse_fixed_storage_and_source_has_no_host_materialization() -> None:
    selector = _selector()
    pointers = None
    for offset in (0, 100):
        session = _session()
        tokens = torch.arange(offset, offset + 16, dtype=torch.int32).view(4, 4)
        scores = -torch.arange(16, dtype=torch.float32).view(4, 4)
        result = selector.select(
            session,
            tokens,
            scores,
            torch.ones_like(tokens, dtype=torch.bool),
            expected_device_step=1,
            is_last_step=False,
            result_width=4,
        )
        current = (
            result.token_ids.data_ptr(),
            result.parent_ids.data_ptr(),
            result.active_mask.data_ptr(),
            result.parent_prefix.data_ptr(),
            result.status.data_ptr(),
        )
        pointers = current if pointers is None else pointers
        assert current == pointers

    tree = ast.parse(textwrap.dedent(inspect.getsource(NPUMaskedBeamDecision.select)))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert attributes.isdisjoint({"cpu", "tolist", "item", "numpy"})
