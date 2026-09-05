# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the isolated NPU persistent Beam-decision helper.

These tests inject CPU golden operators. They do not exercise an NPU,
vllm-ascend ModelRunner integration, attention metadata, or ACL Graph replay.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _beam_search_rec_final_select_torch,
)
from vllm_gr.v1.beam.beam_attention_context import BeamDecisionBuffers
from vllm_gr.v1.worker import npu_gr_decision


class GoldenOps:
    """CPU implementation of the xllm decision ABI, with call evidence."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.group_output: tuple[torch.Tensor, ...] | None = None
        self.final_output: tuple[torch.Tensor, ...] | None = None

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def unavailable_reason() -> str:
        return ""

    def beam_search_group(
        self,
        log_probs: torch.Tensor,
        top_tokens: torch.Tensor,
        top_probs: torch.Tensor,
        sequence: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append("group")
        self.group_output = _beam_search_group_torch(
            log_probs,
            top_tokens,
            top_probs,
            sequence,
            current_step,
            top_k,
        )
        return self.group_output

    def beam_search_rec_final_select(
        self,
        log_probs: torch.Tensor,
        top_tokens: torch.Tensor,
        top_probs: torch.Tensor,
        sequence: torch.Tensor,
        current_step: int,
        result_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append("final")
        self.final_output = _beam_search_rec_final_select_torch(
            log_probs,
            top_tokens,
            top_probs,
            sequence,
            current_step,
            result_width,
        )
        return self.final_output


def _session(width: int, steps: int = 3) -> Any:
    decision = BeamDecisionBuffers(
        max_batch=1,
        max_beams=width,
        max_decode_steps=steps,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    def forbidden_step_complete(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("decision helper must leave KV reorder to its caller")

    return SimpleNamespace(
        beam_width=width,
        slot=0,
        decode_step=0,
        is_active=True,
        context=SimpleNamespace(decision=decision),
        step_complete=forbidden_step_complete,
    )


def _bootstrap(session: Any, ops: GoldenOps) -> tuple[torch.Tensor, torch.Tensor, Any]:
    width = session.beam_width
    tokens = torch.arange(10, 10 + width, dtype=torch.int32).reshape(1, width)
    scores = -torch.arange(width, dtype=torch.float32).reshape(1, width) / 10
    step = npu_gr_decision.bootstrap_from_constrained_prefill(
        session,
        tokens,
        scores,
        expected_width=width,
        ops=ops,
    )
    return tokens, scores, step


def test_bootstrap_writes_and_returns_canonical_device_state() -> None:
    session = _session(4)
    ops = GoldenOps()
    tokens, scores, step = _bootstrap(session, ops)
    decision = session.context.decision

    assert step.device_step == 1
    assert step.token_ids.data_ptr() == decision.sequence.data_ptr()
    assert step.parent_ids.dtype is torch.int32
    assert torch.equal(step.parent_ids, torch.zeros_like(tokens))
    assert step.sequence.data_ptr() == decision.sequence.data_ptr()
    assert step.scores.data_ptr() == decision.beam_scores.data_ptr()
    assert torch.equal(step.sequence[:, :, 0], tokens)
    assert torch.count_nonzero(step.sequence[:, :, 1:]) == 0
    assert torch.equal(step.scores, scores)
    assert step.reorder_parents is None
    assert step.reorder_prefix is None
    assert decision.worker_decode_steps == 1
    assert ops.calls == []


def test_duplicate_bootstrap_is_rejected_without_overwriting_canonical_state() -> None:
    session = _session(8)
    ops = GoldenOps()
    _bootstrap(session, ops)
    before_sequence = session.context.decision.sequence.clone()
    before_scores = session.context.decision.beam_scores.clone()
    with pytest.raises(RuntimeError, match="step zero"):
        npu_gr_decision.bootstrap_from_constrained_prefill(
            session,
            torch.full((1, 8), 999, dtype=torch.int32),
            torch.full((1, 8), -999.0, dtype=torch.float32),
            expected_width=8,
            ops=ops,
        )
    assert torch.equal(session.context.decision.sequence, before_sequence)
    assert torch.equal(session.context.decision.beam_scores, before_scores)
    assert session.context.decision.worker_decode_steps == 1


def test_nonfinal_advance_uses_stable_tie_golden_and_defers_kv_reorder() -> None:
    width = 8
    session = _session(width)
    ops = GoldenOps()
    _bootstrap(session, ops)
    # All cumulative candidates tie. The independent Torch golden therefore
    # fixes selection to lower logical flat indices before parent grouping.
    session.context.decision.beam_scores.zero_()
    candidates = torch.arange(100, 100 + width * width, dtype=torch.int32).reshape(width, width)
    candidate_scores = torch.zeros((width, width), dtype=torch.float32)

    step = npu_gr_decision.advance_fused_decision(
        session,
        candidates,
        candidate_scores,
        expected_device_step=1,
        is_last_step=False,
        result_width=width,
        ops=ops,
    )
    assert ops.calls == ["group"]
    assert ops.group_output is not None
    out_tokens, out_parents, out_scores, out_prefix, out_sequence = ops.group_output
    assert step.device_step == 2
    assert step.token_ids is out_tokens
    assert step.parent_ids is out_parents
    assert step.reorder_parents is out_parents
    assert step.reorder_prefix is out_prefix
    assert out_tokens[0].tolist() == list(range(100, 100 + width))
    assert out_parents[0].tolist() == [0] * width
    decision = session.context.decision
    assert step.sequence.data_ptr() == decision.sequence.data_ptr()
    assert step.scores.data_ptr() == decision.beam_scores.data_ptr()
    assert torch.equal(step.sequence, out_sequence)
    assert torch.equal(step.scores, out_scores)
    assert decision.worker_decode_steps == 2
    assert session.decode_step == 0


def test_three_stage_flow_requires_caller_to_consume_reorder_and_advance_session() -> None:
    width = 32
    session = _session(width)
    ops = GoldenOps()
    _bootstrap(session, ops)
    session.context.decision.beam_scores.zero_()

    first_tokens = torch.arange(100, 100 + width * width, dtype=torch.int32).reshape(width, width)
    tied_scores = torch.zeros((width, width), dtype=torch.float32)
    middle = npu_gr_decision.advance_fused_decision(
        session,
        first_tokens,
        tied_scores,
        expected_device_step=1,
        is_last_step=False,
        result_width=width,
        ops=ops,
    )
    assert session.decode_step == 0
    consumed_reorder = (middle.reorder_parents, middle.reorder_prefix)
    assert all(torch.is_tensor(value) for value in consumed_reorder)

    # This stands in for the future backend's explicit reorder + session
    # lifecycle advance. The decision helper must not perform either action.
    session.decode_step += 1
    final_tokens = torch.arange(1000, 1000 + width * width, dtype=torch.int32).reshape(width, width)
    final = npu_gr_decision.advance_fused_decision(
        session,
        final_tokens,
        tied_scores,
        expected_device_step=2,
        is_last_step=True,
        result_width=width,
        ops=ops,
    )
    assert ops.calls == ["group", "final"]
    assert final.device_step == 3
    assert final.reorder_parents is None
    assert final.reorder_prefix is None
    assert session.decode_step == 1
    assert session.context.decision.worker_decode_steps == 3
    assert final.sequence.shape == (1, width, 3)


def test_final_advance_normalizes_flat_xllm_outputs_without_reorder() -> None:
    width = 32
    session = _session(width)
    ops = GoldenOps()
    _bootstrap(session, ops)
    session.context.decision.beam_scores.zero_()
    candidates = torch.arange(width * width, dtype=torch.int32).reshape(width, width)
    candidate_scores = torch.zeros((width, width), dtype=torch.float32)

    step = npu_gr_decision.advance_fused_decision(
        session,
        candidates,
        candidate_scores,
        expected_device_step=1,
        is_last_step=True,
        result_width=width,
        ops=ops,
    )
    assert ops.calls == ["final"]
    assert ops.final_output is not None
    out_tokens, out_parents, out_scores, out_sequence = ops.final_output
    assert step.token_ids.shape == (1, width)
    assert step.parent_ids.shape == (1, width)
    assert step.token_ids.data_ptr() == out_tokens.data_ptr()
    assert step.parent_ids.data_ptr() == out_parents.data_ptr()
    assert step.reorder_parents is None
    assert step.reorder_prefix is None
    assert step.token_ids[0].tolist() == list(range(width))
    assert step.parent_ids[0].tolist() == [0] * width
    decision = session.context.decision
    assert step.sequence.data_ptr() == decision.sequence.data_ptr()
    assert step.scores.data_ptr() == decision.beam_scores.data_ptr()
    assert torch.equal(step.sequence, out_sequence)
    assert torch.equal(step.scores, out_scores.reshape(1, width))
    assert decision.worker_decode_steps == 2


class ExpectedXllmError(RuntimeError):
    pass


def test_xllm_exception_propagates_without_torch_fallback_or_state_commit() -> None:
    session = _session(8)
    ops = GoldenOps()
    _bootstrap(session, ops)
    before_sequence = session.context.decision.sequence.clone()
    before_scores = session.context.decision.beam_scores.clone()

    expected_error = ExpectedXllmError("xllm failed")

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise expected_error

    ops.beam_search_group = fail  # type: ignore[method-assign]
    with pytest.raises(ExpectedXllmError, match="xllm failed") as exc_info:
        npu_gr_decision.advance_fused_decision(
            session,
            torch.zeros((8, 8), dtype=torch.int32),
            torch.zeros((8, 8), dtype=torch.float32),
            expected_device_step=1,
            is_last_step=False,
            result_width=8,
            ops=ops,
        )
    assert exc_info.value is expected_error
    assert torch.equal(session.context.decision.sequence, before_sequence)
    assert torch.equal(session.context.decision.beam_scores, before_scores)
    assert session.context.decision.worker_decode_steps == 1


@pytest.mark.parametrize(
    ("tokens", "scores", "width", "match"),
    [
        (torch.zeros(4, dtype=torch.int32), torch.zeros((1, 4)), 4, "shape"),
        (torch.zeros((1, 4), dtype=torch.int64), torch.zeros((1, 4)), 4, "int32"),
        (
            torch.zeros((1, 4), dtype=torch.int32),
            torch.zeros((1, 4), dtype=torch.float16),
            4,
            "float32",
        ),
        (torch.zeros((1, 4), dtype=torch.int32), torch.zeros((1, 4)), 3, "width"),
    ],
)
def test_bootstrap_invalid_shape_dtype_or_width_fails_before_mutation(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    width: int,
    match: str,
) -> None:
    session = _session(4)
    with pytest.raises((TypeError, ValueError, RuntimeError), match=match):
        npu_gr_decision.bootstrap_from_constrained_prefill(
            session,
            tokens,
            scores,
            expected_width=width,
            ops=GoldenOps(),
        )
    assert session.context.decision.worker_decode_steps == 0
    assert torch.count_nonzero(session.context.decision.sequence) == 0


@pytest.mark.parametrize(
    ("tokens", "scores", "step", "is_final", "result_width", "match"),
    [
        (torch.zeros((8, 7), dtype=torch.int32), torch.zeros((8, 7)), 1, False, 8, "shape"),
        (torch.zeros((8, 8), dtype=torch.int64), torch.zeros((8, 8)), 1, False, 8, "int32"),
        (
            torch.zeros((8, 8), dtype=torch.int32),
            torch.zeros((8, 8), dtype=torch.float16),
            1,
            False,
            8,
            "float32",
        ),
        (torch.zeros((8, 8), dtype=torch.int32), torch.zeros((8, 8)), 2, False, 8, "step"),
        (torch.zeros((8, 8), dtype=torch.int32), torch.zeros((8, 8)), 1, False, 7, "width"),
        (torch.zeros((8, 8), dtype=torch.int32), torch.zeros((8, 8)), 1, True, 8, "multiple"),
    ],
)
def test_advance_invalid_shape_dtype_step_width_or_final_alignment_fails_before_op(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    step: int,
    is_final: bool,
    result_width: int,
    match: str,
) -> None:
    session = _session(8)
    ops = GoldenOps()
    _bootstrap(session, ops)
    ops.calls.clear()
    with pytest.raises((TypeError, ValueError, RuntimeError), match=match):
        npu_gr_decision.advance_fused_decision(
            session,
            tokens,
            scores,
            expected_device_step=step,
            is_last_step=is_final,
            result_width=result_width,
            ops=ops,
        )
    assert ops.calls == []
    assert session.context.decision.worker_decode_steps == 1


def test_unavailable_xllm_is_strict_and_never_falls_back() -> None:
    session = _session(4)

    class UnavailableOps(GoldenOps):
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def unavailable_reason() -> str:
            return "test missing xllm"

    with pytest.raises(RuntimeError, match="test missing xllm"):
        npu_gr_decision.bootstrap_from_constrained_prefill(
            session,
            torch.zeros((1, 4), dtype=torch.int32),
            torch.zeros((1, 4), dtype=torch.float32),
            expected_width=4,
            ops=UnavailableOps(),
        )


@pytest.mark.parametrize("missing_name", ["beam_search_group", "beam_search_rec_final_select"])
def test_injected_ops_requires_both_xllm_decision_methods(
    missing_name: str,
) -> None:
    session = _session(8)
    ops = GoldenOps()
    setattr(ops, missing_name, None)
    with pytest.raises(RuntimeError, match=missing_name):
        npu_gr_decision.bootstrap_from_constrained_prefill(
            session,
            torch.zeros((1, 8), dtype=torch.int32),
            torch.zeros((1, 8), dtype=torch.float32),
            expected_width=8,
            ops=ops,
        )


def test_default_ops_checks_decision_exports_not_execution_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.ops.npu import _xllm_loader

    session = _session(8)
    monkeypatch.setattr(_xllm_loader, "custom_ops", lambda: object())
    monkeypatch.setattr(
        _xllm_loader,
        "has_op",
        lambda name: name != "beam_search_rec_final_select",
    )
    with pytest.raises(RuntimeError, match="beam_search_rec_final_select"):
        npu_gr_decision.bootstrap_from_constrained_prefill(
            session,
            torch.zeros((1, 8), dtype=torch.int32),
            torch.zeros((1, 8), dtype=torch.float32),
            expected_width=8,
        )


def test_default_ops_needs_only_group_and_final_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.ops.npu import _xllm_loader, beam_ops

    decision_exports = {"beam_search_group", "beam_search_rec_final_select"}
    monkeypatch.setattr(_xllm_loader, "custom_ops", lambda: object())
    monkeypatch.setattr(
        _xllm_loader,
        "has_op",
        lambda name: name in decision_exports,
    )
    resolved, injected = npu_gr_decision._resolve_ops(None)
    assert resolved is beam_ops
    assert injected is False


@pytest.mark.parametrize("bad_field", ["token_shape", "score_dtype"])
def test_invalid_group_output_fails_before_canonical_commit(bad_field: str) -> None:
    session = _session(8)

    class BadOutputOps(GoldenOps):
        def beam_search_group(self, *args: Any) -> tuple[torch.Tensor, ...]:
            output = list(super().beam_search_group(*args))
            if bad_field == "token_shape":
                output[0] = output[0][:, :-1]
            else:
                output[2] = output[2].to(torch.float16)
            return tuple(output)

    ops = BadOutputOps()
    _bootstrap(session, ops)
    before_sequence = session.context.decision.sequence.clone()
    before_scores = session.context.decision.beam_scores.clone()
    with pytest.raises((TypeError, ValueError), match="group"):
        npu_gr_decision.advance_fused_decision(
            session,
            torch.zeros((8, 8), dtype=torch.int32),
            torch.zeros((8, 8), dtype=torch.float32),
            expected_device_step=1,
            is_last_step=False,
            result_width=8,
            ops=ops,
        )
    assert torch.equal(session.context.decision.sequence, before_sequence)
    assert torch.equal(session.context.decision.beam_scores, before_scores)
    assert session.context.decision.worker_decode_steps == 1


def test_released_session_is_rejected_before_canonical_mutation() -> None:
    session = _session(8)
    session.is_active = False
    with pytest.raises(RuntimeError, match="active"):
        npu_gr_decision.bootstrap_from_constrained_prefill(
            session,
            torch.zeros((1, 8), dtype=torch.int32),
            torch.zeros((1, 8), dtype=torch.float32),
            expected_width=8,
            ops=GoldenOps(),
        )
    assert session.context.decision.worker_decode_steps == 0


def test_inactive_session_is_rejected_before_advance_op() -> None:
    session = _session(8)
    ops = GoldenOps()
    _bootstrap(session, ops)
    session.is_active = False
    ops.calls.clear()
    with pytest.raises(RuntimeError, match="active"):
        npu_gr_decision.advance_fused_decision(
            session,
            torch.zeros((8, 8), dtype=torch.int32),
            torch.zeros((8, 8), dtype=torch.float32),
            expected_device_step=1,
            is_last_step=False,
            result_width=8,
            ops=ops,
        )
    assert ops.calls == []
    assert session.context.decision.worker_decode_steps == 1


def test_next_advance_rejects_when_caller_did_not_advance_session_step() -> None:
    session = _session(8)
    ops = GoldenOps()
    _bootstrap(session, ops)
    candidates = torch.zeros((8, 8), dtype=torch.int32)
    scores = torch.zeros((8, 8), dtype=torch.float32)
    npu_gr_decision.advance_fused_decision(
        session,
        candidates,
        scores,
        expected_device_step=1,
        is_last_step=False,
        result_width=8,
        ops=ops,
    )
    assert session.context.decision.worker_decode_steps == 2
    assert session.decode_step == 0
    ops.calls.clear()
    with pytest.raises(RuntimeError, match="session/device step mismatch"):
        npu_gr_decision.advance_fused_decision(
            session,
            candidates,
            scores,
            expected_device_step=2,
            is_last_step=False,
            result_width=8,
            ops=ops,
        )
    assert ops.calls == []


def test_helper_has_no_host_materialization_or_hidden_kv_reorder_calls() -> None:
    for function in (
        npu_gr_decision.bootstrap_from_constrained_prefill,
        npu_gr_decision.advance_fused_decision,
    ):
        tree = ast.parse(inspect.getsource(function))
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert attributes.isdisjoint({"cpu", "tolist", "item", "numpy"})
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "reorder_unshared_kv" not in names
        assert "step_complete" not in attributes
