# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Runner-local stage identity, operator selection, fallback and boundary regressions."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.platforms.binding import (
    BackendBinding,
    close_backend_binding,
    validate_worker_readiness,
    verify_engine_backend_readiness,
)
from vllm_gr.platforms.plan import (
    BACKEND_PLAN_KEY,
    BackendPlan,
    BackendRequest,
    PlanData,
)
from vllm_gr.v1.beam.decision_ops import DecisionOps
from vllm_gr.v1.worker.runner_lifecycle import (
    bind_native_runner,
    get_stage_execution,
    install_runner_lifecycle,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _plan() -> BackendPlan:
    empty = PlanData({})
    request = BackendRequest(empty, empty, empty, empty)
    return BackendPlan(
        "external",
        2,
        request,
        PlanData(
            {
                "runner": "example.runner",
                "stage": "example.stage",
                "operator_policy": "example.choose",
            }
        ),
    )


def test_stage_identity_lifetime_and_worker_agreement() -> None:
    services = []

    def factory(*, plan: BackendPlan | None, runner: Any) -> Any:
        stage = SimpleNamespace(
            implementation_id="example.stage",
            decision=SimpleNamespace(implementation_id=runner.operator),
            close=Mock(),
        )
        services.append(stage)
        return stage

    class Runner:
        def __init__(self, operator: str) -> None:
            self.operator = operator

        def execute_model(self) -> None:
            pass

        def prepare_inputs(self) -> None:
            pass

    install_runner_lifecycle(
        Runner, "example.runner", prepare="prepare_inputs", stage_factory=factory
    )
    first, second = Runner("standard"), Runner("vendor")
    unplanned = get_stage_execution(first)
    assert get_stage_execution(first) is unplanned
    close_backend_binding(first)
    services[0].close.assert_called_once()
    plan = _plan()
    reports = []
    for runner in (first, second):
        adapter = bind_native_runner(plan, runner)
        binding = BackendBinding(plan, adapter, None, None)
        setattr(runner, "_gr_backend_binding", binding)
        assert get_stage_execution(runner) is adapter.stage
        reports.append(binding.report())
    assert reports[0]["services"]["stage"] == "example.stage"
    assert reports[0]["services"]["operator"] == "standard"
    assert reports[1]["services"]["operator"] == "vendor"
    with pytest.raises(RuntimeError, match="disagree"):
        validate_worker_readiness(plan, reports, expected_workers=2)
    close_backend_binding(first)
    close_backend_binding(first)
    services[1].close.assert_called_once()
    services[2].close.assert_not_called()
    assert get_stage_execution(second) is services[2]


def test_operator_selection_strict_failure_and_fallback_preserve_input_order() -> None:
    tokens = SimpleNamespace(shape=(4, 4))
    scores, probs, history = object(), object(), object()
    arguments = (scores, tokens, probs, history, 1, 4)
    group = Mock(return_value=("ids", "parents", "scores", "counts", "history"))
    final = Mock(return_value=("terminal", "parents", "scores", "history"))
    fallback = DecisionOps("reference", group, final, 2)
    failure = RuntimeError("fused op failed")
    fused = Mock(side_effect=failure)
    ops = DecisionOps("vendor", fused, fused, 2, fallback=fallback)
    with pytest.raises(RuntimeError) as error:
        ops.select(*arguments, 4, False, True)
    assert error.value is failure
    group.assert_not_called()
    assert ops.select(*arguments, 4, False, False) == (*group.return_value, 4)
    group.assert_called_once_with(*arguments)
    assert ops.select(*arguments, 2, True, False) == (
        "terminal",
        "parents",
        "scores",
        None,
        "history",
        2,
    )
    final.assert_called_once_with(*arguments[:-1], 2)


@pytest.mark.parametrize(
    ("execution", "mode", "catalog", "decision_available", "expected"),
    [
        ("v1", "xllm", None, False, "torch.reference"),
        ("v1", "cascade", None, True, "ascend.xllm"),
        ("legacy", "xllm", "/catalog.json", False, "torch.reference"),
        ("legacy", "xllm", None, False, None),
    ],
)  # type: ignore[misc]
def test_ascend_binds_only_the_operators_required_at_startup(
    execution: str,
    mode: str,
    catalog: str | None,
    decision_available: bool,
    expected: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.platforms.ascend import stage as ascend_stage

    ops = SimpleNamespace(
        is_available=lambda: mode == "xllm",
        decision_ops_available=lambda: decision_available,
        beam_search_group=Mock(),
        beam_search_rec_final_select=Mock(),
    )
    monkeypatch.setitem(sys.modules, "vllm_gr.ops.npu", SimpleNamespace(beam_ops=ops))
    monkeypatch.setattr(
        ascend_stage, "reference_ops", lambda: DecisionOps("torch.reference", Mock(), Mock(), 32)
    )
    empty = PlanData({})
    request = BackendRequest(
        PlanData(
            {
                "beam_execution_mode": execution,
                "beam_worker_decision": True,
                "beam_max_width": 32,
                "catalog_path": catalog,
                "constraint_backend": "trie",
            }
        ),
        empty,
        empty,
        empty,
    )
    plan = BackendPlan("ascend", 2, request, PlanData({"decode_attention": mode}))
    if expected is None:
        with pytest.raises(RuntimeError, match="worker decision operators"):
            ascend_stage.create_stage(plan=plan, runner=SimpleNamespace())
    else:
        stage = ascend_stage.create_stage(plan=plan, runner=SimpleNamespace())
        assert stage.decision.implementation_id == expected
        assert (stage.decision.group is ops.beam_search_group) is decision_available


@pytest.mark.parametrize(
    ("operator", "steps", "widths", "per_parent", "return_width", "accepted"),
    [
        ("torch.reference", 3, None, None, 32, False),
        ("torch.reference", 3, (32, 32, 32), None, 32, False),
        ("ascend.xllm", 3, None, None, 32, True),
        ("torch.reference", 1, None, None, 32, True),
        ("torch.reference", 3, (32, 16, 8), None, 8, True),
        ("torch.reference", 3, None, (32, 16, 8), 32, True),
        ("torch.reference", 3, None, None, 1, True),
    ],
)  # type: ignore[misc]
def test_ascend_request_admission_uses_worker_operator_agreement(
    operator: str,
    steps: int,
    widths: tuple[int, ...] | None,
    per_parent: tuple[int, ...] | None,
    return_width: int,
    accepted: bool,
) -> None:
    from vllm_gr.v1.engine.npu_admission import validate_npu_device_config

    plan = BackendPlan(
        "ascend",
        2,
        _plan().request,
        PlanData({"stage": "ascend.v1", "operator_policy": "ascend.xllm-with-reference-fallback"}),
    )
    native = SimpleNamespace(additional_config={BACKEND_PLAN_KEY: plan.to_wire()})
    report = {
        "ready": True,
        "fingerprint": plan.fingerprint,
        "services": {"runner": "worker", "stage": "ascend.v1", "operator": operator},
    }
    engine = SimpleNamespace(
        vllm_config=native, model_executor=SimpleNamespace(collective_rpc=lambda _: [report])
    )
    verify_engine_backend_readiness(engine)
    gr = SimpleNamespace(
        constrained_generation=True, constraint_table=SimpleNamespace(max_top_k=32)
    )
    params = SimpleNamespace(
        beam_width=32,
        max_tokens=steps,
        dynamic_beam_widths=widths,
        per_parent_beam_widths=per_parent,
    )
    request = SimpleNamespace(
        beam_params=params, output_options=SimpleNamespace(num_return_sequences=return_width)
    )
    if accepted:
        validate_npu_device_config(native, gr, request)
    else:
        with pytest.raises(ValueError, match="fused decision operators"):
            validate_npu_device_config(native, gr, request)


@pytest.mark.parametrize(
    "filename", ["worker/gr_execution.py", "worker/gr_startup.py", "beam/beam_decision.py"]
)  # type: ignore[misc]
def test_shared_dispatch_does_not_select_hardware(filename: str) -> None:
    root = Path(__file__).resolve().parents[2] / "vllm_gr" / "v1"
    tree = ast.parse((root / filename).read_text(encoding="utf-8"))
    migrated = {
        "execute_gr_stage",
        "execute_gr_retirement",
        "capture_async_forward",
        "_capture_enabled",
        "run_beam_decision",
        "run_beam_decision_candidates",
        "worker_decision_enabled",
        "_run_decision",
        "_finalize_decision",
    }
    for definition in ast.walk(tree):
        if not isinstance(definition, ast.FunctionDef) or definition.name not in migrated:
            continue
        for node in ast.walk(definition):
            if isinstance(node, ast.Constant):
                assert node.value not in ("cuda", "npu", "ascend")
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("vllm_gr.ops.cuda", "vllm_gr.ops.npu", "vllm_gr.platforms.loader")
                )
