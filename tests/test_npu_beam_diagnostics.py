# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""CPU-only checks for the opt-in E2E diagnostic observer."""

import ast
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.st.npu_beam_diagnostics import NPUForwardTrace


@pytest.fixture
def prefill_case_helpers() -> dict[str, Any]:
    # Exercise the pure case policy without importing NPU runtime dependencies.
    source = Path(__file__).parent / "st/test_onerec_v1_npu_e2e.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        "_padded_prefill_requested",
        "_graph_test_environment",
        "_comparison_prompts",
        "_assert_exact_prefill",
    }
    nodes: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {"Any": Any, "Mapping": Mapping}
    namespace["_prompt_tokens"] = lambda tokenizer, length: list(range(1000 - length, 1000))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def test_exact_prefill_inputs_and_opt_in_padding(prefill_case_helpers: dict[str, Any]) -> None:
    helpers = prefill_case_helpers
    prompts = helpers["_comparison_prompts"](None, exact_prefill=True)
    assert [len(prompt) + 1 for prompt in prompts] == [128, 128]
    assert prompts[0] != prompts[1]
    legacy = helpers["_comparison_prompts"](None, exact_prefill=False)
    assert [len(prompt) + 1 for prompt in legacy] == [129, 161]
    inherited = {
        "VLLM_ENABLE_PREFILL_CUDAGRAPH": "0",
        "VLLM_GR_PREFILL_GRAPH_BUCKET_STEP": "384",
        "VLLM_GR_PREFILL_GRAPH_MAX_BUCKET": "384",
        "unrelated": "keep",
    }
    configured = helpers["_graph_test_environment"](inherited)
    assert configured == dict(
        inherited,
        VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
        VLLM_GR_PREFILL_GRAPH_BUCKET_STEP="128",
        VLLM_GR_PREFILL_GRAPH_MAX_BUCKET="256",
    )
    assert inherited["VLLM_ENABLE_PREFILL_CUDAGRAPH"] == "0"
    for flag in ("VLLM_GR_TEST_PADDED_PREFILL",):
        diagnostic = dict(inherited, **{flag: "1"})
        assert helpers["_graph_test_environment"](diagnostic) == diagnostic


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        {"padded_tokens": 384},
        {"gr_prefill_replay_delta": {}},
        {"gr_prefill_replay_delta": {"384": 1}},
        {"skip_compiled": True},
    ],
)
def test_exact_prefill_requires_real_replay(
    prefill_case_helpers: dict[str, Any], mutation: dict[str, Any] | None
) -> None:
    prefill = dict(
        actual_tokens=256,
        padded_tokens=256,
        input_ids_shape=[256],
        positions_shape=[256],
        attention=dict(query_start_loc=[0, 128, 256], seq_lens=[128, 128]),
        enforce_eager=False,
        compilation_mode="VLLM_COMPILE",
        skip_compiled=False,
        gr_prefill_replay_delta={"256": 1},
    )
    check = prefill_case_helpers["_assert_exact_prefill"]
    if mutation:
        with pytest.raises(AssertionError):
            check(dict(prefill, **mutation), "invalid control")
    else:
        check(prefill, "compiled exact bucket")
        check(
            dict(prefill, enforce_eager=True, compilation_mode="NONE", gr_prefill_replay_delta={}),
            "eager exact input",
        )


@pytest.mark.parametrize("mode", ["NONE", "PIECEWISE", "FULL"])
@pytest.mark.parametrize("fails", [False, True])
def test_forward_probe_observes_real_arguments_and_restores(
    mode: str,
    fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Mode(Enum):
        NONE = 0
        PIECEWISE = 1
        FULL = 2
        VLLM_COMPILE = 3
        FULL_AND_PIECEWISE = 4

    calls = []
    sentinel = object()
    input_ids = torch.tensor([10, 11, 12, 13, 0, 0])
    positions = torch.tensor([0, 1, 0, 1, 0, 0])
    query_start = torch.tensor([0, 2, 4])
    context = SimpleNamespace(
        cudagraph_runtime_mode=Mode[mode],
        skip_compiled=False,
        batch_descriptor=SimpleNamespace(num_tokens=6, uniform=False),
        attn_metadata={"layer.0": SimpleNamespace(query_start_loc=query_start)},
    )

    class Runner:
        _prefill_graph_runner: Any

        def _model_forward(self, num_tokens_padded: int, input_ids: Any, positions: Any) -> Any:
            calls.append((num_tokens_padded, input_ids, positions))
            # Simulate an optional GR Prefill interceptor and reused buffers.
            self._prefill_graph_runner._replays_by_bucket[6] += 1
            input_ids.fill_(99)
            query_start.fill_(99)
            if fails:
                raise ValueError("original forward error")
            return sentinel

    runner: Any = Runner()
    runner._prefill_graph_runner = SimpleNamespace(_replays_by_bucket={6: 2})
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=Mode.VLLM_COMPILE,
            cudagraph_mode=Mode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[6],
        )
    )
    runner.model_config = SimpleNamespace(enforce_eager=False)
    runner.model = sentinel
    runner.input_batch = SimpleNamespace(req_ids=["r0", "r1"])
    runner._gr_npu_stage_plan = SimpleNamespace(
        execution=SimpleNamespace(batch_size=2, stage_index=0, kind="prefill"),
        model_token_count=4,
    )
    # Exercise both inherited methods and existing instance overrides.
    if fails:
        runner._model_forward = runner._model_forward
    original_override = vars(runner).get("_model_forward")
    probe = NPUForwardTrace(runner, lambda: context)
    try:
        with monkeypatch.context() as guard:

            def forbidden(*args: Any, **kwargs: Any) -> None:
                raise AssertionError("host transfer inside model forward")

            guard.setattr(torch.Tensor, "cpu", forbidden)
            guard.setattr(torch.Tensor, "tolist", forbidden)
            if fails:
                with pytest.raises(ValueError, match="original forward error"):
                    runner._model_forward(6, input_ids, positions=positions)
            else:
                assert runner._model_forward(6, input_ids, positions=positions) is sentinel
    finally:
        events = probe.finish()
    assert probe.finish() == []
    assert vars(runner).get("_model_forward") is original_override
    assert len(calls) == 1 and calls[0][1] is input_ids and calls[0][2] is positions
    event = events[0]
    assert event["input_ids"] == [10, 11, 12, 13, 0, 0]
    assert event["positions"] == [0, 1, 0, 1, 0, 0]
    assert event["attention"]["query_start_loc"] == [0, 2, 4]
    assert event["attention"]["slot_mapping"] is None  # Missing is not fabricated.
    assert event["runtime_mode"] == mode
    assert event["configured_graph_mode"] == "FULL_AND_PIECEWISE"
    assert event["actual_tokens"] == 4 and event["padded_tokens"] == 6
    assert event["gr_prefill_replay_delta"] == {"6": 1}
    json.dumps(events, allow_nan=False)


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("fails", [False, True])
def test_forward_probe_explicit_b1_b4_cold_metadata_is_snapshotted(
    batch_size: int, fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    total = batch_size * 128
    request_ids = tuple(f"r{row}" for row in range(batch_size))
    input_ids = torch.arange(total)
    positions = torch.arange(128).repeat(batch_size)
    query_start = torch.arange(batch_size + 1) * 128
    context = SimpleNamespace(
        cudagraph_runtime_mode="NONE",
        skip_compiled=False,
        attn_metadata={"layer.0": SimpleNamespace(query_start_loc=query_start)},
    )
    calls = []

    class Runner:
        _prefill_graph_runner: Any
        requests: dict[str, Any]
        _current_scheduler_output: Any

        def _model_forward(self, num_tokens_padded: int, input_ids: Any, positions: Any) -> int:
            calls.append(num_tokens_padded)
            self._prefill_graph_runner._replays_by_bucket[total] += 1
            input_ids.fill_(99)
            query_start.fill_(99)
            for request_id in request_ids:
                self.requests[request_id].num_computed_tokens = 99
                self.requests[request_id].prompt_token_ids.clear()
                self._current_scheduler_output.num_scheduled_tokens[request_id] = 1
            if fails:
                raise ValueError("original matrix forward error")
            return 17

    runner: Any = Runner()
    runner._prefill_graph_runner = SimpleNamespace(_replays_by_bucket={total: 2})
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode="NONE", cudagraph_mode="NONE", cudagraph_capture_sizes=[]
        )
    )
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.model = object()
    runner.input_batch = SimpleNamespace(req_ids=list(request_ids))
    runner.requests = {
        request_id: SimpleNamespace(num_computed_tokens=0, prompt_token_ids=list(range(128)))
        for request_id in request_ids
    }
    runner._current_scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request_id: 128 for request_id in request_ids},
        total_num_scheduled_tokens=total,
    )
    runner._gr_npu_stage_plan = SimpleNamespace(
        execution=SimpleNamespace(
            batch_size=batch_size, stage_index=0, kind="prefill", request_ids=request_ids
        ),
        model_token_count=total,
    )
    probe = NPUForwardTrace(
        runner, lambda: context, expected_batch_size=batch_size, record_cold=True
    )
    try:
        with monkeypatch.context() as guard:

            def forbidden(*args: Any, **kwargs: Any) -> None:
                raise AssertionError("host transfer inside model forward")

            guard.setattr(torch.Tensor, "cpu", forbidden)
            guard.setattr(torch.Tensor, "tolist", forbidden)
            if fails:
                with pytest.raises(ValueError, match="original matrix forward error"):
                    runner._model_forward(total, input_ids, positions)
            else:
                assert runner._model_forward(total, input_ids, positions) == 17
    finally:
        events = probe.finish()
    assert "_model_forward" not in vars(runner)
    assert calls == [total]
    assert events[0]["input_ids"] == list(range(total))
    assert events[0]["attention"]["query_start_loc"] == [row * 128 for row in range(batch_size + 1)]
    assert events[0]["cold_rows"] == [
        dict(num_computed_tokens=0, prompt_tokens=128, scheduled_tokens=128) for _ in request_ids
    ]
    assert events[0]["scheduled_total"] == total
    assert events[0]["gr_prefill_replay_delta"] == {str(total): 1}


@pytest.mark.parametrize("batch_size,padded", [(1, 128), (4, 512), (2, 0), (2, 513)])
def test_forward_probe_default_b2_guard_is_preserved(batch_size: int, padded: int) -> None:
    calls = []

    def original(num_tokens_padded: int) -> None:
        calls.append(num_tokens_padded)

    runner = SimpleNamespace(
        _model_forward=original,
        _gr_npu_stage_plan=SimpleNamespace(execution=SimpleNamespace(batch_size=batch_size)),
        vllm_config=SimpleNamespace(compilation_config=None),
    )
    probe = NPUForwardTrace(runner, lambda: SimpleNamespace())
    try:
        with pytest.raises(RuntimeError, match="unexpected or unbounded shape"):
            runner._model_forward(padded)
    finally:
        assert probe.finish() == []
    assert runner._model_forward is original and calls == []


@pytest.fixture
def p0_matrix_helpers() -> dict[str, Any]:
    source = Path(__file__).parent / "st/test_onerec_v1_npu_p0_matrix.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_matrix_environment", "_assert_matrix_trace", "_npu_test_unavailable"}
    nodes: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "pytest": pytest,
        "_INPUT_LEN": 128,
        "_BEAM_WIDTH": 128,
    }
    exec("import os", namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("graph", [False, True])
def test_p0_matrix_trace_requires_actual_inputs_and_replays(
    batch_size: int, graph: bool, p0_matrix_helpers: dict[str, Any]
) -> None:
    prompts = [[row] * 127 for row in range(batch_size)]
    total = batch_size * 128
    prefill = dict(
        stage=0,
        kind="prefill",
        actual_tokens=total,
        padded_tokens=total,
        input_ids_shape=[total],
        positions_shape=[total],
        input_ids=[token for prompt in prompts for token in [*prompt, 999]],
        positions=list(range(128)) * batch_size,
        input_request_ids=[f"r{row}" for row in range(batch_size)],
        attention=dict(
            query_start_loc=[row * 128 for row in range(batch_size + 1)],
            seq_lens=[128] * batch_size,
        ),
        cold_rows=[
            dict(num_computed_tokens=0, prompt_tokens=128, scheduled_tokens=128) for _ in prompts
        ],
        scheduled_total=total,
        gr_prefill_replay_delta={str(total): 1} if graph else {},
        enforce_eager=not graph,
        compilation_mode="VLLM_COMPILE" if graph else "NONE",
        runtime_mode="NONE",
        skip_compiled=False,
    )
    decode = dict(
        kind="decode",
        actual_tokens=total,
        padded_tokens=total,
        gr_prefill_replay_delta={},
        enforce_eager=not graph,
        compilation_mode="VLLM_COMPILE" if graph else "NONE",
        runtime_mode="FULL" if graph else "NONE",
    )
    events = [prefill, dict(stage=1, **decode), dict(stage=2, **decode)]
    check = p0_matrix_helpers["_assert_matrix_trace"]
    check(events, prompts, 999, graph=graph)
    for mutation in (
        {"input_ids": [0] * total},
        {"positions": [0] * total},
        {"padded_tokens": total + 128},
        {"cold_rows": [dict(num_computed_tokens=1, prompt_tokens=128, scheduled_tokens=127)]},
        {"gr_prefill_replay_delta": {"wrong-bucket": 1}},
    ):
        with pytest.raises(AssertionError):
            check([dict(prefill, **mutation), *events[1:]], prompts, 999, graph=graph)
    if graph:
        with pytest.raises(AssertionError):
            check(
                [dict(prefill, gr_prefill_replay_delta={}), *events[1:]], prompts, 999, graph=True
            )
        with pytest.raises(AssertionError):
            check(
                [prefill, dict(events[1], runtime_mode="NONE"), events[2]], prompts, 999, graph=True
            )


def test_p0_matrix_environment_is_exact_and_does_not_change_device_assignment(
    p0_matrix_helpers: dict[str, Any],
) -> None:
    inherited = {
        "ASCEND_RT_VISIBLE_DEVICES": "8",
        "VLLM_ASCEND_ENABLE_NZ": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "TORCH_MULTIPROCESSING_METHOD": "fork",
        "VLLM_ENABLE_PREFILL_CUDAGRAPH": "0",
        "VLLM_GR_PREFILL_GRAPH_BUCKETS": "8192",
        "VLLM_GR_PREFILL_GRAPH_KV_BOUND": "65536",
        "VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS": "512,4",
        "VLLM_GR_TEST_PADDED_PREFILL": "1",
    }
    result = p0_matrix_helpers["_matrix_environment"](inherited)
    assert result["ASCEND_RT_VISIBLE_DEVICES"] == "8"
    assert result["VLLM_GR_TARGET_DEVICE"] == "npu"
    assert result["TORCH_MULTIPROCESSING_METHOD"] == "spawn"
    assert result["VLLM_ASCEND_ENABLE_NZ"] == "0"
    assert result["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
    assert result["VLLM_ENABLE_PREFILL_CUDAGRAPH"] == "1"
    assert result["VLLM_GR_PREFILL_GRAPH_BUCKETS"] == "128,512"
    assert result["VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS"] == ""
    assert "VLLM_GR_PREFILL_GRAPH_KV_BOUND" not in result
    assert "VLLM_GR_TEST_PADDED_PREFILL" not in result
    assert inherited["VLLM_ENABLE_PREFILL_CUDAGRAPH"] == "0"


def test_p0_matrix_required_npu_cannot_skip(
    p0_matrix_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    unavailable = p0_matrix_helpers["_npu_test_unavailable"]
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", "npu")
    with pytest.raises(pytest.fail.Exception, match="required NPU P0 runtime"):
        unavailable("operators missing")
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", "cuda")
    with pytest.raises(pytest.skip.Exception, match="operators missing"):
        unavailable("operators missing")
