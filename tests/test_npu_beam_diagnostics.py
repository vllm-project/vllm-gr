# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


@pytest.mark.parametrize("batch_size,input_len", [(1, 1024), (4, 1024), (1, 4096), (4, 4096)])
def test_forward_probe_long_bound_and_cold_scheduler(batch_size: int, input_len: int) -> None:
    total = batch_size * input_len
    request_ids = [f"r{row}" for row in range(batch_size)]
    context = SimpleNamespace(cudagraph_runtime_mode="NONE", skip_compiled=False, attn_metadata={})

    class Runner:
        def _model_forward(self, num_tokens_padded: int, input_ids: Any, positions: Any) -> Any:
            return input_ids

    runner: Any = Runner()
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode="NONE", cudagraph_mode="NONE", cudagraph_capture_sizes=[]
        )
    )
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.model = None
    runner.input_batch = SimpleNamespace(req_ids=request_ids)
    runner._gr_npu_stage_plan = SimpleNamespace(
        execution=SimpleNamespace(
            batch_size=batch_size, stage_index=0, kind="prefill", request_ids=request_ids
        ),
        model_token_count=total,
    )
    runner.requests = {
        rid: SimpleNamespace(num_computed_tokens=0, prompt_token_ids=[1] * input_len)
        for rid in request_ids
    }
    runner._current_scheduler_output = SimpleNamespace(
        num_scheduled_tokens={rid: input_len for rid in request_ids},
        total_num_scheduled_tokens=total,
    )
    trace = NPUForwardTrace(
        runner,
        lambda: context,
        expected_batch_size=batch_size,
        max_padded_tokens=total,
        record_cold=True,
    )
    inputs = torch.arange(total)
    try:
        with pytest.raises(RuntimeError, match="unexpected or unbounded"):
            runner._model_forward(total + 1, inputs, inputs)
        assert runner._model_forward(total, inputs, inputs) is inputs
    finally:
        events = trace.finish()
    assert len(events) == 1
    assert events[0]["input_ids"] == list(range(total))
    assert events[0]["scheduled_total"] == total
    assert (
        events[0]["cold_rows"]
        == [{"num_computed_tokens": 0, "prompt_tokens": input_len, "scheduled_tokens": input_len}]
        * batch_size
    )
    assert "_model_forward" not in vars(runner)


@pytest.mark.parametrize("batch_size,max_tokens", [(0, 512), (9, 512), (1, 0)])
def test_forward_probe_rejects_unbounded_configuration(batch_size: int, max_tokens: int) -> None:
    with pytest.raises(ValueError, match="bounded"):
        NPUForwardTrace(
            SimpleNamespace(),
            lambda: None,
            expected_batch_size=batch_size,
            max_padded_tokens=max_tokens,
        )
