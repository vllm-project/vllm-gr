# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""CPU-only checks for the opt-in E2E diagnostic observer."""

import ast
import json
import os
from collections.abc import Mapping
from copy import deepcopy
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
        "_assert_b4_graph_trace",
        "_npu_test_unavailable",
        "_b4_test_environment",
    }
    nodes: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "os": os,
        "pytest": pytest,
        "_BEAM_WIDTH": 8,
    }
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


@pytest.mark.parametrize("bsnd_capture_specs", [None, "512,4", "512,4;1024,4"])
def test_b4_environment_overrides_stale_flags_without_mutating_input(
    prefill_case_helpers: dict[str, Any],
    bsnd_capture_specs: str | None,
) -> None:
    inherited = {
        "VLLM_GR_TARGET_DEVICE": "cuda",
        "VLLM_GR_TEST_PADDED_PREFILL": "1",
        "VLLM_ENABLE_PREFILL_CUDAGRAPH": "0",
        "VLLM_GR_PREFILL_GRAPH_BUCKETS": "256",
        "VLLM_GR_PREFILL_GRAPH_BUCKET_STEP": "384",
        "VLLM_GR_PREFILL_GRAPH_MAX_BUCKET": "256",
        "VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS": "2",
        "VLLM_GR_PREFILL_GRAPH_KV_BOUND": "2048",
        "VLLM_GR_NPU_FUSED_FFN": "1",
        "VLLM_ASCEND_ENABLE_NZ": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "unrelated": "keep",
    }
    if bsnd_capture_specs is not None:
        inherited["VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS"] = bsnd_capture_specs
    original = inherited.copy()
    configured = prefill_case_helpers["_b4_test_environment"](inherited)
    assert configured == {
        "VLLM_GR_TARGET_DEVICE": "npu",
        "VLLM_ENABLE_PREFILL_CUDAGRAPH": "1",
        "VLLM_GR_PREFILL_GRAPH_BUCKETS": "512",
        "VLLM_GR_PREFILL_GRAPH_BUCKET_STEP": "128",
        "VLLM_GR_PREFILL_GRAPH_MAX_BUCKET": "512",
        "VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS": "4",
        "VLLM_GR_NPU_FUSED_FFN": "0",
        "VLLM_ASCEND_ENABLE_NZ": "0",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "unrelated": "keep",
    }
    assert configured is not inherited
    assert inherited == original


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


def _b4_trace(*, graph: bool) -> list[dict[str, Any]]:
    prefill = dict(
        stage=0,
        kind="prefill",
        actual_tokens=512,
        padded_tokens=512,
        input_ids_shape=[512],
        positions_shape=[512],
        input_request_ids=[f"r{index}" for index in range(4)],
        attention=dict(query_start_loc=[0, 128, 256, 384, 512], seq_lens=[128] * 4),
        enforce_eager=not graph,
        compilation_mode="VLLM_COMPILE" if graph else "NONE",
        # GR Prefill replay is separate from the native outer runtime mode.
        runtime_mode="PIECEWISE" if graph else "NONE",
        skip_compiled=False,
        gr_prefill_replay_delta={"512": 1} if graph else {},
        cold_rows=[
            dict(num_computed_tokens=0, prompt_tokens=128, scheduled_tokens=128) for _ in range(4)
        ],
        scheduled_total=512,
    )
    return [
        prefill,
        *[
            dict(
                stage=stage,
                kind="decode",
                actual_tokens=32,
                padded_tokens=32,
                runtime_mode="FULL" if graph else "NONE",
                enforce_eager=not graph,
                compilation_mode="VLLM_COMPILE" if graph else "NONE",
                gr_prefill_replay_delta={},
            )
            for stage in (1, 2)
        ],
    ]


@pytest.mark.parametrize("graph", [False, True])
def test_b4_trace_accepts_real_prefill_and_decode_evidence(
    prefill_case_helpers: dict[str, Any], graph: bool
) -> None:
    events = _b4_trace(graph=graph)
    original = deepcopy(events)
    prefill_case_helpers["_assert_exact_prefill"](events[0], "B4 exact input", batch_size=4)
    prefill_case_helpers["_assert_b4_graph_trace"](events, graph=graph)
    assert events == original, "evidence validation must not rewrite the observed trace"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_replay",
        "wrong_bucket",
        "replayed_twice",
        "padded_input",
        "wrong_input_shape",
        "wrong_query_offsets",
        "wrong_sequence_lengths",
        "cached_history",
        "wrong_prompt_length",
        "partial_scheduling",
        "missing_cold_row",
        "wrong_scheduled_total",
        "missing_decode",
        "wrong_stage_order",
        "wrong_stage_kind",
        "nonfull_decode",
        "wrong_decode_rows",
    ],
)
def test_b4_graph_trace_rejects_incomplete_or_wrong_execution(
    prefill_case_helpers: dict[str, Any], mutation: str
) -> None:
    events = _b4_trace(graph=True)
    prefill = events[0]
    if mutation == "missing_replay":
        prefill["gr_prefill_replay_delta"] = {}
    elif mutation == "wrong_bucket":
        prefill["gr_prefill_replay_delta"] = {"256": 1}
    elif mutation == "replayed_twice":
        prefill["gr_prefill_replay_delta"] = {"512": 2}
    elif mutation == "padded_input":
        prefill["padded_tokens"] = 640
    elif mutation == "wrong_input_shape":
        prefill["input_ids_shape"] = [256]
    elif mutation == "wrong_query_offsets":
        prefill["attention"]["query_start_loc"] = [0, 128, 256, 384, 511]
    elif mutation == "wrong_sequence_lengths":
        prefill["attention"]["seq_lens"] = [128, 128, 128, 127]
    elif mutation == "cached_history":
        prefill["cold_rows"][0]["num_computed_tokens"] = 1
    elif mutation == "wrong_prompt_length":
        prefill["cold_rows"][0]["prompt_tokens"] = 127
    elif mutation == "partial_scheduling":
        prefill["cold_rows"][0]["scheduled_tokens"] = 127
    elif mutation == "missing_cold_row":
        prefill["cold_rows"].pop()
    elif mutation == "wrong_scheduled_total":
        prefill["scheduled_total"] = 511
    elif mutation == "missing_decode":
        events.pop()
    elif mutation == "wrong_stage_order":
        events[1], events[2] = events[2], events[1]
    elif mutation == "wrong_stage_kind":
        events[1]["kind"] = "prefill"
    elif mutation == "nonfull_decode":
        events[1]["runtime_mode"] = "PIECEWISE"
    elif mutation == "wrong_decode_rows":
        events[1]["actual_tokens"] = 16
    else:
        raise AssertionError(f"unhandled negative control {mutation}")
    with pytest.raises(AssertionError):
        prefill_case_helpers["_assert_b4_graph_trace"](events, graph=True)


@pytest.mark.parametrize("mutation", ["graph_replay", "compiled_prefill", "full_decode"])
def test_b4_eager_trace_rejects_graph_execution(
    prefill_case_helpers: dict[str, Any], mutation: str
) -> None:
    events = _b4_trace(graph=False)
    if mutation == "graph_replay":
        events[0]["gr_prefill_replay_delta"] = {"512": 1}
    elif mutation == "compiled_prefill":
        events[0]["compilation_mode"] = "VLLM_COMPILE"
    else:
        events[1]["runtime_mode"] = "FULL"
    with pytest.raises(AssertionError):
        prefill_case_helpers["_assert_b4_graph_trace"](events, graph=False)


@pytest.mark.parametrize("reason", ["torch-npu missing", "NPU unavailable", "xllm missing"])
def test_missing_npu_gate_resources_fail_instead_of_skipping(
    prefill_case_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", "npu")
    with pytest.raises(pytest.fail.Exception, match=reason):
        prefill_case_helpers["_npu_test_unavailable"](reason)


@pytest.mark.parametrize("target", [None, "cpu", "cuda"])
def test_missing_npu_resources_can_skip_other_platforms(
    prefill_case_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch, target: str | None
) -> None:
    if target is None:
        monkeypatch.delenv("VLLM_GR_TARGET_DEVICE", raising=False)
    else:
        monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", target)
    with pytest.raises(pytest.skip.Exception, match="NPU unavailable"):
        prefill_case_helpers["_npu_test_unavailable"]("NPU unavailable")


@pytest.mark.parametrize("fails", [False, True])
def test_b4_forward_probe_snapshots_cold_rows_and_restores(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    class Mode(Enum):
        NONE = 0
        PIECEWISE = 1
        FULL_AND_PIECEWISE = 2
        VLLM_COMPILE = 3

    request_ids = tuple(f"r{index}" for index in range(4))
    input_ids = torch.arange(512)
    positions = torch.arange(128).repeat(4)
    query_start = torch.arange(5) * 128
    sentinel = object()
    calls: list[Any] = []
    context = SimpleNamespace(
        cudagraph_runtime_mode=Mode.PIECEWISE,
        skip_compiled=False,
        batch_descriptor=SimpleNamespace(num_tokens=512, num_reqs=4, uniform=False),
        attn_metadata={"layer.0": SimpleNamespace(query_start_loc=query_start, seq_lens=[128] * 4)},
    )

    class Runner:
        _prefill_graph_runner: Any
        requests: dict[str, Any]
        _current_scheduler_output: Any

        def _model_forward(self, num_tokens_padded: int, input_ids: Any, positions: Any) -> Any:
            calls.append((num_tokens_padded, input_ids, positions))
            self._prefill_graph_runner._replays_by_bucket[512] += 1
            input_ids.fill_(99)
            query_start.fill_(99)
            for request_id in request_ids:
                self.requests[request_id].num_computed_tokens = 99
                self.requests[request_id].prompt_token_ids.clear()
                self._current_scheduler_output.num_scheduled_tokens[request_id] = 1
            if fails:
                raise ValueError("original B4 forward error")
            return sentinel

    runner: Any = Runner()
    runner._prefill_graph_runner = SimpleNamespace(_replays_by_bucket={512: 5})
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=Mode.VLLM_COMPILE,
            cudagraph_mode=Mode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[32, 512],
        )
    )
    runner.model_config = SimpleNamespace(enforce_eager=False)
    runner.model = sentinel
    runner.input_batch = SimpleNamespace(req_ids=list(request_ids))
    runner.requests = {
        request_id: SimpleNamespace(
            num_computed_tokens=0, num_prompt_tokens=128, prompt_token_ids=list(range(128))
        )
        for request_id in request_ids
    }
    runner._current_scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request_id: 128 for request_id in request_ids},
        total_num_scheduled_tokens=512,
    )
    runner._gr_npu_stage_plan = SimpleNamespace(
        execution=SimpleNamespace(
            batch_size=4, stage_index=0, kind="prefill", request_ids=request_ids
        ),
        model_token_count=512,
    )
    if fails:
        runner._model_forward = runner._model_forward
    original_override = vars(runner).get("_model_forward")
    probe = NPUForwardTrace(runner, lambda: context, expected_batch_size=4, record_cold=True)
    try:
        with monkeypatch.context() as guard:

            def forbidden(*args: Any, **kwargs: Any) -> None:
                raise AssertionError("host transfer inside model forward")

            guard.setattr(torch.Tensor, "cpu", forbidden)
            guard.setattr(torch.Tensor, "tolist", forbidden)
            if fails:
                with pytest.raises(ValueError, match="original B4 forward error"):
                    runner._model_forward(512, input_ids, positions=positions)
            else:
                assert runner._model_forward(512, input_ids, positions=positions) is sentinel
    finally:
        events = probe.finish()
    assert probe.finish() == []
    assert vars(runner).get("_model_forward") is original_override
    assert len(calls) == 1 and calls[0][1] is input_ids and calls[0][2] is positions
    assert len(events) == 1
    event = events[0]
    assert event["input_ids"] == list(range(512))
    assert event["positions"] == list(range(128)) * 4
    assert event["attention"]["query_start_loc"] == [0, 128, 256, 384, 512]
    assert event["cold_rows"] == [
        dict(num_computed_tokens=0, prompt_tokens=128, scheduled_tokens=128) for _ in range(4)
    ]
    assert event["scheduled_total"] == 512
    assert event["input_request_ids"] == list(request_ids)
    assert event["gr_prefill_replay_delta"] == {"512": 1}
    json.dumps(events, allow_nan=False)
