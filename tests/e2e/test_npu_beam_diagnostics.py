# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""CPU-only checks for the opt-in E2E diagnostic observer."""

import ast
import json
from collections.abc import Mapping
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.e2e.npu_beam_diagnostics import NPUForwardTrace
from tests.helpers.paths import TESTS_DIR


@pytest.fixture
def prefill_case_helpers() -> dict[str, Any]:
    # Exercise the pure case policy without importing NPU runtime dependencies.
    source = TESTS_DIR / "st/test_onerec_v1_npu_e2e.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        "_padded_prefill_requested",
        "_graph_test_settings",
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
    settings = helpers["_graph_test_settings"]
    assert settings({}) == {"prefill_token_buckets": [128, 256]}
    assert settings({"VLLM_GR_TEST_PADDED_PREFILL": "1"}) == {
        "prefill_token_buckets": [384],
    }


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


pytestmark = pytest.mark.cpu_test
