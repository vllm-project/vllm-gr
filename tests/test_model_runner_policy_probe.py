# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Validate model probes' startup and request contracts without an accelerator."""

from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.function.model_runner_policy_isolation import startup_expectations
from tests.e2e.function.test_modelrunner_v2_stage_parity_st import _SESSIONS


@pytest.mark.parametrize(  # type: ignore[misc]
    "runner,sizes,max_seqs,enabled,eager,startup,expected",
    [
        pytest.param("v2", None, 8, True, False, "4", [1, 2, 4], id="default"),
        pytest.param("v2", None, 1, True, False, "4", [1], id="single-capacity"),
        pytest.param("v2", None, 3, True, False, "4", [1, 2], id="bounded-capacity"),
        pytest.param("v2", [7, 1, 3], 7, True, False, "4", [1, 3, 7], id="custom"),
        pytest.param("v2", None, 8, False, False, "4", [], id="graph-disabled"),
        pytest.param("v2", None, 8, True, True, "4", [], id="eager"),
        pytest.param("v1", [1, 3, 7], 8, True, False, None, [1], id="v1-default"),
        pytest.param("v1", [1, 3, 7], 8, True, False, "4", [4], id="v1-startup"),
        pytest.param("v1", None, 8, False, False, "4", [], id="v1-graph-disabled"),
        pytest.param("v1", None, 8, True, True, "4", [], id="v1-eager"),
    ],
)
def test_probe_separates_capture_policy_from_backend_capacity(
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
    sizes: list[int] | None,
    max_seqs: int,
    enabled: bool,
    eager: bool,
    startup: str | None,
    expected: list[int],
) -> None:
    monkeypatch.delenv("VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE", raising=False)
    if startup is not None:
        monkeypatch.setenv("VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE", startup)
    monkeypatch.setenv("VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS", "1,6,8")
    beam: dict[str, Any] = {"max_width": 128, "graph_enabled": enabled}
    if sizes is not None:
        beam["online_batch_sizes"] = sizes
    assert startup_expectations(
        runner=runner,
        config={"schema_version": 1, "beam": beam, "constraint_table": {"max_top_k": 128}},
        max_num_seqs=max_seqs,
        enforce_eager=eager,
    ) == (1024 if runner == "v2" else 128, expected)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("length,steps", _SESSIONS)  # type: ignore[misc]
def test_v2_parity_sessions_pass_constrained_request_validation(length: int, steps: int) -> None:
    from tests.e2e.function.test_onerec_v1_scheduler_st import _params
    from tests.test_beam_search_v1_input_processor import _make_engine
    from vllm_gr.entrypoints.beam_search_v1.input_processor import (
        _create_request_params,
    )

    # Exercise the real ST parameters and request normalization before paying
    # for model startup. A boundary-wrapped SID must still contain three tokens.
    engine = _make_engine()
    tokenizer = SimpleNamespace(
        eos_token_id=2,
        convert_tokens_to_ids=lambda token: {"<|sid_begin|>": 7, "<|sid_end|>": 8}[token],
    )
    native, begin, end = _create_request_params(
        _params(steps), tokenizer, vllm_config=engine.vllm_config
    )
    assert native.max_tokens == 3 and (begin, end) == (7, 8), (
        f"invalid parity session {length=}, {steps=}"
    )


pytestmark = pytest.mark.cpu_test
