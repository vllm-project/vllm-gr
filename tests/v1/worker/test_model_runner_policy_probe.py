# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Validate model probes' startup and request contracts without an accelerator."""

from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.model_runner_policy_isolation import startup_expectations
from tests.e2e.test_modelrunner_v2_stage_parity_st import _SESSIONS


@pytest.mark.parametrize(  # type: ignore[misc]
    "runner,sizes,max_seqs,enabled,eager,expected",
    [
        pytest.param("v2", None, 8, True, False, [1, 2, 4], id="default"),
        pytest.param("v2", None, 1, True, False, [1], id="single-capacity"),
        pytest.param("v2", None, 3, True, False, [1, 2], id="bounded-capacity"),
        pytest.param("v2", [1, 6, 8], 7, True, False, [1, 6], id="custom"),
        pytest.param("v2", None, 8, False, False, [], id="graph-disabled"),
        pytest.param("v2", None, 8, True, True, [], id="eager"),
        pytest.param("v1", [1, 6, 8], 8, True, False, [1, 6, 8], id="v1-default"),
        pytest.param("v1", [1, 6, 8], 7, True, False, [1, 6], id="v1-startup"),
        pytest.param("v1", None, 8, False, False, [], id="v1-graph-disabled"),
        pytest.param("v1", None, 8, True, True, [], id="v1-eager"),
    ],
)
def test_probe_separates_capture_policy_from_backend_capacity(
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
    sizes: list[int] | None,
    max_seqs: int,
    enabled: bool,
    eager: bool,
    expected: list[int],
) -> None:
    deployment = {"beam_batch_sizes": sizes if sizes is not None else [1, 2, 4]}
    beam: dict[str, Any] = {
        "constraint_mode": "none",
        "beam_width": [128] * 3,
        "beam_graph_enabled": enabled,
        "candidate_width": [128] * 3,
    }
    assert startup_expectations(
        runner=runner,
        config={**beam, **deployment},
        max_num_seqs=max_seqs,
        enforce_eager=eager,
    ) == (1024 if runner == "v2" else 128, expected)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("length,steps", _SESSIONS)  # type: ignore[misc]
def test_v2_parity_sessions_pass_constrained_request_validation(length: int, steps: int) -> None:
    from tests.e2e.test_onerec_v1_scheduler_st import _params
    from tests.entrypoints.beam_search.test_beam_search_input_processor import (
        _make_engine,
    )
    from vllm_gr.entrypoints.beam_search.input_processor import (
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
