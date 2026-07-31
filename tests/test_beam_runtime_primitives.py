# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for platform-independent Beam runtime primitives."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.geometry import _compute_beam_bounds


@pytest.mark.parametrize(  # type: ignore[misc]
    ("args", "expected"),
    [
        ((0, 10, 6, 4, 99), (8, 8, 0, 0)),
        ((0, 10, 12, 4, 99), (2, 4, 2, 0)),
        ((1, 10, 16, 4, 99), (2, 4, 2, 1)),
        ((0, 10, 6, 4, 2), (2, 2, 0, 0)),
    ],
    ids=[
        "prefix-incomplete",
        "first-beam-suffix",
        "later-beam-partial-cache",
        "chunk-budget-smaller-than-remaining",
    ],
)
def test_compute_beam_bounds_behavior(
    args: tuple[int, int, int, int, int],
    expected: tuple[int, int, int, int],
) -> None:
    assert _compute_beam_bounds(*args) == expected


def test_beam_data_var_defaults_to_none_and_reset_restores_default() -> None:
    assert BEAM_DATA_VAR.get() is None
    value = {"beam_data": {"request": {}}, "req_ids": ["request"]}
    token = BEAM_DATA_VAR.set(value)
    try:
        assert BEAM_DATA_VAR.get() is value
    finally:
        BEAM_DATA_VAR.reset(token)
    assert BEAM_DATA_VAR.get() is None


def test_execute_model_wrapper_publishes_and_restores_context() -> None:
    from vllm_gr.v1.worker.model_runner_common import make_patched_execute_model

    beam_data = {"request": {"is_beam_decode": True}}
    scheduler_output = SimpleNamespace(beam_data=beam_data)
    observed: list[Any] = []

    def execute_model(runner: Any, output: Any) -> str:
        observed.append(BEAM_DATA_VAR.get())
        assert output is scheduler_output
        assert runner._current_beam_info is observed[-1]
        return "result"

    runner = SimpleNamespace()
    wrapped = make_patched_execute_model(execute_model)

    assert wrapped(runner, scheduler_output) == "result"
    assert observed == [{"beam_data": beam_data, "req_ids": []}]
    assert BEAM_DATA_VAR.get() is None
    assert runner._current_beam_data == {}
    assert runner._current_beam_info is None
