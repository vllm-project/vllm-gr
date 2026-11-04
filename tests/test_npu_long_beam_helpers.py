# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""CPU controls for long-sequence test evidence, without NPU runtime imports."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, TypeAlias

import pytest

from tests.st.npu_long_beam_helpers import (
    assert_long_graph_trace,
    assert_long_outputs_equal,
    long_test_environment,
)

SHAPES = [(1, 1024), (1, 4096), (4, 1024), (4, 4096)]
Trace: TypeAlias = list[dict[str, Any]]
Packed: TypeAlias = list[list[tuple[list[int], float, str, str | None]]]


def _trace(batch_size: int, input_len: int, graph: bool) -> Trace:
    total = batch_size * input_len
    shared = dict(
        compilation_mode="VLLM_COMPILE" if graph else "NONE",
        enforce_eager=not graph,
        skip_compiled=not graph,
    )
    prefill = dict(
        shared,
        stage=0,
        kind="prefill",
        # GR Prefill is a separate replay, even when the outer mode is PIECEWISE.
        runtime_mode="PIECEWISE" if graph else "NONE",
        actual_tokens=total,
        padded_tokens=total,
        input_ids_shape=[total],
        positions_shape=[total],
        input_request_ids=[f"request-{row}" for row in range(batch_size)],
        cold_rows=[
            dict(num_computed_tokens=0, prompt_tokens=input_len, scheduled_tokens=input_len)
            for _ in range(batch_size)
        ],
        scheduled_total=total,
        attention=dict(
            query_start_loc=[row * input_len for row in range(batch_size + 1)],
            seq_lens=[input_len] * batch_size,
        ),
        gr_prefill_replay_delta={str(total): 1} if graph else {},
    )
    return [prefill] + [
        dict(
            shared,
            stage=stage,
            kind="decode",
            runtime_mode="FULL" if graph else "NONE",
            actual_tokens=batch_size * 8,
            padded_tokens=batch_size * 8,
            input_ids_shape=[batch_size * 8],
            positions_shape=[batch_size * 8],
            gr_prefill_replay_delta={},
        )
        for stage in (1, 2)
    ]


def _packed(batch_size: int) -> Packed:
    return [
        [
            ([1000 + 100 * row + rank, 2000 + rank, 3000 + row], -1.0 - rank, "length", None)
            for rank in range(8)
        ]
        for row in range(batch_size)
    ]


@pytest.mark.parametrize(("batch_size", "input_len"), SHAPES)
def test_environment_overrides_stale_graph_policy_without_mutating_input(
    batch_size: int, input_len: int
) -> None:
    inherited = {
        "unrelated": "keep",
        "VLLM_GR_TEST_PADDED_PREFILL": "1",
        "VLLM_GR_PREFILL_GRAPH_BUCKETS": "256",
        "VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS": "2",
        "VLLM_GR_PREFILL_GRAPH_KV_BOUND": "128",
        "VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS": "stale-shape-spec",
        "VLLM_GR_PREFILL_GRAPH_MAX_BUCKET": "256",
        "VLLM_ENABLE_PREFILL_CUDAGRAPH": "0",
        "VLLM_GR_NPU_FUSED_FFN": "1",
        "VLLM_ASCEND_ENABLE_NZ": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
    }
    original = inherited.copy()
    configured = long_test_environment(
        MappingProxyType(inherited), batch_size=batch_size, input_len=input_len
    )
    assert inherited == original and configured is not inherited
    assert configured["unrelated"] == "keep"
    assert "VLLM_GR_TEST_PADDED_PREFILL" not in configured
    assert configured["VLLM_GR_PREFILL_GRAPH_BUCKETS"] == str(batch_size * input_len)
    assert configured["VLLM_GR_PREFILL_GRAPH_MAX_BUCKET"] == str(batch_size * input_len)
    assert configured["VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS"] == str(batch_size)
    assert "VLLM_GR_PREFILL_GRAPH_KV_BOUND" not in configured
    assert "VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS" not in configured
    assert configured["VLLM_ENABLE_PREFILL_CUDAGRAPH"] == "1"
    assert configured["VLLM_GR_TARGET_DEVICE"] == "npu"
    assert configured["VLLM_GR_NPU_FUSED_FFN"] == "0"
    assert configured["VLLM_ASCEND_ENABLE_NZ"] == "0"
    assert configured["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"


@pytest.mark.parametrize(
    ("batch_size", "input_len"), [(0, 1024), (2, 1024), (8, 4096), (1, 128), (4, 2048)]
)
def test_environment_rejects_shapes_outside_declared_matrix(
    batch_size: int, input_len: int
) -> None:
    with pytest.raises(ValueError):
        long_test_environment({}, batch_size=batch_size, input_len=input_len)


@pytest.mark.parametrize(("batch_size", "input_len"), SHAPES)
@pytest.mark.parametrize("graph", [False, True])
def test_complete_long_prefill_and_two_decodes_are_accepted(
    batch_size: int, input_len: int, graph: bool
) -> None:
    events = _trace(batch_size, input_len, graph)
    saved = deepcopy(events)
    assert_long_graph_trace(events, batch_size=batch_size, input_len=input_len, graph=graph)
    assert events == saved, "Evidence validation must not alter the observed trace"


@pytest.mark.parametrize(
    "problem",
    [
        "missing_decode",
        "reordered_stages",
        "wrong_kind",
        "cached_history",
        "chunked_prefill",
        "duplicate_request_ids",
        "wrong_scheduled_total",
        "larger_padding_bucket",
        "wrong_input_shape",
        "wrong_positions_shape",
        "wrong_request_boundary",
        "wrong_history_length",
        "capture_without_replay",
        "wrong_replay_bucket",
        "double_replay",
        "skip_compiled",
        "decode_fallback",
        "wrong_decode_rows",
        "wrong_compilation",
        "wrong_enforce_eager",
    ],
)
def test_graph_evidence_rejects_incomplete_work_and_fallback(problem: str) -> None:
    batch, length = 4, 4096
    events = _trace(batch, length, True)
    prefill = events[0]
    if problem == "missing_decode":
        events.pop()
    elif problem == "reordered_stages":
        events[1], events[2] = events[2], events[1]
    elif problem == "wrong_kind":
        events[1]["kind"] = "prefill"
    elif problem == "cached_history":
        prefill["cold_rows"][2]["num_computed_tokens"] = length
    elif problem == "chunked_prefill":
        prefill["cold_rows"][2]["scheduled_tokens"] = length // 2
    elif problem == "duplicate_request_ids":
        prefill["input_request_ids"][3] = prefill["input_request_ids"][2]
    elif problem == "wrong_scheduled_total":
        prefill["scheduled_total"] -= length
    elif problem == "larger_padding_bucket":
        prefill["padded_tokens"] += 128
    elif problem == "wrong_input_shape":
        prefill["input_ids_shape"] = [batch, length]
    elif problem == "wrong_positions_shape":
        prefill["positions_shape"] = [batch, length]
    elif problem == "wrong_request_boundary":
        prefill["attention"]["query_start_loc"][2] += 1
    elif problem == "wrong_history_length":
        prefill["attention"]["seq_lens"][2] += 1
    elif problem == "capture_without_replay":
        prefill["gr_prefill_replay_delta"] = {}
    elif problem == "wrong_replay_bucket":
        prefill["gr_prefill_replay_delta"] = {str(batch * length + 128): 1}
    elif problem == "double_replay":
        prefill["gr_prefill_replay_delta"] = {str(batch * length): 2}
    elif problem == "skip_compiled":
        prefill["skip_compiled"] = True
    elif problem == "decode_fallback":
        events[1]["runtime_mode"] = "PIECEWISE"
    elif problem == "wrong_decode_rows":
        events[1]["actual_tokens"] = batch * 8 - 1
    elif problem == "wrong_compilation":
        prefill["compilation_mode"] = "NONE"
    elif problem == "wrong_enforce_eager":
        events[2]["enforce_eager"] = True
    else:
        raise AssertionError(f"Unimplemented control: {problem}")
    with pytest.raises(AssertionError):
        assert_long_graph_trace(events, batch_size=batch, input_len=length, graph=True)


@pytest.mark.parametrize("problem", ["prefill_replay", "decode_full", "compiled_eager"])
def test_eager_evidence_cannot_pass_using_graph_execution(problem: str) -> None:
    events = _trace(1, 1024, False)
    if problem == "prefill_replay":
        events[0]["gr_prefill_replay_delta"] = {"1024": 1}
    elif problem == "decode_full":
        events[1]["runtime_mode"] = "FULL"
    else:
        events[0]["compilation_mode"] = "VLLM_COMPILE"
    with pytest.raises(AssertionError):
        assert_long_graph_trace(events, batch_size=1, input_len=1024, graph=False)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_equal_outputs_accept_distinct_objects_without_mutation(batch_size: int) -> None:
    reference = _packed(batch_size)
    candidate = deepcopy(reference)
    assert_long_outputs_equal(reference, candidate, label="same logical batch")
    assert reference == candidate == _packed(batch_size)


@pytest.mark.parametrize(
    ("score", "delta", "accepted"),
    [(-1.0, 0.00009, True), (-1.0, 0.0002, False), (-100.0, 0.0009, True), (-100.0, 0.002, False)],
)
def test_score_comparison_keeps_existing_absolute_and_relative_limits(
    score: float, delta: float, accepted: bool
) -> None:
    reference = _packed(1)
    tokens, _, finish, stop = reference[0][0]
    reference[0][0] = (tokens, score, finish, stop)
    candidate = deepcopy(reference)
    candidate[0][0] = (tokens.copy(), score + delta, finish, stop)
    if accepted:
        assert_long_outputs_equal(reference, candidate, label="existing score tolerance")
    else:
        with pytest.raises(AssertionError):
            assert_long_outputs_equal(reference, candidate, label="excess score drift")


@pytest.mark.parametrize(
    "problem", ["token", "rank", "missing_request", "missing_beam", "finish", "stop", "nan", "inf"]
)
def test_output_comparison_preserves_order_and_completion_semantics(problem: str) -> None:
    reference = _packed(4)
    candidate = deepcopy(reference)
    if problem == "token":
        candidate[2][5][0][1] += 1
    elif problem == "rank":
        candidate[2][0], candidate[2][1] = candidate[2][1], candidate[2][0]
    elif problem == "missing_request":
        candidate.pop()
    elif problem == "missing_beam":
        candidate[2].pop()
    else:
        tokens, score, finish, stop = candidate[2][5]
        if problem == "finish":
            finish = "stop"
        elif problem == "stop":
            stop = "eos"
        elif problem == "nan":
            score = float("nan")
        else:
            score = float("inf")
        candidate[2][5] = (tokens, score, finish, stop)
    with pytest.raises(AssertionError):
        assert_long_outputs_equal(reference, candidate, label="different public output")
