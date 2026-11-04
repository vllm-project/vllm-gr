# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-testable acceptance checks for bounded long-context NPU STs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def long_test_environment(
    env: Mapping[str, str], *, batch_size: int, input_len: int
) -> dict[str, str]:
    if batch_size not in (1, 4) or input_len not in (1024, 4096):
        raise ValueError("long NPU coverage is bounded to B=1/4 and L=1024/4096")
    total = batch_size * input_len
    result = dict(env)
    result.pop("VLLM_GR_TEST_PADDED_PREFILL", None)
    result.pop("VLLM_GR_PREFILL_GRAPH_KV_BOUND", None)
    result.pop("VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS", None)
    result.update(
        VLLM_GR_TARGET_DEVICE="npu",
        VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
        VLLM_GR_PREFILL_GRAPH_BUCKETS=str(total),
        VLLM_GR_PREFILL_GRAPH_BUCKET_STEP="128",
        VLLM_GR_PREFILL_GRAPH_MAX_BUCKET=str(total),
        VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS=str(batch_size),
        VLLM_GR_NPU_FUSED_FFN="0",
        VLLM_ASCEND_ENABLE_NZ="0",
        VLLM_USE_BREAKABLE_CUDAGRAPH="0",
    )
    return result


def assert_long_graph_trace(
    events: list[dict[str, Any]], *, batch_size: int, input_len: int, graph: bool
) -> None:
    total = batch_size * input_len
    assert [event["stage"] for event in events] == [0, 1, 2], "incomplete stage trace"
    assert [event["kind"] for event in events] == ["prefill", "decode", "decode"]
    prefill = events[0]
    assert prefill["actual_tokens"] == prefill["padded_tokens"] == total
    assert prefill["input_ids_shape"] == prefill["positions_shape"] == [total]
    assert prefill["attention"]["query_start_loc"] == list(range(0, total + 1, input_len))
    assert prefill["attention"]["seq_lens"] == [input_len] * batch_size
    assert len(prefill["input_request_ids"]) == len(set(prefill["input_request_ids"])) == batch_size
    assert (
        prefill["cold_rows"]
        == [{"num_computed_tokens": 0, "prompt_tokens": input_len, "scheduled_tokens": input_len}]
        * batch_size
    ), "long Prefill must be cold and unchunked"
    assert prefill["scheduled_total"] == total
    assert prefill["enforce_eager"] is (not graph)
    assert prefill["compilation_mode"] == ("VLLM_COMPILE" if graph else "NONE")
    assert prefill["gr_prefill_replay_delta"] == ({str(total): 1} if graph else {})
    if graph:
        assert prefill["skip_compiled"] is False
    else:
        assert prefill["runtime_mode"] == "NONE"
    for event in events[1:]:
        assert event["actual_tokens"] == batch_size * 8, "Decode lost Beam rows"
        assert event["runtime_mode"] == ("FULL" if graph else "NONE")
        assert event["enforce_eager"] is (not graph)


def assert_long_outputs_equal(reference: Any, candidate: Any, *, label: str) -> None:
    """Keep the original whole-model ST's score tolerance; tokens stay exact."""
    assert len(reference) == len(candidate), f"{label}: request count differs"
    for row, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        assert len(left) == len(right) == 8, f"{label}: Beam count differs, row={row}"
        for beam, (expected, actual) in enumerate(zip(left, right, strict=True)):
            assert actual[0] == expected[0], (
                f"{label}: ordered tokens differ, row={row}, beam={beam}"
            )
            assert actual[2:] == expected[2:], f"{label}: finish metadata differs, row={row}"
            expected_score, actual_score = float(expected[1]), float(actual[1])
            assert math.isfinite(expected_score) and math.isfinite(actual_score)
            assert abs(actual_score - expected_score) <= max(1e-4, 1e-5 * abs(expected_score)), (
                f"{label}: score differs, row={row}, beam={beam}: "
                f"{actual_score} vs {expected_score}"
            )
