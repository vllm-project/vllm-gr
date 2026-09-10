# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for BeamProc scheduling and request handling."""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import msgspec
import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.ops.sliced_logits import set_vocab_slice_config
from vllm_gr.v1.engine.core import (
    _BEAMPROC_CHILD_BATCH_REQUEST,
    _BEAMPROC_ENABLED_ENV,
    _BEAMPROC_WAKE_REQUEST,
    _apply_beamproc_grouped_plan,
    _apply_beamproc_plan,
    _beamproc_batch_payload,
    _beamproc_grouped_parent_state,
    _beamproc_logprobs_arrays,
    _beamproc_prepare_grouped_parent_state,
    _beamproc_queue_put,
    _beamproc_request_context,
    _beamproc_trace_headers_with_context,
    _cache_beam_request,
    _decode_beamproc_plan_v3_frames,
    _decode_beamproc_plan_v4_frames,
    _decode_beamproc_plan_v5_frames,
    _drain_pb_queue,
    _handle_beamproc_plan,
    _process_beamproc_plan_frames,
    make_patched_handle_client_request,
)


def test_beamproc_queue_put_reports_sustained_backpressure() -> None:
    bounded_queue: queue.Queue[str] = queue.Queue(maxsize=1)
    bounded_queue.put_nowait("occupied")

    with (
        patch("vllm_gr.v1.engine.core._BEAMPROC_QUEUE_PUT_TIMEOUT_S", 0.001),
        pytest.raises(RuntimeError, match="remained full"),
    ):
        _beamproc_queue_put(bounded_queue, "overflow", "test queue")


def _beamproc_plan_engine() -> SimpleNamespace:
    parent = SimpleNamespace(
        _all_token_ids=[10, 20],
        prompt_token_ids=[10, 20],
        num_prompt_tokens=2,
        sampling_params=SamplingParams(
            max_tokens=1,
            logprobs=3,
            temperature=0.0,
            detokenize=False,
        ),
        eos_token_id=2,
        client_index=4,
        current_wave=7,
        priority=3,
        lora_request=None,
        cache_salt=None,
        trace_headers=None,
        prompt_embeds=None,
        mm_features=None,
        block_hashes=[],
    )
    return SimpleNamespace(
        beam_cache={"p0": (parent, 0)},
        beam_cache_lock=threading.Lock(),
        request_block_hasher=None,
        beamproc_block_size=None,
        input_queue=queue.SimpleQueue(),
        output_queue=queue.SimpleQueue(),
        engine_index=0,
        pb_queue=queue.SimpleQueue(),
    )


def _two_child_plan() -> dict[str, Any]:
    return {
        "root_request_id": "p0",
        "parent_ids": ["p0", "p0"],
        "child_ids": ["c0", "c1"],
        "token_ids": [100, 200],
        "beam_tokens": [[100], [200]],
        "abort_ids": [],
        "client_index": 4,
        "beam_width": 2,
    }


def _mock_request(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs, block_hashes=[])


def test_decode_beamproc_plan_v3_histories() -> None:
    payload: dict[str, Any] = {
        "version": 3,
        "beam_width": 2,
        "token_ids_len": 2,
        "scores_len": 2,
        "flat_indices_len": 2,
        "beam_tokens_len": 5,
        "beam_token_offsets_len": 3,
    }
    frames = [
        b"BEAM_PLAN",
        b"metadata",
        np.asarray([100, 200], dtype="<i4").tobytes(),
        np.asarray([-1.0, -2.0], dtype="<f4").tobytes(),
        np.asarray([3, 4], dtype="<u8").tobytes(),
        np.asarray([10, 100, 20, 30, 200], dtype="<i4").tobytes(),
        np.asarray([0, 2, 5], dtype="<u8").tobytes(),
    ]

    assert _decode_beamproc_plan_v3_frames(payload, frames) is True
    assert payload["token_ids"] == [100, 200]
    assert payload["flat_indices"] == [3, 4]
    assert payload["beam_tokens"] == [[10, 100], [20, 30, 200]]


def test_decode_beamproc_plan_v4_keeps_flat_fixed_stride_history() -> None:
    payload: dict[str, Any] = {
        "version": 4,
        "beam_width": 2,
        "token_ids_len": 2,
        "scores_len": 2,
        "flat_indices_len": 2,
        "beam_tokens_len": 4,
        "beam_token_stride": 2,
    }
    frames = [
        b"BEAM_PLAN",
        b"metadata",
        np.asarray([100, 200], dtype="<i4").tobytes(),
        np.asarray([-1.0, -2.0], dtype="<f4").tobytes(),
        np.asarray([3, 4], dtype="<u8").tobytes(),
        np.asarray([10, 100, 20, 200], dtype="<i4").tobytes(),
    ]

    assert _decode_beamproc_plan_v4_frames(payload, frames) is True
    assert isinstance(payload["beam_tokens_flat"], np.ndarray)
    assert payload["beam_tokens_flat"].tolist() == [10, 100, 20, 200]


def test_decode_beamproc_plan_v4_skips_unused_auxiliary_lists() -> None:
    payload: dict[str, Any] = {
        "version": 4,
        "beam_width": 2,
        "token_ids_len": 2,
        "scores_len": 2,
        "flat_indices_len": 2,
        "beam_tokens_len": 4,
        "beam_token_stride": 2,
    }
    frames = [
        b"BEAM_PLAN",
        b"metadata",
        np.asarray([100, 200], dtype="<i4").tobytes(),
        np.asarray([-1.0, -2.0], dtype="<f4").tobytes(),
        np.asarray([3, 4], dtype="<u8").tobytes(),
        np.asarray([10, 100, 20, 200], dtype="<i4").tobytes(),
    ]

    assert (
        _decode_beamproc_plan_v4_frames(
            payload,
            frames,
            materialize_auxiliary=False,
        )
        is True
    )
    assert payload["token_ids"] == [100, 200]
    assert "scores" not in payload
    assert "flat_indices" not in payload
    assert payload["beam_tokens_flat"].tolist() == [10, 100, 20, 200]


def test_decode_beamproc_plan_v4_rejects_invalid_stride() -> None:
    payload: dict[str, Any] = {
        "version": 4,
        "beam_width": 2,
        "token_ids_len": 2,
        "scores_len": 2,
        "flat_indices_len": 2,
        "beam_tokens_len": 3,
        "beam_token_stride": 2,
    }
    frames = [
        b"BEAM_PLAN",
        b"metadata",
        np.asarray([100, 200], dtype="<i4").tobytes(),
        np.asarray([-1.0, -2.0], dtype="<f4").tobytes(),
        np.asarray([3, 4], dtype="<u8").tobytes(),
        np.asarray([10, 100, 20], dtype="<i4").tobytes(),
    ]

    assert _decode_beamproc_plan_v4_frames(payload, frames) is False


def test_decode_beamproc_plan_v5_uses_only_selected_tokens_and_histories() -> None:
    payload: dict[str, Any] = {
        "version": 5,
        "beam_width": 2,
        "token_ids_len": 2,
        "beam_tokens_len": 4,
        "beam_token_stride": 2,
    }
    frames = [
        b"BEAM_PLAN",
        b"metadata",
        np.asarray([100, 200], dtype="<i4").tobytes(),
        np.asarray([10, 100, 20, 200], dtype="<i4").tobytes(),
    ]

    assert _decode_beamproc_plan_v5_frames(payload, frames) is True
    assert payload["token_ids"] == [100, 200]
    assert payload["beam_tokens_flat"].tolist() == [10, 100, 20, 200]
    assert "scores" not in payload
    assert "flat_indices" not in payload


def test_grouped_rust_plan_enqueues_one_beam_decode_request() -> None:
    engine = _beamproc_plan_engine()

    with patch("vllm.v1.request.Request", side_effect=_mock_request):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True

    marker, requests, wave = engine.pb_queue.get_nowait()
    assert marker is _BEAMPROC_CHILD_BATCH_REQUEST
    assert len(requests) == 1
    grouped_request = requests[0]
    assert wave == 7
    assert grouped_request.prompt_token_ids == [10, 20, 100, 200]
    assert grouped_request.is_beam_decode is True
    assert grouped_request.beam_width == 2
    assert grouped_request.beam_decode_steps == 1
    assert grouped_request.prefix_len == 2


def test_grouped_rust_plan_accepts_flat_fixed_stride_history() -> None:
    engine = _beamproc_plan_engine()
    plan = _two_child_plan()
    plan.pop("beam_tokens")
    plan["beam_tokens_flat"] = np.asarray([100, 200], dtype=np.int32)
    plan["beam_token_stride"] = 1

    with patch("vllm.v1.request.Request", side_effect=_mock_request):
        assert _apply_beamproc_plan(engine, plan) is True

    _, requests, _ = engine.pb_queue.get_nowait()
    assert requests[0].prompt_token_ids == [10, 20, 100, 200]
    assert requests[0].beam_decode_steps == 1


def test_grouped_rust_flat_history_strips_fixed_response_prefix() -> None:
    engine = _beamproc_plan_engine()
    parent = engine.beam_cache["p0"][0]
    parent._all_token_ids = [10, 20, 99]
    parent.prompt_token_ids = [10, 20, 99]
    parent.num_prompt_tokens = 3
    parent.trace_headers = _beamproc_trace_headers_with_context(
        None,
        {"prompt_token_count": 2, "step_index": 0},
    )
    plan = _two_child_plan()
    plan.pop("beam_tokens")
    plan["beam_tokens_flat"] = np.asarray(
        [99, 100, 99, 200],
        dtype=np.int32,
    )
    plan["beam_token_stride"] = 2

    with patch("vllm.v1.request.Request", side_effect=_mock_request):
        assert _apply_beamproc_plan(engine, plan) is True

    _, requests, _ = engine.pb_queue.get_nowait()
    assert requests[0].prompt_token_ids == [10, 20, 99, 100, 200]
    assert requests[0].beam_decode_steps == 1


def test_grouped_rust_plan_resolves_duplicate_parent_once() -> None:
    engine = _beamproc_plan_engine()

    with (
        patch(
            "vllm_gr.v1.engine.core._beamproc_grouped_parent_state",
            wraps=_beamproc_grouped_parent_state,
        ) as parent_state,
        patch("vllm.v1.request.Request", side_effect=_mock_request),
    ):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True

    assert parent_state.call_count == 1


def test_grouped_rust_plan_reuses_compact_root_state_on_later_step() -> None:
    engine = _beamproc_plan_engine()

    with patch("vllm.v1.request.Request", side_effect=_mock_request):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True
        root_state = engine.beamproc_grouped_parent_states["p0"]
        first_grouped_id = engine.beamproc_grouped_request_by_root["p0"]
        first_grouped_request = engine.beam_cache[first_grouped_id][0]
        first_grouped_request.prompt_token_ids[:2] = [98, 99]
        second_plan = {
            **_two_child_plan(),
            "parent_ids": ["c0", "c1"],
            "child_ids": ["d0", "d1"],
            "token_ids": [101, 201],
            "beam_tokens": [[100, 101], [200, 201]],
        }
        assert _apply_beamproc_plan(engine, second_plan) is True

    assert engine.beamproc_grouped_parent_states["p0"] is root_state
    assert engine.beamproc_grouped_active_ids_by_root["p0"] == {"d0", "d1"}
    grouped_id = engine.beamproc_grouped_request_by_root["p0"]
    grouped_request = engine.beam_cache[grouped_id][0]
    assert grouped_request.prompt_token_ids == [10, 20, 100, 101, 200, 201]
    assert _beamproc_request_context(grouped_request)["step_index"] == 2
    assert root_state["step_index"] == 2
    grouped_state = engine.beamproc_grouped_output_states[grouped_id]
    assert grouped_state["child_ids"] == ["d0", "d1"]
    assert "response_token_ids" not in grouped_state


def test_grouped_rust_plan_reuses_structured_context_without_json_roundtrip() -> None:
    engine = _beamproc_plan_engine()
    parent = engine.beam_cache["p0"][0]
    parent.trace_headers = _beamproc_trace_headers_with_context(
        {"traceparent": "00-test-trace"},
        {
            "prompt_token_count": 2,
            "step_index": 0,
            "total_decode_steps": 4,
        },
    )
    root_state = _beamproc_prepare_grouped_parent_state(
        parent,
        parent._all_token_ids,
        parent.block_hashes,
    )
    engine.beamproc_grouped_parent_states = {"p0": root_state}

    with (
        patch(
            "vllm_gr.v1.engine.core._beamproc_context_from_trace_headers",
            side_effect=AssertionError("unexpected context JSON decode"),
        ),
        patch(
            "vllm_gr.v1.engine.core._beamproc_trace_headers_with_context",
            side_effect=AssertionError("unexpected context JSON encode"),
        ),
        patch("vllm.v1.request.Request", side_effect=_mock_request),
    ):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True
        second_plan = {
            **_two_child_plan(),
            "parent_ids": ["c0", "c1"],
            "child_ids": ["d0", "d1"],
            "token_ids": [101, 201],
            "beam_tokens": [[100, 101], [200, 201]],
        }
        assert _apply_beamproc_plan(engine, second_plan) is True

    grouped_id = engine.beamproc_grouped_request_by_root["p0"]
    grouped_request = engine.beam_cache[grouped_id][0]
    assert grouped_request.trace_headers is root_state["trace_headers"]
    assert grouped_request.trace_headers == {"traceparent": "00-test-trace"}
    assert grouped_request._beamproc_trace_context["step_index"] == 2
    assert grouped_request._beamproc_trace_context["is_final_step"] is False


def test_grouped_rust_plan_skips_value_comparison_for_shared_prefix() -> None:
    engine = _beamproc_plan_engine()
    parent_state = _beamproc_grouped_parent_state(engine, "p0")

    class IdentityOnlyPrefix(list[int]):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("shared prefix should use identity")

        def __ne__(self, other: object) -> bool:
            raise AssertionError("shared prefix should use identity")

    parent_state["prefix_token_ids"] = IdentityOnlyPrefix(parent_state["prefix_token_ids"])
    engine.beamproc_grouped_parent_states = {"p0": parent_state}

    with patch("vllm.v1.request.Request", side_effect=_mock_request):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True


def test_grouped_rust_plan_rejects_distinct_parent_prefixes() -> None:
    engine = _beamproc_plan_engine()
    first_state = _beamproc_grouped_parent_state(engine, "p0")
    second_state = dict(first_state)
    second_state["prefix_token_ids"] = [99]
    engine.beamproc_grouped_parent_states = {
        "p0": first_state,
        "p1": second_state,
    }
    plan = _two_child_plan()
    plan["parent_ids"] = ["p0", "p1"]

    assert _apply_beamproc_grouped_plan(engine, plan) is False


def test_grouped_parent_state_is_prepared_when_request_is_cached() -> None:
    engine = _beamproc_plan_engine()
    engine.beam_cache.clear()
    parent = SimpleNamespace(
        request_id="p0",
        _all_token_ids=[10, 20],
        prompt_token_ids=[10, 20],
        num_prompt_tokens=2,
        block_hashes=[],
        eos_token_id=2,
        lora_request=None,
        cache_salt=None,
        mm_features=None,
        prompt_embeds=None,
        trace_headers=None,
    )

    with patch.dict(
        "os.environ",
        {_BEAMPROC_ENABLED_ENV: "1"},
    ):
        _cache_beam_request(engine, parent)

    cached = engine.beam_cache[parent.request_id]
    prepared = engine.beamproc_grouped_parent_states[parent.request_id]
    assert _beamproc_grouped_parent_state(engine, parent.request_id) is prepared
    assert prepared["prefix_token_ids"] is cached["all_token_ids"]
    assert prepared["block_hashes"] is cached["block_hashes"]


def test_root_grouped_parent_state_includes_fixed_model_prefix_token() -> None:
    request = SimpleNamespace(
        _all_token_ids=[10, 20, 99, 123],
        prompt_token_ids=[10, 20, 99],
        num_prompt_tokens=3,
        is_beam_decode=False,
        trace_headers=_beamproc_trace_headers_with_context(
            None,
            {
                # API prompt length intentionally excludes fixed sid-begin 99.
                "prompt_token_count": 2,
                "step_index": 0,
            },
        ),
    )

    state = _beamproc_prepare_grouped_parent_state(request, request._all_token_ids, [])

    assert state["prefix_token_ids"] == [10, 20, 99]
    assert state["generated_token_ids"] == (123,)
    assert state["response_token_ids"] == (99, 123)
    assert state["trace_context"]["prompt_token_count"] == 2


def test_grouped_parent_state_uses_explicit_packed_prefix_boundary() -> None:
    request = SimpleNamespace(
        prompt_token_ids=[10, 20, 99, 123, 124],
        num_prompt_tokens=5,
        is_beam_decode=True,
        block_hashes=[],
        trace_headers=_beamproc_trace_headers_with_context(
            None,
            {
                "prompt_token_count": 2,
                "beam_prefix_token_count": 3,
                "step_index": 1,
            },
        ),
    )
    engine = SimpleNamespace(
        beamproc_grouped_parent_states={},
        beam_cache={"group": (request, 0)},
        beam_cache_lock=threading.Lock(),
    )

    state = _beamproc_grouped_parent_state(engine, "group")

    assert state["prefix_token_ids"] == [10, 20, 99]
    assert state["generated_token_ids"] == (123, 124)
    assert state["response_token_ids"] == (99, 123, 124)


def test_grouped_engine_output_expands_to_virtual_rust_parent_rows() -> None:
    engine = _beamproc_plan_engine()
    with patch("vllm.v1.request.Request", side_effect=_mock_request):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True

    _, requests, _ = engine.pb_queue.get_nowait()
    grouped_request = requests[0]
    group_id = grouped_request.request_id
    output = SimpleNamespace(
        request_id=group_id,
        new_token_ids=[],
        finish_reason="length",
        stop_reason=None,
        prefill_stats=SimpleNamespace(num_cached_tokens=2),
        new_logprobs=SimpleNamespace(
            logprob_token_ids=np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
            logprobs=np.asarray([[0.0, -1.0, -2.0, 0.0, -1.5, -2.5]], dtype=np.float32),
            sampled_token_ranks=np.asarray([0], dtype=np.int32),
        ),
    )

    payloads, contexts, token_frame, _, rank_frame, _ = _beamproc_batch_payload(
        engine,
        [output],
        correlation_base=0,
    )

    assert [payload[0] for payload in payloads] == ["c0", "c1"]
    assert [payload[7:9] for payload in payloads] == [[1, 3], [1, 3]]
    assert [payload[11] for payload in payloads] == [0, 0]
    assert np.frombuffer(token_frame, dtype=np.int32).tolist() == [1, 2, 3, 4, 5, 6]
    assert np.frombuffer(rank_frame, dtype=np.int32).tolist() == [0, 0]
    assert len(contexts) == 1
    assert "parent_token_ids" not in contexts[0]
    assert contexts[0]["use_shadow_parent_tokens"] is True
    assert contexts[0]["prompt_token_count"] == 0
    assert contexts[0]["root_request_id"] == "p0"


def test_final_grouped_output_releases_retained_root_state() -> None:
    engine = _beamproc_plan_engine()
    with patch("vllm.v1.request.Request", side_effect=_mock_request):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True

    group_id = engine.beamproc_grouped_request_by_root["p0"]
    engine.beamproc_grouped_output_states[group_id]["is_final_step"] = True
    output = SimpleNamespace(
        request_id=group_id,
        new_token_ids=[],
        finish_reason="length",
        stop_reason=None,
        prefill_stats=SimpleNamespace(num_cached_tokens=2),
        new_logprobs=SimpleNamespace(
            logprob_token_ids=np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
            logprobs=np.asarray(
                [[0.0, -1.0, -2.0, 0.0, -1.5, -2.5]],
                dtype=np.float32,
            ),
            sampled_token_ranks=np.asarray([0], dtype=np.int32),
        ),
    )

    _beamproc_batch_payload(engine, [output], correlation_base=0)

    assert "p0" not in engine.beamproc_grouped_parent_states
    assert "p0" not in engine.beamproc_grouped_request_by_root
    assert "p0" not in engine.beamproc_grouped_active_ids_by_root
    assert group_id not in engine.beam_cache


def test_candidate_serialization_preserves_beam_width_above_128() -> None:
    beam_width = 257
    token_ids: np.ndarray[Any, np.dtype[np.int32]] = np.arange(
        beam_width + 1, dtype=np.int32
    ).reshape(1, -1)
    logprobs: np.ndarray[Any, np.dtype[np.float32]] = np.linspace(
        0.0, -1.0, beam_width + 1, dtype=np.float32
    ).reshape(1, -1)
    flat = SimpleNamespace(
        logprob_token_ids=token_ids,
        logprobs=logprobs,
        sampled_token_ranks=np.asarray([0], dtype=np.int32),
    )

    serialized_tokens, serialized_logprobs, _ = _beamproc_logprobs_arrays(flat)

    assert serialized_tokens.shape == (1, beam_width + 1)
    assert serialized_logprobs.shape == (1, beam_width + 1)
    assert serialized_tokens[0, -1] == beam_width


def test_candidate_serialization_restores_sliced_vocab_offset() -> None:
    set_vocab_slice_config(offset=151669, slice_size=24576)
    flat = SimpleNamespace(
        logprob_token_ids=np.asarray([[0, 7, 24575, 24576, -1, 151700]], dtype=np.int32),
        logprobs=np.zeros((1, 6), dtype=np.float32),
        sampled_token_ranks=np.asarray([0], dtype=np.int32),
    )

    serialized_tokens, _, _ = _beamproc_logprobs_arrays(flat)

    assert serialized_tokens.tolist() == [[151669, 151676, 176244, 24576, -1, 151700]]


def test_grouped_candidate_serialization_restores_sliced_vocab_offset() -> None:
    set_vocab_slice_config(offset=1000, slice_size=100)
    engine = _beamproc_plan_engine()
    with patch("vllm.v1.request.Request", side_effect=_mock_request):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True

    group_id = engine.beamproc_grouped_request_by_root["p0"]
    output = SimpleNamespace(
        request_id=group_id,
        new_token_ids=[],
        finish_reason="length",
        stop_reason=None,
        prefill_stats=SimpleNamespace(num_cached_tokens=2),
        new_logprobs=SimpleNamespace(
            logprob_token_ids=np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
            logprobs=np.zeros((1, 6), dtype=np.float32),
            sampled_token_ranks=np.asarray([0], dtype=np.int32),
        ),
    )

    _, _, token_frame, _, _, _ = _beamproc_batch_payload(
        engine,
        [output],
        correlation_base=0,
    )

    assert np.frombuffer(token_frame, dtype=np.int32).tolist() == [
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
    ]


def test_candidate_serialization_reuses_buffer_for_sliced_vocab_offset() -> None:
    set_vocab_slice_config(offset=151669, slice_size=24576)
    engine = _beamproc_plan_engine()
    output = SimpleNamespace(
        request_id="p0",
        new_token_ids=[],
        finish_reason=None,
        stop_reason=None,
        prefill_stats=None,
        new_logprobs=SimpleNamespace(
            logprob_token_ids=np.asarray(
                [[0, 7, 24575, 24576, -1, 151700]],
                dtype=np.int32,
            ),
            logprobs=np.zeros((1, 6), dtype=np.float32),
            sampled_token_ranks=np.asarray([0], dtype=np.int32),
        ),
    )
    buffer_set: dict[str, Any] = {"_beamproc_tx_buffer_set": True}

    _, _, token_frame, _, _, _ = _beamproc_batch_payload(
        engine,
        [output],
        correlation_base=0,
        buffer_set=buffer_set,
    )

    assert np.frombuffer(token_frame, dtype=np.int32).tolist() == [
        151669,
        151676,
        176244,
        24576,
        -1,
        151700,
    ]
    assert buffer_set["token_id_local_mask"].dtype == np.bool_
    assert buffer_set["token_id_local_mask"].size == 6


def test_untagged_outputs_are_not_forwarded_to_beamproc() -> None:
    engine = _beamproc_plan_engine()
    output = SimpleNamespace(request_id="ordinary-request")

    payloads, contexts, token_frame, logprob_frame, rank_frame, new_token_frame = (
        _beamproc_batch_payload(
            engine,
            [output],
            correlation_base=0,
            output_contexts={},
        )
    )

    assert payloads == []
    assert contexts == []
    assert all(
        len(frame) == 0 for frame in (token_frame, logprob_frame, rank_frame, new_token_frame)
    )


def test_finished_full_plan_releases_enginecore_root_state() -> None:
    engine = _beamproc_plan_engine()
    engine.beam_cache["p0"] = object()
    engine.beamproc_grouped_request_by_root = {"p0": "group"}
    engine.beamproc_grouped_active_ids_by_root = {"p0": {"child"}}
    engine.beamproc_grouped_response_prefix_by_root = {"p0": 1}
    engine.beamproc_grouped_parent_states = {"p0": object()}
    engine.beamproc_grouped_output_states = {"group": {"root_request_id": "p0"}}
    engine.beam_cache.update({"group": object(), "child": object()})

    _handle_beamproc_plan(
        engine,
        {
            "action": "observed_full",
            "root_request_id": "p0",
            "finished": True,
        },
    )

    assert engine.beam_cache == {}
    assert engine.beamproc_grouped_request_by_root == {}
    assert engine.beamproc_grouped_active_ids_by_root == {}
    assert engine.beamproc_grouped_response_prefix_by_root == {}
    assert engine.beamproc_grouped_parent_states == {}
    assert engine.beamproc_grouped_output_states == {}


def test_finished_rust_plan_frame_releases_state_before_final_decode_step(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv(_BEAMPROC_ENABLED_ENV, "1")
    engine = _beamproc_plan_engine()
    engine.beam_cache["group"] = object()
    engine.beam_cache["child"] = object()
    engine.beamproc_grouped_request_by_root = {"p0": "group"}
    engine.beamproc_grouped_active_ids_by_root = {"p0": {"child"}}
    engine.beamproc_grouped_response_prefix_by_root = {"p0": 1}
    engine.beamproc_grouped_parent_states = {"p0": object()}
    engine.beamproc_grouped_output_states = {
        "group": {"root_request_id": "p0", "is_final_step": False}
    }
    metadata = msgspec.msgpack.encode(
        {
            "version": 3,
            "action": "observed_full",
            "root_request_id": "p0",
            "request_id": "group",
            "beam_width": 2,
            "finished": True,
            "token_ids_len": 0,
            "scores_len": 0,
            "flat_indices_len": 0,
            "beam_tokens_len": 0,
            "beam_token_offsets_len": 3,
        }
    )

    def frame(value: bytes) -> SimpleNamespace:
        return SimpleNamespace(buffer=memoryview(value))

    _process_beamproc_plan_frames(
        engine,
        [
            frame(b"BEAM_PLAN"),
            frame(metadata),
            frame(b""),
            frame(b""),
            frame(b""),
            frame(b""),
            frame(np.asarray([0, 0, 0], dtype="<u8").tobytes()),
        ],
    )
    _drain_pb_queue(engine)

    assert engine.beam_cache == {}
    assert engine.beamproc_grouped_request_by_root == {}
    assert engine.beamproc_grouped_active_ids_by_root == {}
    assert engine.beamproc_grouped_output_states == {}


def test_handle_beamproc_plan_accepts_explicit_none_lists() -> None:
    _handle_beamproc_plan(
        SimpleNamespace(),
        {
            "action": "observed",
            "parent_ids": None,
            "token_ids": None,
        },
    )


def test_beamproc_children_are_admitted_before_normal_input() -> None:
    engine = SimpleNamespace(
        pb_queue=queue.SimpleQueue(),
        beamproc_wake_lock=threading.Lock(),
        beamproc_wake_pending=True,
    )
    child0 = SimpleNamespace(request_id="c0")
    child1 = SimpleNamespace(request_id="c1")
    engine.pb_queue.put_nowait((_BEAMPROC_CHILD_BATCH_REQUEST, [child0, child1], 7))
    calls: list[tuple[str, Any, Any]] = []
    engine.add_request = lambda req, wave: calls.append(("child", req.request_id, wave))

    def handle_normal(_self: Any, request_type: Any, request: Any) -> None:
        calls.append(("normal", request_type, request))

    handle_request = make_patched_handle_client_request(handle_normal)
    normal_request = (SimpleNamespace(request_id="new0"), 0)
    handle_request(engine, EngineCoreRequestType.ADD, normal_request)

    assert calls == [
        ("child", "c0", 7),
        ("child", "c1", 7),
        ("normal", EngineCoreRequestType.ADD, normal_request),
    ]
    assert engine.pb_queue.empty()

    handle_request(engine, _BEAMPROC_WAKE_REQUEST, None)
    assert engine.beamproc_wake_pending is False


def test_pb_queue_drains_children_and_metadata() -> None:
    engine = SimpleNamespace(pb_queue=queue.SimpleQueue())
    child = SimpleNamespace(request_id="c0")
    metadata = {"action": "observed", "parent_ids": []}
    engine.pb_queue.put_nowait((_BEAMPROC_CHILD_BATCH_REQUEST, [child], 7))
    engine.pb_queue.put_nowait(metadata)
    admitted = []
    engine.add_request = lambda req, wave: admitted.append((req.request_id, wave))

    with patch("vllm_gr.v1.engine.core._handle_beamproc_plan") as handle_plan:
        assert _drain_pb_queue(engine) == 1

    assert admitted == [("c0", 7)]
    handle_plan.assert_called_once_with(engine, metadata)
    assert engine.pb_queue.empty()


def test_pb_queue_drains_two_tuple_child_batch_envelope() -> None:
    engine = SimpleNamespace(pb_queue=queue.SimpleQueue())
    child0 = SimpleNamespace(request_id="c0")
    child1 = SimpleNamespace(request_id="c1")
    engine.pb_queue.put_nowait(
        (
            _BEAMPROC_CHILD_BATCH_REQUEST,
            [(child0, 7), (child1, 8)],
        )
    )
    admitted = []
    engine.add_request = lambda req, wave: admitted.append((req.request_id, wave))

    assert _drain_pb_queue(engine) == 2
    assert admitted == [("c0", 7), ("c1", 8)]
    assert engine.pb_queue.empty()


def test_apply_beamproc_plan_enqueues_child_batch_on_priority_queue() -> None:
    engine = _beamproc_plan_engine()

    with patch(
        "vllm_gr.v1.engine.core._apply_beamproc_grouped_plan",
        return_value=False,
    ):
        assert _apply_beamproc_plan(engine, _two_child_plan()) is True

    request_type, children, current_wave = engine.pb_queue.get_nowait()
    assert request_type is _BEAMPROC_CHILD_BATCH_REQUEST
    assert [child.request_id for child in children] == ["c0", "c1"]
    assert current_wave == 7
    assert engine.pb_queue.empty()
    assert engine.input_queue.empty()
