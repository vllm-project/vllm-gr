# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts for NPU Beam graph dispatch decisions."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests._npu_model_runner_patch_helpers import (
    _identity_layout_runner,
    _make_beam_info,
    _make_determine_wrapper,
)
from vllm_gr.v1.worker import npu_model_runner_capture as capture_mod


def test_npu_determine_no_beam_info_passthrough() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=16))
    runner = SimpleNamespace(_current_beam_info=None)
    wrapped = _make_determine_wrapper(native)
    result = wrapped(runner, num_tokens=16)
    assert result[0] == CUDAGraphMode.NONE
    assert result[1].num_tokens == 16


def test_npu_determine_session_missing_downgrades_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    beam_info: dict[str, Any] = {
        "beam_data": {"req-1": {"is_beam_decode": True, "decode_step": 0}},
        "req_ids": ["req-1"],
        # No "beam_session" key
    }
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True))
    runner = SimpleNamespace(_current_beam_info=beam_info)

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    capture_mod._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm


def test_npu_determine_mixed_batch_downgrades_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True)
    beam_info = _make_beam_info(session, ["beam-1", "std-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=8, uniform=True))
    runner = SimpleNamespace(_current_beam_info=beam_info)

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    capture_mod._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=8)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm


def test_npu_determine_nonzero_slot_downgrades_full() -> None:
    """The captured graph bakes pool slot 0; other slots must not replay it."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=1)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True))
    runner = SimpleNamespace(_current_beam_info=beam_info)

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    capture_mod._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm


def test_npu_determine_cascade_mode_downgrades_full() -> None:
    """Only the xllm execution mode is graph-safe; cascade must go eager."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    orig_enabled = capture_mod._beam_graph_enabled_for_run
    capture_mod._npu_xllm_decode_mode_active = lambda: False
    capture_mod._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm
        capture_mod._beam_graph_enabled_for_run = orig_enabled


def test_npu_determine_pure_beam_batch_grants_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(
        is_active=True, slot=0, batch_size=1, request_width=4, attention_width=4
    )
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=4))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
        _gr_npu_beam_graph_shapes={(1, 4)},
        _current_scheduler_output=SimpleNamespace(num_scheduled_tokens={"beam-1": 4}),
        query_start_loc=SimpleNamespace(np=np.array([0, 4], dtype=np.int32)),
    )

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    orig_enabled = capture_mod._beam_graph_enabled_for_run
    capture_mod._npu_xllm_decode_mode_active = lambda: True
    capture_mod._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.FULL
        assert result[1].num_tokens == 4
        assert result[1].uniform is True
        assert runner._beam_graph_full_dispatch_count == 1
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm
        capture_mod._beam_graph_enabled_for_run = orig_enabled


@pytest.mark.parametrize("native_mode", ["NONE", "PIECEWISE", "FULL"])  # type: ignore[misc]
def test_npu_determine_same_token_count_wrong_shape_falls_back(native_mode: str) -> None:
    """B*W equality alone must not replay a graph captured for another shape."""

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(
        is_active=True,
        slot=0,
        batch_size=4,
        request_width=2,
        attention_width=8,
    )
    request_ids = [f"beam-{index}" for index in range(4)]
    beam_info = _make_beam_info(session, request_ids, request_ids)
    native = (getattr(CUDAGraphMode, native_mode), BatchDescriptor(num_tokens=8), "native-extra")
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={8},
        _gr_npu_beam_graph_shapes={(2, 4)},
        _current_scheduler_output=SimpleNamespace(
            num_scheduled_tokens=dict.fromkeys(request_ids, 2)
        ),
        query_start_loc=SimpleNamespace(np=np.arange(0, 9, 2, dtype=np.int32)),
    )

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    orig_enabled = capture_mod._beam_graph_enabled_for_run
    capture_mod._npu_xllm_decode_mode_active = lambda: True
    capture_mod._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=8)
        expected_mode = (
            CUDAGraphMode.PIECEWISE if native_mode == "PIECEWISE" else CUDAGraphMode.NONE
        )
        assert result[0] == expected_mode
        assert result[1].num_tokens == 8 and result[2:] == native[2:]
        if native_mode != "FULL":
            assert result is native
        assert runner._beam_graph_fallback_count == 1
        assert runner._beam_graph_fallback_reasons == {"shape_miss": 1}
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm
        capture_mod._beam_graph_enabled_for_run = orig_enabled


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
def test_npu_identity_layout_gate_accepts_physical_request_order(batch_size: int) -> None:
    runner = _identity_layout_runner(batch_size)
    assert capture_mod._npu_beam_identity_layout_matches(
        runner, runner._current_beam_info, batch_size * 4
    )


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("native_mode", ["NONE", "PIECEWISE", "FULL"])  # type: ignore[misc]
def test_npu_determine_attributes_uncaptured_nonidentity_batch_to_shape_miss(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, native_mode: str
) -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    runner = _identity_layout_runner(batch_size)
    runner._gr_npu_beam_graph_shapes.clear()
    beam_info = runner._current_beam_info
    cached_prefix = beam_info["beam_data"]["beam-0"]["cache_len"]
    beam_info["beam_data"]["beam-0"]["cache_len"] = 0
    tokens = batch_size * 4
    assert not capture_mod._npu_beam_identity_layout_matches(runner, beam_info, tokens)
    native = (getattr(CUDAGraphMode, native_mode), BatchDescriptor(num_tokens=tokens), "extra")
    monkeypatch.setattr(capture_mod, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(capture_mod, "_beam_graph_enabled_for_run", lambda _runner: True)

    wrapped = _make_determine_wrapper(native)
    result = wrapped(runner, num_tokens=tokens)
    expected_mode = CUDAGraphMode.PIECEWISE if native_mode == "PIECEWISE" else CUDAGraphMode.NONE
    assert result[0] == expected_mode
    assert result[1].num_tokens == tokens and result[2:] == native[2:]
    if native_mode != "FULL":
        assert result is native
    assert runner._beam_graph_fallback_count == 1
    assert runner._beam_graph_fallback_reasons == {"shape_miss": 1}
    assert getattr(runner, "_beam_graph_full_dispatch_count", 0) == 0

    runner._gr_npu_beam_graph_shapes.add((batch_size, 4))
    assert wrapped(runner, num_tokens=tokens)[0] == expected_mode
    assert runner._beam_graph_fallback_count == 2
    assert runner._beam_graph_fallback_reasons == {"shape_miss": 1, "identity_layout": 1}

    beam_info["beam_data"]["beam-0"]["cache_len"] = cached_prefix
    assert wrapped(runner, num_tokens=tokens)[0] == CUDAGraphMode.FULL
    assert runner._beam_graph_full_dispatch_count == 1
    assert runner._beam_graph_fallback_count == 2


@pytest.mark.parametrize("native_mode", ["NONE", "PIECEWISE", "FULL"])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "invalid_layout",
    [
        "missing-schedule",
        "logical-counts",
        "unequal-counts",
        "request-order",
        "cached-prefix-lost",
        "wrong-query-starts",
        "missing-query-mirror",
        "wrong-attention-width",
        "boolean-decode-step",
        "missing-decode-rows",
        "inconsistent-decode-step",
    ],
)
def test_npu_determine_rejects_nonidentity_layout_at_same_shape(
    monkeypatch: pytest.MonkeyPatch, native_mode: str, invalid_layout: str
) -> None:
    """An exact B/W bucket must not replay a graph with baked identity views."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    runner = _identity_layout_runner()
    beam_info = runner._current_beam_info
    if invalid_layout == "missing-schedule":
        runner._current_scheduler_output = None
    elif invalid_layout == "logical-counts":
        runner._current_scheduler_output.num_scheduled_tokens = {"beam-0": 1, "beam-1": 1}
    elif invalid_layout == "unequal-counts":
        runner._current_scheduler_output.num_scheduled_tokens = {"beam-0": 3, "beam-1": 5}
    elif invalid_layout == "request-order":
        runner._current_scheduler_output.num_scheduled_tokens = {"beam-1": 4, "beam-0": 4}
    elif invalid_layout == "cached-prefix-lost":
        beam_info["beam_data"]["beam-0"]["cache_len"] = 0
    elif invalid_layout == "wrong-query-starts":
        runner.query_start_loc.np = np.array([0, 3, 8], dtype=np.int32)
    elif invalid_layout == "missing-query-mirror":
        runner.query_start_loc = None
    elif invalid_layout == "wrong-attention-width":
        beam_info["beam_session"].attention_width = 4
    elif invalid_layout == "boolean-decode-step":
        beam_info["beam_data"]["beam-0"]["decode_step"] = True
    elif invalid_layout == "missing-decode-rows":
        beam_info["beam_data"]["beam-0"]["beam_decode_steps"] = 0
    elif invalid_layout == "inconsistent-decode-step":
        beam_info["beam_data"]["beam-0"]["beam_decode_steps"] = 2

    native = (getattr(CUDAGraphMode, native_mode), BatchDescriptor(num_tokens=8), "extra")
    monkeypatch.setattr(capture_mod, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(capture_mod, "_beam_graph_enabled_for_run", lambda _runner: True)
    result = _make_determine_wrapper(native)(runner, num_tokens=8)
    expected_mode = CUDAGraphMode.PIECEWISE if native_mode == "PIECEWISE" else CUDAGraphMode.NONE
    assert result[0] == expected_mode
    assert result[1].num_tokens == 8 and result[2:] == native[2:]
    assert runner._beam_graph_fallback_reasons == {"identity_layout": 1}


def test_npu_determine_does_not_reuse_identity_grant_after_prefix_layout_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    runner = _identity_layout_runner()
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=8))
    monkeypatch.setattr(capture_mod, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(capture_mod, "_beam_graph_enabled_for_run", lambda _runner: True)
    wrapped = _make_determine_wrapper(native)
    assert wrapped(runner, num_tokens=8)[0] == CUDAGraphMode.FULL
    runner._current_beam_info["beam_data"]["beam-0"]["cache_len"] = 0
    assert wrapped(runner, num_tokens=8)[0] == CUDAGraphMode.NONE
    assert runner._beam_graph_full_dispatch_count == 1
    assert runner._beam_graph_fallback_reasons == {"identity_layout": 1}


def test_npu_determine_active_lora_downgrades_full() -> None:
    """Active-LoRA beam batches must run eager: the beam graph is captured
    only for the no-LoRA descriptor, and the FULL rewrite would drop
    has_lora/num_active_loras (silently routing to the LoRA-free graph)."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True, has_lora=True))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    orig_enabled = capture_mod._beam_graph_enabled_for_run
    capture_mod._npu_xllm_decode_mode_active = lambda: True
    capture_mod._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm
        capture_mod._beam_graph_enabled_for_run = orig_enabled


def test_npu_determine_graph_disabled_no_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=4))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    orig_enabled = capture_mod._beam_graph_enabled_for_run
    capture_mod._npu_xllm_decode_mode_active = lambda: True
    capture_mod._beam_graph_enabled_for_run = lambda r: False
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm
        capture_mod._beam_graph_enabled_for_run = orig_enabled


def test_npu_determine_width_mismatch_no_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0, batch_size=1, request_width=4)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=8))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},  # captured at 4, batch is 8 tokens
        _gr_npu_beam_graph_shapes={(1, 4)},
    )

    orig_xllm = capture_mod._npu_xllm_decode_mode_active
    orig_enabled = capture_mod._beam_graph_enabled_for_run
    capture_mod._npu_xllm_decode_mode_active = lambda: True
    capture_mod._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=8)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        capture_mod._npu_xllm_decode_mode_active = orig_xllm
        capture_mod._beam_graph_enabled_for_run = orig_enabled


pytestmark = pytest.mark.cpu_test
