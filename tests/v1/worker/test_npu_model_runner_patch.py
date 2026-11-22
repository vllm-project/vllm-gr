# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Core contracts for NPU ModelRunner wrappers."""

import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import NewRequestData

from tests.helpers.gr_config import table_constraint
from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.v1.attention.backends.npu_cache_layout import restore_npu_cache_pool_layout
from vllm_gr.v1.beam.execution_batch import BeamExecutionContractError
from vllm_gr.v1.worker import gr_context as _gr_context
from vllm_gr.v1.worker import gr_execution as _gr_dispatch
from vllm_gr.v1.worker.npu import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.npu.beam_execution_batch import NPUBeamExecutionBatch
from vllm_gr.v1.worker.npu.npu_beam_stage_adapter import (
    NPUEagerStagePlan,
    validate_npu_eager_capabilities,
)
from vllm_gr.v1.worker.npu.npu_beam_stage_runner import NPUEagerBeamStageRunner
from vllm_gr.v1.worker.npu.npu_model_runner_inputs import InputBatchProxy


def _eager_stage_output(
    *,
    kind: str,
    batch_size: int = 2,
    beam_width: int = 3,
    decode_step: int = 0,
) -> SimpleNamespace:
    request_ids = tuple(f"request-{index}" for index in range(batch_size))
    prompt_lengths = tuple(range(2, batch_size + 2))
    stage_index = 0 if kind == "prefill" else decode_step + 1
    metadata = SimpleNamespace(
        session_id="batch-0",
        dispatch_id=stage_index,
        stage_index=stage_index,
        kind=kind,
        decode_step=None if kind == "prefill" else decode_step,
        owner_generation=1,
        request_ids=request_ids,
        num_prompt_tokens_by_request=prompt_lengths,
        num_prompt_tokens=sum(prompt_lengths),
        beam_params=SimpleNamespace(beam_width=beam_width),
        produces_output=True,
        execution_capacity=SimpleNamespace(beam_width=beam_width),
    )
    counts = prompt_lengths if kind == "prefill" else (1,) * batch_size
    return SimpleNamespace(
        gr_stage_metadata={metadata.session_id: metadata},
        num_scheduled_tokens=dict(zip(request_ids, counts)),
        total_num_scheduled_tokens=sum(counts),
        beam_data={},
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )


def test_npu_eager_plan_keeps_logical_prefill_and_expands_decode_rows() -> None:
    prefill = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="prefill"))
    assert prefill.logical_scheduled_tokens == (2, 3)
    assert prefill.model_token_count == 5
    assert prefill.logits_indices == (1, 4)
    assert prefill.sampling_row_indices == (0, 1)

    decode = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="decode"))
    assert decode.logical_scheduled_tokens == (1, 1)
    assert decode.model_token_count == 6
    assert decode.logits_indices == tuple(range(6))
    assert decode.sampling_row_indices == (0, 0, 0, 1, 1, 1)
    assert decode.decode_positions == (2, 2, 2, 3, 3, 3)
    next_decode = NPUEagerStagePlan.from_scheduler_output(
        _eager_stage_output(kind="decode", decode_step=1)
    )
    assert next_decode.decode_positions == (3, 3, 3, 4, 4, 4)
    assert next_decode.execution.stage_index == 2
    assert next_decode.execution.decode_step == 1


def _cached_prefill_output(cached: tuple[int, ...]) -> SimpleNamespace:
    scheduled = _eager_stage_output(kind="prefill", batch_size=len(cached))
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    lengths = tuple(257 + 128 * index for index in range(len(cached)))
    metadata.num_prompt_tokens_by_request = lengths
    metadata.num_prompt_tokens = sum(lengths)
    scheduled.scheduled_new_reqs = [
        NewRequestData(
            req_id=request_id,
            prompt_token_ids=[1] * length,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([index],),
            num_computed_tokens=computed,
            lora_request=None,
        )
        for index, (request_id, length, computed) in enumerate(
            zip(metadata.request_ids, lengths, cached, strict=True)
        )
    ]
    scheduled.num_scheduled_tokens = {
        request.req_id: length - request.num_computed_tokens
        for request, length in zip(scheduled.scheduled_new_reqs, lengths, strict=True)
    }
    scheduled.total_num_scheduled_tokens = sum(scheduled.num_scheduled_tokens.values())
    return scheduled


@pytest.mark.parametrize(  # type: ignore[misc]
    ("cached", "expected_logits"),
    [
        ((0,), (256,)),
        ((128,), (128,)),
        ((256,), (0,)),
        ((0, 0), (256, 641)),
        ((128, 0), (128, 513)),
        ((256, 384), (0, 1)),
        ((128, 0, 512, 256), (128, 513, 514, 899)),
        ((256, 384, 512, 640), (0, 1, 2, 3)),
    ],
)
def test_npu_prefill_cache_hit_uses_scheduled_logits_and_full_decode_positions(
    cached: tuple[int, ...], expected_logits: tuple[int, ...]
) -> None:
    scheduled = _cached_prefill_output(cached)
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    assert plan.logits_indices == expected_logits
    assert plan.model_token_count == expected_logits[-1] + 1
    assert plan.logical_scheduled_tokens == tuple(scheduled.num_scheduled_tokens.values())
    assert tuple(r.num_computed_tokens for r in scheduled.scheduled_new_reqs) == cached

    # Cached tokens contribute to Decode positions, but not Prefill hidden rows.
    decode = _eager_stage_output(kind="decode", batch_size=len(cached))
    metadata = next(iter(decode.gr_stage_metadata.values()))
    metadata.num_prompt_tokens_by_request = plan.execution.num_prompt_tokens_by_request
    metadata.num_prompt_tokens = sum(metadata.num_prompt_tokens_by_request)
    decode_plan = NPUEagerStagePlan.from_scheduler_output(decode)
    winners = torch.arange(len(cached) * 3, dtype=torch.int32).view(len(cached), 3)
    input_ids = torch.empty(len(cached) * 3, dtype=torch.int32)
    positions = torch.empty_like(input_ids)
    decode_plan.prepare_decode_inputs(input_ids, positions, winners)
    assert input_ids.tolist() == winners.flatten().tolist()
    assert positions.tolist() == [
        length for length in plan.execution.num_prompt_tokens_by_request for _ in range(3)
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    "fault",
    ["chunk", "missing_progress", "short", "long", "negative", "noninteger", "boolean"],
)
def test_npu_prefill_rejects_incomplete_or_invalid_cached_progress(fault: str) -> None:
    scheduled = _cached_prefill_output((128, 256))
    request = scheduled.scheduled_new_reqs[0]
    if fault == "chunk":
        next(iter(scheduled.gr_stage_metadata.values())).produces_output = False
    elif fault == "missing_progress":
        scheduled.scheduled_new_reqs.clear()
    elif fault in ("short", "long"):
        scheduled.num_scheduled_tokens[request.req_id] += -1 if fault == "short" else 1
        scheduled.total_num_scheduled_tokens = sum(scheduled.num_scheduled_tokens.values())
    else:
        setattr(
            request,
            "num_computed_tokens",
            {"negative": -1, "noninteger": 128.0, "boolean": True}[fault],
        )
    with pytest.raises(BeamExecutionContractError):
        NPUEagerStagePlan.from_scheduler_output(scheduled)


def test_npu_prefill_preparation_preserves_cached_geometry() -> None:
    scheduled = _cached_prefill_output((128, 0, 512, 256))
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    counts = np.asarray([129, 385, 1, 385], dtype=np.int32)
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _current_scheduler_output=scheduled,
        _current_beam_info={"req_ids": []},
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.pending_plan = worker, plan
    hidden = torch.arange(900)
    spec_metadata = object()

    def native(self: Any, output: Any, native_counts: Any) -> tuple[Any, ...]:
        assert self is worker and output is scheduled and native_counts is counts
        assert tuple(r.num_computed_tokens for r in output.scheduled_new_reqs) == (128, 0, 512, 256)
        return object(), spec_metadata, 900

    logits, spec, total = stage.prepare_inputs(native, scheduled, counts)
    assert hidden[logits.long()].tolist() == [128, 513, 514, 899]
    assert spec is spec_metadata and total == 900
    assert counts.tolist() == [129, 385, 1, 385]


def test_npu_prepare_unproxies_and_preserves_native_contract() -> None:
    input_batch = SimpleNamespace(req_ids=["request"])
    proxy = InputBatchProxy(input_batch, ["request"], [0], [0], device=torch.device("cpu"))
    spec_metadata = object()
    runner = SimpleNamespace(
        input_batch=proxy,
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )

    def original(
        self: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[object, object, int]:
        assert self.input_batch is input_batch
        assert self._current_beam_info["req_ids"] is input_batch.req_ids
        return object(), spec_metadata, 7

    wrapped = npu_patch.make_patched_prepare_inputs(original)
    assert npu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary[1:] == (spec_metadata, 7)


def _make_determine_wrapper(native_result: Any) -> Any:
    """Build a patched NPU determine with a stubbed native _orig_determine."""

    def _orig_determine(self: Any, *args: Any, **kwargs: Any) -> Any:
        return native_result

    return npu_patch.make_patched_determine_batch_execution_and_padding_npu(_orig_determine)


def _make_beam_info(session: Any, req_ids: list[str], beam_keys: list[str]) -> dict[str, Any]:
    width = int(getattr(session, "request_width", getattr(session, "beam_width", 4)))
    beam_data = {
        req_id: (
            {
                "is_beam_decode": True,
                "decode_step": 0,
                "beam_decode_steps": 1,
                "beam_width": width,
                "prefix_len": 128,
                "cache_len": 128,
            }
            if req_id in beam_keys
            else {}
        )
        for req_id in req_ids
    }
    return {
        "beam_data": beam_data,
        "req_ids": req_ids,
        "beam_session": session,
    }


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

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm


def test_npu_determine_mixed_batch_downgrades_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True)
    beam_info = _make_beam_info(session, ["beam-1", "std-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=8, uniform=True))
    runner = SimpleNamespace(_current_beam_info=beam_info)

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=8)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm


def test_npu_determine_nonzero_slot_downgrades_full() -> None:
    """The captured graph bakes pool slot 0; other slots must not replay it."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=1)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True))
    runner = SimpleNamespace(_current_beam_info=beam_info)

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm


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

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: False
    npu_patch._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


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

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    npu_patch._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.FULL
        assert result[1].num_tokens == 4
        assert result[1].uniform is True
        assert runner._beam_graph_full_dispatch_count == 1
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


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

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    npu_patch._beam_graph_enabled_for_run = lambda r: True
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
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


def _identity_layout_runner(batch_size: int = 2, width: int = 4) -> SimpleNamespace:
    request_ids = [f"beam-{index}" for index in range(batch_size)]
    session = SimpleNamespace(
        is_active=True,
        slot=0,
        batch_size=batch_size,
        request_width=width,
        attention_width=batch_size * width,
    )
    return SimpleNamespace(
        _current_beam_info=_make_beam_info(session, request_ids, request_ids),
        _beam_graph_keys={batch_size * width},
        _gr_npu_beam_graph_shapes={(batch_size, width)},
        _current_scheduler_output=SimpleNamespace(
            num_scheduled_tokens=dict.fromkeys(request_ids, width)
        ),
        query_start_loc=SimpleNamespace(
            np=np.arange(0, (batch_size + 1) * width, width, dtype=np.int32)
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
def test_npu_identity_layout_gate_accepts_physical_request_order(batch_size: int) -> None:
    runner = _identity_layout_runner(batch_size)
    assert npu_patch._npu_beam_identity_layout_matches(
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
    assert not npu_patch._npu_beam_identity_layout_matches(runner, beam_info, tokens)
    native = (getattr(CUDAGraphMode, native_mode), BatchDescriptor(num_tokens=tokens), "extra")
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(npu_patch, "_beam_graph_enabled_for_run", lambda _runner: True)

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
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(npu_patch, "_beam_graph_enabled_for_run", lambda _runner: True)
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
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(npu_patch, "_beam_graph_enabled_for_run", lambda _runner: True)
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

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    npu_patch._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


def test_npu_determine_graph_disabled_no_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=4))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    npu_patch._beam_graph_enabled_for_run = lambda r: False
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


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

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    npu_patch._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=8)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


# ---------------------------------------------------------------------------
# capture_beam_decode_graph — entry guards
# ---------------------------------------------------------------------------


def _stub_gr_config(monkeypatch: pytest.MonkeyPatch, gr_cfg: Any) -> None:
    monkeypatch.setattr(_gr_context, "_get_gr_config", lambda r: gr_cfg)


def test_npu_capture_disabled_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(monkeypatch, SimpleNamespace(beam_graph_active=False))
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False)
    )
    assert npu_patch.capture_beam_decode_graph(runner) is False


def test_npu_capture_cascade_mode_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the graph-safe xllm execution mode, capture must refuse."""
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: False)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False)
    )
    assert npu_patch.capture_beam_decode_graph(runner) is False


def test_npu_capture_tp_gt1_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, use_ubatching=False),
    )
    assert npu_patch.capture_beam_decode_graph(runner) is False


def test_npu_capture_ubatching_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=True),
    )
    assert npu_patch.capture_beam_decode_graph(runner) is False


def _stub_beam_graph_install(
    monkeypatch: pytest.MonkeyPatch, ready_widths: tuple[int, ...] = ()
) -> list[Any]:
    import sys
    from types import ModuleType

    calls: list[Any] = []
    module = ModuleType("vllm_gr.v1.worker.npu.npu_beam_decode_graph")

    def install(runner: Any) -> Any:
        calls.append(runner)
        model = SimpleNamespace(graph_ready=lambda width: width in ready_widths)
        runner._gr_npu_graph_model = runner.model = model
        return model

    module.install_npu_beam_graph_model = install  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.mark.parametrize("graph_widths", [None, (4,)])  # type: ignore[misc]
def test_npu_capture_reentry_returns_true(
    monkeypatch: pytest.MonkeyPatch, graph_widths: tuple[int, ...] | None
) -> None:
    """If a graph for the width is already recorded, capture short-circuits True."""
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(
            beam_graph_active=True,
            beam_max_width=4,
            beam_max_decode_steps=3,
            beam_graph_widths=graph_widths,
        ),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    installs = _stub_beam_graph_install(monkeypatch, ready_widths=(4,))
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_graph_keys={4},
        _gr_npu_beam_graph_shapes={(1, 4)},
    )
    assert npu_patch.capture_beam_decode_graph(runner) is True
    assert installs == [runner]
    assert runner.model is runner._gr_npu_graph_model


@pytest.mark.parametrize("graph_widths", [None, (4,)])  # type: ignore[misc]
def test_npu_capture_forwards_run_under_inference_mode(
    monkeypatch: pytest.MonkeyPatch, graph_widths: tuple[int, ...] | None
) -> None:
    """Warmup + capture forwards must run inside torch.inference_mode().

    The ACL graph pool holds inference tensors created by the native
    ``_dummy_run`` (decorated with ``@torch.inference_mode()``).  Running the
    capture forward outside inference mode makes ``capture_begin``'s in-place
    pool updates raise "Inplace update to inference tensor outside
    InferenceMode is not allowed".
    """
    import contextlib

    import vllm.compilation.monitor as monitor

    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(
            beam_graph_active=True,
            beam_graph_batch_size=2,
            beam_max_width=4,
            beam_max_decode_steps=3,
            beam_graph_widths=graph_widths,
        ),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(monitor, "set_cudagraph_capturing_enabled", lambda *_: None)

    side_stream = SimpleNamespace(wait_stream=lambda *_: None)
    fake_npu = SimpleNamespace(
        Stream=lambda: side_stream,
        current_stream=lambda: SimpleNamespace(wait_stream=lambda *_: None),
        stream=lambda _side: contextlib.nullcontext(),
        synchronize=lambda: None,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    dummy_sessions = tuple(SimpleNamespace(session_id=f"dummy-{index}") for index in range(2))
    attention_batch = SimpleNamespace(
        sessions=dummy_sessions,
        end_session=lambda: None,
    )
    create_args: list[tuple[Any, ...]] = []

    def fake_create(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        create_args.append(args)
        return object(), attention_batch, object(), object()

    monkeypatch.setattr(
        npu_patch,
        "_capture_create_dummy_context_npu",
        fake_create,
    )

    installs = _stub_beam_graph_install(monkeypatch)
    inference_flags: list[bool] = []

    def fake_forward(*args: Any, **kwargs: Any) -> None:
        inference_flags.append(torch.is_inference_mode_enabled())

    monkeypatch.setattr(npu_patch, "_dummy_beam_forward_npu", fake_forward)
    monkeypatch.setattr(
        npu_patch, "_verify_and_record_captured_graph_npu", lambda r, width: width == 8
    )

    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_sessions={},
        _current_beam_info=None,
    )
    assert npu_patch.capture_beam_decode_graph(runner) is True
    assert installs == [runner]
    assert create_args[0][2] == 2
    assert runner._gr_npu_beam_graph_shapes == {(2, 4)}
    assert len(inference_flags) == npu_patch._BEAM_CAPTURE_NUM_WARMUPS + 1
    assert all(inference_flags), f"capture forwards ran outside inference_mode: {inference_flags}"


# ---------------------------------------------------------------------------
# _verify_and_record_captured_graph_npu — ACL entry verification
# ---------------------------------------------------------------------------
def test_npu_verify_and_record_records_on_match() -> None:
    beam_model = SimpleNamespace(graph_ready=lambda width: width == 4)
    runner = SimpleNamespace(model=beam_model, _gr_npu_graph_model=beam_model)
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=4) is True
    assert runner._beam_graph_keys == {4}


@pytest.mark.parametrize("owner_state", ["absent", "replaced", "uncaptured"])  # type: ignore[misc]
def test_npu_verify_and_record_rejects_unowned_or_missing_graph(owner_state: str) -> None:
    # A matching descriptor elsewhere must never establish Beam ownership.
    ordinary = SimpleNamespace(concrete_aclgraph_entries={4: object()})
    runner = SimpleNamespace(model=ordinary)
    if owner_state != "absent":
        beam_model = SimpleNamespace(graph_ready=lambda width: owner_state != "uncaptured")
        runner._gr_npu_graph_model = beam_model
        if owner_state == "uncaptured":
            runner.model = beam_model
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=4) is False
    assert not hasattr(runner, "_beam_graph_keys")


def test_npu_verify_and_record_appends_to_existing_keys() -> None:
    beam_model = SimpleNamespace(graph_ready=lambda width: width == 8)
    runner = SimpleNamespace(model=beam_model, _gr_npu_graph_model=beam_model, _beam_graph_keys={4})
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=8) is True
    assert runner._beam_graph_keys == {4, 8}


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.parametrize("first_path", ["ordinary", "beam"])  # type: ignore[misc]
def test_npu_beam_and_ordinary_graphs_isolate_matching_descriptors(
    monkeypatch: pytest.MonkeyPatch, first_path: str
) -> None:
    acl_graph = pytest.importorskip(
        "vllm_ascend.compilation.acl_graph", reason="vllm-ascend is not installed"
    )
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    from vllm_gr.v1.beam import BEAM_DATA_VAR
    from vllm_gr.v1.worker.npu.npu_beam_decode_graph import install_npu_beam_graph_model

    descriptor = BatchDescriptor(num_tokens=4, uniform=True)

    def native_init(wrapper: Any, runnable: Any, config: Any, mode: Any, **kwargs: Any) -> None:
        wrapper.runnable = runnable
        wrapper.runtime_mode = mode
        wrapper.concrete_aclgraph_entries = {}
        wrapper.update_stream = kwargs["update_stream"]
        wrapper.enable_enpu = kwargs["enable_enpu"]

    def native_forward(wrapper: Any) -> Any:
        entries = wrapper.concrete_aclgraph_entries
        if descriptor not in entries:
            entries[descriptor] = SimpleNamespace(aclgraph=object())
        return entries[descriptor].aclgraph

    monkeypatch.setattr(acl_graph.ACLGraphWrapper, "__init__", native_init)
    monkeypatch.setattr(acl_graph.ACLGraphWrapper, "__call__", native_forward)
    config = object()
    stream = object()
    ordinary = acl_graph.ACLGraphWrapper(
        lambda: None, config, CUDAGraphMode.FULL, update_stream=stream, enable_enpu=True
    )
    runner = SimpleNamespace(
        model=ordinary, vllm_config=config, update_stream=stream, enable_enpu=True
    )
    beam = install_npu_beam_graph_model(runner)
    assert beam.update_stream is stream and beam.enable_enpu
    assert install_npu_beam_graph_model(runner) is beam

    def forward(path: str) -> Any:
        token = BEAM_DATA_VAR.set({"beam_session": object()} if path == "beam" else None)
        try:
            return beam()
        finally:
            BEAM_DATA_VAR.reset(token)

    first_graph = forward(first_path)
    if first_path == "ordinary":
        assert not npu_patch._verify_and_record_captured_graph_npu(runner, 4)
    other_path = "beam" if first_path == "ordinary" else "ordinary"
    other_graph = forward(other_path)
    assert other_graph is not first_graph
    assert forward(first_path) is first_graph
    assert forward(other_path) is other_graph
    assert beam.concrete_aclgraph_entries is not ordinary.concrete_aclgraph_entries
    assert npu_patch._verify_and_record_captured_graph_npu(runner, 4)
    assert not beam.graph_ready(8)


def test_npu_suffix_bank_storage_count_deduplicates_packed_views() -> None:
    packed = torch.empty((2, 2, 1, 4, 3), dtype=torch.float32)
    pools = [
        SimpleNamespace(unshared_key=packed[0, layer], unshared_value=packed[1, layer])
        for layer in range(2)
    ]
    context = SimpleNamespace(pools=pools, _npu_beam_width_pools={})
    worker = SimpleNamespace(
        _gr_npu_graph_context=context, _gr_npu_graph_contexts={(1, 4): context}
    )
    assert npu_patch._retained_npu_suffix_kv_bytes(worker) == packed.numel() * packed.element_size()


def test_npu_capture_logs_requested_and_retained_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_graph_widths=(2,)),
    )
    worker = SimpleNamespace(_gr_npu_beam_graph_shapes={(1, 4)})
    capture = Mock(side_effect=[True, False])
    logger = Mock()
    monkeypatch.setattr(npu_patch, "_capture_beam_decode_graph_width", capture)
    monkeypatch.setattr(npu_patch, "logger", logger)
    assert npu_patch.capture_beam_decode_graph(worker)
    assert [call.args[1] for call in capture.call_args_list] == [4, 2]
    messages = [call.args[0] % call.args[1:] for call in logger.info.call_args_list]
    assert "widths=[4, 2] (count=2, B=1)" in messages[0]
    assert "captured (B, W)=[(1, 4)]" in messages[-1]
    assert "excludes graph and scratch memory" in messages[-1]


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("decode_step", "expected_positions"),
    [(0, [2, 2, 2, 3, 3, 3, -1, -1]), (1, [3, 3, 3, 4, 4, 4, -1, -1])],
)
def test_npu_eager_decode_binds_previous_winners(
    decode_step: int, expected_positions: list[int]
) -> None:
    plan = NPUEagerStagePlan.from_scheduler_output(
        _eager_stage_output(kind="decode", decode_step=decode_step)
    )
    input_ids = torch.full((8,), -1, dtype=torch.int32)
    positions = torch.full((8,), -1, dtype=torch.int64)
    winners = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)

    plan.prepare_decode_inputs(input_ids, positions, winners)

    assert input_ids.tolist() == [11, 12, 13, 21, 22, 23, -1, -1]
    assert positions.tolist() == expected_positions


class _NPUPositionTensor(torch.Tensor):  # type: ignore[misc]
    @property
    def device(self) -> torch.device:
        # Only the Python device guard is mocked; storage remains on CPU.
        return cast(torch.device, SimpleNamespace(type="npu"))


class _CUDAPositionTensor(torch.Tensor):  # type: ignore[misc]
    @property
    def device(self) -> torch.device:
        return cast(torch.device, SimpleNamespace(type="cuda"))


def _position_plan(
    prompt_lengths: tuple[int, ...],
    step: int,
    beam_width: int = 3,
    dynamic_beam_widths: tuple[int, ...] | None = None,
) -> NPUEagerStagePlan:
    output = _eager_stage_output(
        kind="decode", batch_size=len(prompt_lengths), beam_width=beam_width, decode_step=step
    )
    metadata = next(iter(output.gr_stage_metadata.values()))
    metadata.num_prompt_tokens_by_request = prompt_lengths
    metadata.num_prompt_tokens = sum(prompt_lengths)
    metadata.beam_params.dynamic_beam_widths = dynamic_beam_widths
    return NPUEagerStagePlan.from_scheduler_output(output)


class _CountingPositionTuple(tuple[int, ...]):
    reads: list[int]

    def __new__(cls, values: tuple[int, ...]) -> "_CountingPositionTuple":
        instance = super().__new__(cls, values)
        instance.reads = []
        return instance

    def __getitem__(self, index: Any) -> Any:
        assert isinstance(index, int)
        self.reads.append(index)
        return super().__getitem__(index)


@pytest.mark.parametrize("beam_width", [1, 3, 128])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "prompt_lengths",
    [
        (128,),
        (128, 128, 128, 128),
        (1, 127, 4096, 8192),
        (128, 128, 4096, 4096),
        (128, 4096, 128, 4096),
    ],
)
def test_npu_decode_positions_read_only_request_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    beam_width: int,
    prompt_lengths: tuple[int, ...],
) -> None:
    plan = _position_plan(prompt_lengths, 1, beam_width)
    assert plan.decode_positions is not None
    expected = torch.tensor(plan.decode_positions, dtype=torch.int64)
    counted = _CountingPositionTuple(plan.decode_positions)
    plan = replace(plan, decode_positions=counted)
    rows = expected.numel()
    input_ids = torch.full((rows + 2,), -7, dtype=torch.int32)
    positions = torch.full((rows + 2,), -9, dtype=torch.int64).as_subclass(_NPUPositionTensor)
    winners = torch.arange(rows, dtype=torch.int32).reshape(len(prompt_lengths), beam_width)
    calls: list[tuple[int, int, int]] = []

    def fill(tensor: torch.Tensor, value: int) -> torch.Tensor:
        calls.append((tensor.storage_offset(), tensor.numel(), value))
        return torch.Tensor.fill_(tensor, value)

    monkeypatch.setattr(_NPUPositionTensor, "fill_", fill)
    plan.prepare_decode_inputs(input_ids, positions, winners)

    # This is independent of width: the former per-row scan reads B * W
    # positions, including interior rows, even when every request is equal.
    assert counted.reads == [index * beam_width for index in range(len(prompt_lengths))]
    expected_runs = sum(
        index == 0 or length != prompt_lengths[index - 1]
        for index, length in enumerate(prompt_lengths)
    )
    assert len(calls) == expected_runs
    assert sum(count for _offset, count, _value in calls) == rows
    torch.testing.assert_close(positions[:rows].as_subclass(torch.Tensor), expected)
    torch.testing.assert_close(input_ids[:rows], winners.flatten())
    assert positions[rows:].as_subclass(torch.Tensor).tolist() == [-9, -9]
    assert input_ids[rows:].tolist() == [-7, -7]


@pytest.mark.parametrize(  # type: ignore[misc]
    "dynamic_beam_widths", [(128, 64, 32), (32, 128, 256)]
)
@pytest.mark.parametrize("step", [0, 1])  # type: ignore[misc]
@pytest.mark.parametrize("prompt_lengths", [(128,), (128, 128, 4096, 4096)])  # type: ignore[misc]
def test_npu_decode_positions_dynamic_boundaries_use_active_input_width(
    monkeypatch: pytest.MonkeyPatch,
    dynamic_beam_widths: tuple[int, ...],
    step: int,
    prompt_lengths: tuple[int, ...],
) -> None:
    capacity = 256
    plan = _position_plan(prompt_lengths, step, capacity, dynamic_beam_widths)
    width = dynamic_beam_widths[step]
    assert plan.execution.input_width == width < capacity
    assert plan.decode_positions is not None
    expected = torch.tensor(plan.decode_positions, dtype=torch.int64)
    counted = _CountingPositionTuple(plan.decode_positions)
    plan = replace(plan, decode_positions=counted)
    rows = len(prompt_lengths) * width
    # Capacity is larger than the active input: never fill inactive Beam rows.
    input_ids = torch.full((len(prompt_lengths) * capacity + 2,), -7, dtype=torch.int32)
    positions = torch.full_like(input_ids, -9, dtype=torch.int64).as_subclass(_NPUPositionTensor)
    winners = torch.arange(rows, dtype=torch.int32).reshape(len(prompt_lengths), width)
    calls: list[tuple[int, int, int]] = []

    def fill(tensor: torch.Tensor, value: int) -> torch.Tensor:
        calls.append((tensor.storage_offset(), tensor.numel(), value))
        return torch.Tensor.fill_(tensor, value)

    monkeypatch.setattr(_NPUPositionTensor, "fill_", fill)
    plan.prepare_decode_inputs(input_ids, positions, winners)

    assert counted.reads == [index * width for index in range(len(prompt_lengths))]
    expected_calls = (
        [(0, width, prompt_lengths[0] + step)]
        if len(prompt_lengths) == 1
        else [(0, 2 * width, 128 + step), (2 * width, 2 * width, 4096 + step)]
    )
    assert calls == expected_calls
    torch.testing.assert_close(positions[:rows].as_subclass(torch.Tensor), expected)
    torch.testing.assert_close(input_ids[:rows], winners.flatten())
    assert torch.all(positions[rows:].as_subclass(torch.Tensor) == -9)
    assert torch.all(input_ids[rows:] == -7)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("prompt_lengths", "runs"),
    [
        ((128,), 1),
        ((128, 128, 128, 128), 1),
        ((1, 127, 4096, 8192), 4),
        ((128, 128, 4096, 4096), 2),
        ((128, 4096, 128, 4096), 4),
    ],
)
def test_npu_decode_positions_fill_only_adjacent_equal_rows_without_host_transfer(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    prompt_lengths: tuple[int, ...],
    runs: int,
) -> None:
    rows = len(prompt_lengths) * 3
    input_ids = torch.full((rows + 2,), -7, dtype=torch.int32)
    positions = torch.full((rows + 2,), -9, dtype=dtype).as_subclass(_NPUPositionTensor)
    pointers = (input_ids.data_ptr(), positions.data_ptr())
    calls: list[tuple[int, int, int]] = []

    def fill(tensor: torch.Tensor, value: int) -> torch.Tensor:
        calls.append((tensor.storage_offset(), tensor.numel(), value))
        return torch.Tensor.fill_(tensor, value)

    def forbid_host_transfer(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("NPU position binding must not create or read a host tensor")

    monkeypatch.setattr(_NPUPositionTensor, "fill_", fill)
    for step in (0, 1, 7):
        plan = _position_plan(prompt_lengths, step)
        winners = torch.arange(rows, dtype=torch.int32) + step * 100
        expected_positions = torch.tensor(plan.decode_positions, dtype=dtype)
        calls.clear()
        with monkeypatch.context() as guard:
            guard.setattr(torch, "as_tensor", forbid_host_transfer)
            guard.setattr(torch.Tensor, "cpu", forbid_host_transfer)
            guard.setattr(torch.Tensor, "item", forbid_host_transfer)
            guard.setattr(torch.Tensor, "tolist", forbid_host_transfer)
            plan.prepare_decode_inputs(
                input_ids,
                positions,
                winners if step == 1 else winners.reshape(len(prompt_lengths), 3),
            )
        assert len(calls) == runs
        assert sum(count for _offset, count, _value in calls) == rows
        assert pointers == (input_ids.data_ptr(), positions.data_ptr())
        torch.testing.assert_close(positions[:rows].as_subclass(torch.Tensor), expected_positions)
        torch.testing.assert_close(input_ids[:rows], winners)
        assert positions[rows:].as_subclass(torch.Tensor).tolist() == [-9, -9]
        assert input_ids[rows:].tolist() == [-7, -7]


@pytest.mark.parametrize("mock_cuda", [False, True])  # type: ignore[misc]
def test_non_npu_decode_positions_keep_original_tensor_copy(
    monkeypatch: pytest.MonkeyPatch, mock_cuda: bool
) -> None:
    plan = _position_plan((128, 4096), 1)
    positions = torch.full((8,), -9, dtype=torch.int64)
    if mock_cuda:
        positions = positions.as_subclass(_CUDAPositionTensor)
    input_ids = torch.full((8,), -7, dtype=torch.int32)
    winners = torch.arange(6, dtype=torch.int32)
    original = torch.as_tensor
    calls: list[tuple[Any, torch.dtype, str]] = []

    def as_tensor(data: Any, *, dtype: torch.dtype, device: Any) -> torch.Tensor:
        calls.append((data, dtype, device.type))
        # Fake CUDA storage is still CPU storage, so keep the real operation on CPU.
        return original(data, dtype=dtype, device="cpu")

    monkeypatch.setattr(torch, "as_tensor", as_tensor)
    plan.prepare_decode_inputs(input_ids, positions, winners)
    assert calls == [(plan.decode_positions, torch.int64, "cuda" if mock_cuda else "cpu")]
    torch.testing.assert_close(
        positions.as_subclass(torch.Tensor),
        torch.tensor([129, 129, 129, 4097, 4097, 4097, -9, -9]),
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    ("invalid", "message"),
    [
        ("prefill", "requires a Decode plan"),
        ("input-rank", "input_ids has insufficient"),
        ("input-capacity", "input_ids has insufficient"),
        ("position-rank", "positions has insufficient"),
        ("position-capacity", "positions has insufficient"),
        ("winner-shape", "winner_token_ids must have shape"),
        ("winner-device", "must be on the same device"),
    ],
)
def test_npu_decode_positions_retain_input_validation(invalid: str, message: str) -> None:
    plan = _position_plan((128,), 1)
    input_ids = torch.full((5,), -7, dtype=torch.int32)
    positions = torch.full((5,), -9, dtype=torch.int64).as_subclass(_NPUPositionTensor)
    winners = torch.arange(3, dtype=torch.int32)
    if invalid == "prefill":
        plan = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="prefill"))
    elif invalid == "input-rank":
        input_ids = input_ids.unsqueeze(0)
    elif invalid == "input-capacity":
        input_ids = input_ids[:2]
    elif invalid == "position-rank":
        positions = positions.unsqueeze(0)
    elif invalid == "position-capacity":
        positions = positions[:2]
    elif invalid == "winner-shape":
        winners = winners.unsqueeze(1)
    elif invalid == "winner-device":
        winners = winners.as_subclass(_CUDAPositionTensor)
    with pytest.raises(BeamExecutionContractError, match=message):
        plan.prepare_decode_inputs(input_ids, positions, winners)
    assert torch.all(input_ids == -7)
    assert torch.all(positions.as_subclass(torch.Tensor) == -9)


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "prompt_lengths",
    [(128,), (128, 128, 128, 128), (1, 127, 4096, 8192), (128, 4096, 128, 4096)],
)
def test_npu_decode_positions_match_original_construction_across_async_steps(
    dtype: torch.dtype, prompt_lengths: tuple[int, ...]
) -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    plans = [_position_plan(prompt_lengths, step, beam_width=128) for step in (0, 1, 7)]
    rows = len(prompt_lengths) * 128
    input_ids = torch.full((rows + 2,), -7, dtype=torch.int32, device="npu")
    positions = torch.full((rows + 2,), -9, dtype=dtype, device="npu")
    winners = torch.empty((len(prompt_lengths), 128), dtype=torch.int32, device="npu")
    original_positions = [
        torch.as_tensor(plan.decode_positions, dtype=dtype, device=positions.device)
        for plan in plans
    ]
    pointers = (input_ids.data_ptr(), positions.data_ptr(), winners.data_ptr())
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    snapshots = []
    with torch.npu.stream(stream):
        for index, plan in enumerate(plans):
            winners.fill_(100 + index)
            plan.prepare_decode_inputs(input_ids, positions, winners)
            snapshots.append((input_ids.clone(), positions.clone()))
    # No host reads or synchronization between successive device updates.
    stream.synchronize()
    assert pointers == (input_ids.data_ptr(), positions.data_ptr(), winners.data_ptr())
    for index, ((ids, actual), expected) in enumerate(zip(snapshots, original_positions)):
        torch.testing.assert_close(actual[:rows], expected, rtol=0, atol=0)
        assert ids[:rows].cpu().tolist() == [100 + index] * rows
        assert ids[rows:].cpu().tolist() == [-7, -7]
        assert actual[rows:].cpu().tolist() == [-9, -9]


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "dynamic_beam_widths", [(128, 64, 32), (32, 128, 256)]
)
@pytest.mark.parametrize("prompt_lengths", [(128,), (128, 128, 4096, 4096)])  # type: ignore[misc]
def test_npu_decode_positions_dynamic_widths_match_original_across_async_steps(
    dtype: torch.dtype,
    dynamic_beam_widths: tuple[int, ...],
    prompt_lengths: tuple[int, ...],
) -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    capacity = 256
    batch = len(prompt_lengths)
    plans = [_position_plan(prompt_lengths, step, capacity, dynamic_beam_widths) for step in (0, 1)]
    input_ids = torch.full((batch * capacity + 2,), -7, dtype=torch.int32, device="npu")
    positions = torch.full((batch * capacity + 2,), -9, dtype=dtype, device="npu")
    winners = torch.empty((batch * capacity,), dtype=torch.int32, device="npu")
    expected_positions = [
        torch.as_tensor(plan.decode_positions, dtype=dtype, device="npu") for plan in plans
    ]
    pointers = (input_ids.data_ptr(), positions.data_ptr(), winners.data_ptr())
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    snapshots = []
    with torch.npu.stream(stream):
        for step, plan in enumerate(plans):
            width = dynamic_beam_widths[step]
            rows = batch * width
            input_ids.fill_(-7)
            positions.fill_(-9)
            winners.fill_(100 + step)
            plan.prepare_decode_inputs(input_ids, positions, winners[:rows].view(batch, width))
            snapshots.append((input_ids.clone(), positions.clone()))
    stream.synchronize()
    assert pointers == (input_ids.data_ptr(), positions.data_ptr(), winners.data_ptr())
    for step, ((ids, actual), expected) in enumerate(zip(snapshots, expected_positions)):
        rows = batch * dynamic_beam_widths[step]
        torch.testing.assert_close(actual[:rows], expected, rtol=0, atol=0)
        assert ids[:rows].cpu().tolist() == [100 + step] * rows
        assert torch.all(ids[rows:] == -7)
        assert torch.all(actual[rows:] == -9)


def test_npu_capability_gate_allows_graph_mode_and_remains_npu_only() -> None:
    worker = SimpleNamespace(
        device="npu",
        model_config=SimpleNamespace(enforce_eager=True, dtype=torch.bfloat16),
        use_async_scheduling=True,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        speculative_config=None,
    )
    validate_npu_eager_capabilities(worker)
    worker.model_config.enforce_eager = False
    validate_npu_eager_capabilities(worker)

    worker.device = "cuda"
    with pytest.raises(ValueError, match="NPU worker"):
        validate_npu_eager_capabilities(worker)


@pytest.mark.parametrize("decode_step", [0, 1])  # type: ignore[misc]
def test_npu_stage_runner_binds_physical_decode_without_mutating_scheduler(
    decode_step: int,
) -> None:
    scheduler_output = _eager_stage_output(kind="decode", decode_step=decode_step)
    plan = NPUEagerStagePlan.from_scheduler_output(scheduler_output)
    winners = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([0.5, 0.9]),
        top_p=torch.tensor([0.8, 0.7]),
        top_k=torch.tensor([5, 9], dtype=torch.int32),
    )
    input_batch = SimpleNamespace(
        req_ids=list(plan.execution.request_ids), sampling_metadata=sampling_metadata
    )
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        input_ids=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
        positions=torch.zeros(8, dtype=torch.int64),
        input_batch=input_batch,
        _current_scheduler_output=scheduler_output,
        _current_beam_info={"req_ids": []},
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker = worker
    stage.pending_plan = plan
    stage.latest_token_ids = winners
    runtime_output = stage._runtime_scheduler_output(scheduler_output, plan)
    worker._current_scheduler_output = runtime_output

    def original(self: Any, physical_output: Any, counts: Any) -> tuple[Any, ...]:
        assert self is worker
        assert physical_output is runtime_output
        assert physical_output is not scheduler_output
        assert physical_output.num_scheduled_tokens == {
            "request-0": 3,
            "request-1": 3,
        }
        assert counts.dtype == np.int32
        assert torch.from_numpy(counts).tolist() == [3, 3]
        return object(), object(), 6

    counts = np.asarray([1, 1], dtype=np.int32)
    logits, _, total = stage.prepare_inputs(original, runtime_output, counts)

    assert counts.tolist() == [1, 1]
    assert scheduler_output.num_scheduled_tokens == {
        "request-0": 1,
        "request-1": 1,
    }
    assert worker.input_ids.gpu[:6].tolist() == [11, 12, 13, 21, 22, 23]
    assert worker.positions[:6].tolist() == [value + decode_step for value in (2, 2, 2, 3, 3, 3)]
    assert logits.tolist() == list(range(6))
    assert total == 6
    assert isinstance(worker.input_batch, InputBatchProxy)
    # Bookkeeping retains B logical ids; indexed/copy access maps B*W rows.
    assert len(worker.input_batch.req_ids) == 2
    assert worker.input_batch.req_ids.copy() == ["request-0"] * 3 + ["request-1"] * 3
    assert worker.input_batch.req_ids[3] == "request-1"
    assert input_batch.req_ids == ["request-0", "request-1"]
    assert worker._current_beam_info["req_ids"] == input_batch.req_ids
    remapped = worker.input_batch.sampling_metadata
    assert remapped is sampling_metadata
    assert remapped.temperature.tolist() == pytest.approx([0.5] * 3 + [0.9] * 3)
    assert remapped.top_p.tolist() == pytest.approx([0.8] * 3 + [0.7] * 3)
    assert remapped.top_k.tolist() == [5] * 3 + [9] * 3
    assert remapped.top_k.dtype == torch.int32
    assert remapped.temperature.is_contiguous() and remapped.top_p.is_contiguous()


@pytest.mark.parametrize("device", ["cuda", "npu", "cpu"])  # type: ignore[misc]
def test_shared_stage_dispatch_routes_devices(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    from unittest.mock import Mock

    from vllm_gr.v1.worker.cuda import gpu_beam_stage_runner
    from vllm_gr.v1.worker.npu import npu_beam_stage_runner

    output = object()
    stage = SimpleNamespace(execute=Mock(return_value=output))
    cuda = Mock(return_value=stage)
    npu = Mock(return_value=stage)
    monkeypatch.setattr(gpu_beam_stage_runner, "get_stage_runner", cuda)
    monkeypatch.setattr(npu_beam_stage_runner, "get_npu_stage_runner", npu)
    hooks = object()
    worker = SimpleNamespace(device=device)
    scheduled = _eager_stage_output(kind="prefill")
    if device == "cpu":
        with pytest.raises(ValueError, match="no execution adapter"):
            _gr_dispatch._dispatch_stage_frame(worker, scheduled, object(), hooks_type=hooks)
        cuda.assert_not_called()
        npu.assert_not_called()
    else:
        assert (
            _gr_dispatch._dispatch_stage_frame(worker, scheduled, object(), hooks_type=hooks)
            is output
        )
        if device == "cuda":
            assert cuda.call_args.kwargs["hooks_type"] is hooks
            npu.assert_not_called()
        else:
            npu.assert_called_once()
            cuda.assert_not_called()


def test_npu_graph_context_survives_shape_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.beam.attention_context import BeamAttentionContext

    def context(batch_size: int) -> BeamAttentionContext:
        return BeamAttentionContext(batch_size, 4, 1, 3, 2, 1, torch.device("cpu"), torch.float16)

    graph_context = context(2)

    def pointers() -> tuple[int, ...]:
        return tuple(
            tensor.data_ptr()
            for tensor in (
                graph_context.pools[0].unshared_key,
                graph_context.pools[0].unshared_value,
                graph_context.decision.sequence,
                graph_context.decision.beam_scores,
                graph_context.decode_step_buf,
                graph_context.decode_step_1based_buf,
                graph_context.xllm_block_table_buf,
            )
        )

    captured_pointers = pointers()
    worker = SimpleNamespace(
        _gr_npu_graph_context=graph_context, _gr_npu_beam_graph_shapes={(2, 4)}
    )
    created: list[int] = []

    def create(worker: Any, config: Any, *, max_batch: int) -> BeamAttentionContext:
        created.append(max_batch)
        return context(max_batch)

    monkeypatch.setattr(_gr_context, "_create_beam_context", create)
    monkeypatch.setattr(_gr_context, "_get_gr_config", lambda worker: object())
    for batch_size in (2, 1, 2):
        scheduled = _eager_stage_output(kind="decode", batch_size=batch_size, beam_width=4)
        next(iter(scheduled.gr_stage_metadata.values())).beam_params.max_tokens = 3
        worker._current_scheduler_output = scheduled
        stage = object.__new__(NPUEagerBeamStageRunner)
        stage.worker, stage.attention_batch = worker, None
        stage.prefill_candidates = ConstraintOutput(
            token_ids=torch.arange(batch_size * 4, dtype=torch.int32).view(batch_size, 4),
            logprobs=torch.zeros(batch_size, 4),
            valid_mask=torch.ones(batch_size, 4, dtype=torch.bool),
        )
        stage._ensure_attention_batch(NPUEagerStagePlan.from_scheduler_output(scheduled))
        assert stage.attention_batch is not None
        expected = graph_context if batch_size == 2 else worker._gr_npu_eager_context
        assert stage.attention_batch.context is expected
        assert worker._gr_npu_graph_context is graph_context
        assert pointers() == captured_pointers
        assert expected.registry.num_active_sessions == batch_size
        stage._release()
        assert expected.registry.num_active_sessions == 0 and not worker._beam_sessions
    assert created == [1]


def test_npu_first_decode_native_contract_keeps_six_forward_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror the 0.30.0rc2 first-decode call order through the real GR hooks."""
    from vllm.v1.cudagraph_dispatcher import CUDAGraphMode

    from vllm_gr.v1.worker.npu import npu_beam_stage_runner, npu_model_runner_inputs

    logical = _eager_stage_output(kind="decode")
    metadata = next(iter(logical.gr_stage_metadata.values()))
    metadata.is_last_stage = False
    metadata.beam_params.max_tokens = 3
    prefill = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="prefill"))
    winners = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    with torch.inference_mode():
        context = SimpleNamespace(
            decode_step_buf=torch.zeros(1, dtype=torch.int32),
            decode_step_1based_buf=torch.zeros(1, dtype=torch.int32),
            xllm_block_table_buf=torch.zeros(2, dtype=torch.int32),
        )
    worker = SimpleNamespace(
        device="npu",
        input_ids=SimpleNamespace(gpu=torch.zeros(6, dtype=torch.int32)),
        positions=torch.zeros(6, dtype=torch.int64),
        input_batch=SimpleNamespace(req_ids=list(logical.num_scheduled_tokens)),
        execute_model_state=None,
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=False),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.previous = worker, prefill.execution
    stage.closed, stage.pending_plan, stage.execution_error = False, None, None
    stage.latest_token_ids = winners
    stage.session_request = None
    stage.attention_batch = SimpleNamespace(
        execution=NPUEagerStagePlan.from_scheduler_output(logical).execution,
        is_active=True,
        slot=0,
        sessions=tuple(
            SimpleNamespace(_state=SimpleNamespace(is_last_step=False)) for _ in winners
        ),
        context=context,
    )
    monkeypatch.setattr(npu_beam_stage_runner, "get_npu_stage_runner", lambda *args: stage)
    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *args, **kwargs: arange(*args, dtype=torch.int32))
    monkeypatch.setattr(
        npu_model_runner_inputs, "_materialize_logits_indices", lambda _, rows: torch.tensor(rows)
    )
    seen: list[str] = []

    def update(self: Any, scheduled: Any) -> None:
        assert scheduled.total_num_scheduled_tokens == 2
        assert tuple(scheduled.num_scheduled_tokens.values()) == (1, 1)
        seen.append("logical_state")

    worker._update_states = npu_patch.make_patched_update_states(update).__get__(worker)

    def prepare(self: Any, scheduled: Any, counts: np.ndarray) -> tuple[Any, ...]:
        assert counts.dtype == np.int32
        assert torch.from_numpy(counts).tolist() == [3, 3]
        assert scheduled.total_num_scheduled_tokens == 6
        seen.append("physical_inputs")
        return object(), None, int(counts.sum())

    worker._prepare_inputs = npu_patch.make_patched_prepare_inputs(prepare).__get__(worker)

    def determine(self: Any, **kwargs: Any) -> tuple[Any, ...]:
        assert kwargs["num_tokens"] == 6
        assert kwargs["max_num_scheduled_tokens"] == 3
        assert kwargs["num_scheduled_tokens_np"].tolist() == [3, 3]
        seen.append("physical_padding")
        return CUDAGraphMode.NONE, SimpleNamespace(num_tokens=6), False, None, None

    worker._determine = npu_patch.make_patched_determine_batch_execution_and_padding(
        determine
    ).__get__(worker)

    def native(self: Any, runtime: Any) -> Any:
        assert torch.is_inference_mode_enabled()
        # Native execute reads this scalar before updating request state.
        assert runtime.total_num_scheduled_tokens == 6
        self._update_states(runtime)
        counts = np.asarray(list(runtime.num_scheduled_tokens.values()), dtype=np.int32)
        maximum = int(counts.max())
        logits_indices, _, total = self._prepare_inputs(runtime, counts)
        assert total == 6 and logits_indices.tolist() == list(range(6))
        _, descriptor, *_ = self._determine(
            num_tokens=runtime.total_num_scheduled_tokens,
            num_scheduled_tokens_np=counts,
            max_num_scheduled_tokens=maximum,
        )
        # Native preprocess slices by the descriptor, and forward consumes that slice.
        input_ids = self.input_ids.gpu[: descriptor.num_tokens]
        assert input_ids.tolist() == winners.reshape(-1).tolist()
        assert self.positions[: descriptor.num_tokens].tolist() == [2, 2, 2, 3, 3, 3]
        seen.append("six_token_forward")
        return SimpleNamespace(logits=input_ids[:, None].float())

    result = _gr_dispatch._dispatch_stage_frame(worker, logical, native)
    assert result.logits.shape == (6, 1)
    assert logical.total_num_scheduled_tokens == 2
    assert tuple(logical.num_scheduled_tokens.values()) == (1, 1)
    assert seen == ["logical_state", "physical_inputs", "physical_padding", "six_token_forward"]
    # Ordinary requests bypass the logical GR view after sampling is complete.
    stage.pending_plan = None
    worker._update_states(logical)


@pytest.mark.parametrize("failure_point", ["plan", "forward"])  # type: ignore[misc]
def test_npu_failed_execute_reaches_sample_without_native_fallback(
    monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    from unittest.mock import Mock

    from vllm_gr.v1.worker.npu import npu_beam_stage_runner

    failure = ValueError("injected execute failure")
    worker = SimpleNamespace(device="npu", execute_model_state=None)
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.previous = worker, None
    stage.closed, stage.pending_plan, stage.execution_error = False, None, None
    worker._gr_npu_stage_runner = stage
    monkeypatch.setattr(npu_beam_stage_runner, "get_npu_stage_runner", lambda *args: stage)
    scheduled = _eager_stage_output(kind="prefill")
    next(iter(scheduled.gr_stage_metadata.values())).is_last_stage = False
    fail = Mock(side_effect=failure)
    if failure_point == "plan":
        monkeypatch.setattr(stage, "_validate_sequence", fail)
    with pytest.raises(ValueError) as execute_error:
        _gr_dispatch._dispatch_stage_frame(worker, scheduled, fail)
    assert execute_error.value is failure
    native_sample = Mock(side_effect=AssertionError("native sampler must not run"))

    def sample_original(*args: Any, **kwargs: Any) -> Any:
        return native_sample(*args, **kwargs)

    with pytest.raises(RuntimeError, match="execute_model failed") as sample_error:
        npu_patch.make_patched_sample_tokens(sample_original)(worker, None)
    assert sample_error.value.__cause__ is failure
    native_sample.assert_not_called()
    with pytest.raises(RuntimeError, match="owner failed"):
        stage.execute(scheduled, fail)


def test_npu_retirement_clears_failed_dispatch_state() -> None:
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.attention_batch = None
    stage.closed = False
    stage.logical_scheduler_output = object()
    stage.execution_error = ValueError("failed dispatch")
    stage._release()
    assert stage.closed
    assert stage.logical_scheduler_output is None
    assert stage.execution_error is None


def test_npu_stage_runner_receipt_preserves_owner_generation() -> None:
    scheduler_output = _eager_stage_output(
        kind="prefill",
        batch_size=1,
        beam_width=2,
    )
    metadata = next(iter(scheduler_output.gr_stage_metadata.values()))
    metadata.beam_params.temperature = 0.0
    metadata.is_last_stage = False
    metadata.output_device_state = object()
    plan = NPUEagerStagePlan.from_scheduler_output(scheduler_output)

    def select(logits: torch.Tensor, constraint: ConstraintMetadata) -> ConstraintOutput:
        assert torch.is_inference_mode_enabled()
        assert logits.shape[0] == 1
        assert constraint.top_k == 2
        return ConstraintOutput(
            token_ids=torch.tensor([[11, 12]], dtype=torch.int32),
            logprobs=torch.tensor([[-0.1, -0.2]], dtype=torch.float32),
            valid_mask=torch.ones((1, 2), dtype=torch.bool),
        )

    worker = SimpleNamespace(
        _current_scheduler_output=scheduler_output,
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(max_rows=1),
            select=select,
        ),
        execute_model_state=object(),
        kv_connector_output=object(),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker = worker
    stage.execution_error = None
    stage.pending_plan = plan
    stage.logits = torch.zeros((1, 16), dtype=torch.float32)
    stage.prefill_candidates = None
    stage.latest_token_ids = None
    stage.attention_batch = None
    stage.closed = False

    output = stage.sample()

    result = output.gr_worker_results[plan.execution.session_id]
    assert result.owner_generation == plan.execution.owner_generation == 1


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("candidate_top_k", "candidate_shape", "error_type", "message"),
    [
        (3, (2, 3), BeamExecutionContractError, "Prefill winner_token_ids must have shape (2, 2)"),
        (1, (2, 1), BeamExecutionContractError, "Prefill winner_token_ids must have shape (2, 2)"),
        (3, (2, 2), RuntimeError, "NPU candidate selection returned an invalid candidate shape"),
        (2, (1, 2), RuntimeError, "NPU candidate selection returned an invalid candidate shape"),
    ],
)
def test_npu_prefill_sample_preserves_candidate_and_winner_shape_errors(
    candidate_top_k: int,
    candidate_shape: tuple[int, int],
    error_type: type[Exception],
    message: str,
) -> None:
    scheduled = _eager_stage_output(kind="prefill", beam_width=2 if candidate_top_k == 1 else 4)
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    metadata.beam_params.temperature = 0.0
    # A dynamic schedule rejects K < W before sampling; exercise that case
    # through fixed-width execution so it reaches the winner-shape check.
    if candidate_top_k != 1:
        metadata.beam_params.dynamic_beam_widths = (2, 2, 2)
    metadata.beam_params.candidate_width = candidate_top_k
    metadata.is_last_stage = False
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    candidates = ConstraintOutput(
        token_ids=torch.ones(candidate_shape, dtype=torch.int32),
        logprobs=torch.zeros(candidate_shape),
        valid_mask=torch.ones(candidate_shape, dtype=torch.bool),
    )

    def select(logits: torch.Tensor, constraint: ConstraintMetadata) -> ConstraintOutput:
        assert constraint.current_step == 0 and constraint.top_k == candidate_top_k
        return candidates

    worker = SimpleNamespace(
        _current_scheduler_output=scheduled,
        constraint_backend=SimpleNamespace(info=SimpleNamespace(max_rows=2), select=select),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.pending_plan, stage.execution_error = worker, plan, None
    previous_candidates = object()
    previous_token_ids = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    logits = torch.zeros((2, 16))
    stage.prefill_candidates = previous_candidates
    stage.latest_token_ids = previous_token_ids
    stage.logits = logits

    with pytest.raises(error_type) as error:
        stage.sample()

    assert error.type is error_type
    assert str(error.value) == message

    if error_type is BeamExecutionContractError:
        assert stage.prefill_candidates is candidates
    else:
        assert stage.prefill_candidates is previous_candidates
    assert stage.latest_token_ids is previous_token_ids
    assert stage.pending_plan is plan
    assert stage.logits is logits


@pytest.mark.l1  # type: ignore[misc]
def test_npu_decode_sample_accepts_candidate_width_distinct_from_winner_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker.npu import npu_beam_stage_runner

    scheduled = _eager_stage_output(kind="decode", batch_size=1, beam_width=2)
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    metadata.beam_params.temperature = 0.0
    metadata.beam_params.candidate_width = 3
    metadata.is_last_stage = False
    metadata.output_device_state = None
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    candidates = ConstraintOutput(
        token_ids=torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32),
        logprobs=torch.zeros((2, 3)),
        valid_mask=torch.ones((2, 3), dtype=torch.bool),
    )
    winners = torch.tensor([[11, 21]], dtype=torch.int32)

    def select(logits: torch.Tensor, constraint: ConstraintMetadata) -> ConstraintOutput:
        assert constraint.current_step == 1 and constraint.top_k == 3
        return candidates

    def decide(
        worker: Any,
        attention_batch: Any,
        token_ids: torch.Tensor,
        logprobs: torch.Tensor,
        *,
        is_last_step: bool,
        result_width: int,
    ) -> dict[str, torch.Tensor]:
        assert token_ids is candidates.token_ids
        assert logprobs.shape == (2, 3)
        assert not is_last_step and result_width == 2
        return {"out_token_ids": winners}

    monkeypatch.setattr(npu_beam_stage_runner, "run_beam_decision_candidates", decide)
    worker = SimpleNamespace(
        _current_scheduler_output=scheduled,
        constraint_backend=SimpleNamespace(info=SimpleNamespace(max_rows=2), select=select),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.pending_plan, stage.execution_error = worker, plan, None
    stage.logits = torch.zeros((2, 16))
    stage.session_request = SimpleNamespace(
        output_options=SimpleNamespace(include_parent_beam_indices=False)
    )
    stage.attention_batch = SimpleNamespace(
        context=SimpleNamespace(decision=SimpleNamespace(sequence=torch.zeros((1, 2, 1))))
    )

    output = stage.sample()

    assert stage.latest_token_ids is winners
    assert output.gr_worker_results[plan.execution.session_id].stage_index == 1


def test_npu_execution_batch_flattens_only_the_xllm_attention_view() -> None:
    execution = NPUEagerStagePlan.from_scheduler_output(
        _eager_stage_output(kind="decode")
    ).execution
    source = SimpleNamespace(
        unshared_key=torch.arange(6).view(2, 3, 1),
        unshared_value=torch.arange(6, 12).view(2, 3, 1),
        max_batch=2,
        max_beams=3,
    )
    context = SimpleNamespace(
        pools=[source],
        max_beams=3,
        max_decode_steps=2,
        suffix_k_buf=torch.zeros(12, 1),
        suffix_v_buf=torch.zeros(12, 1),
    )
    sessions = [
        SimpleNamespace(
            session_id=request_id,
            slot=index,
            context=context,
            beam_width=3,
            decode_step=0,
            is_active=True,
            _state=object(),
        )
        for index, request_id in enumerate(execution.request_ids)
    ]

    batch = NPUBeamExecutionBatch(execution, sessions)
    flat = batch.pools[0].unshared_key

    assert flat.shape == (1, 6, 1)
    assert flat.data_ptr() == source.unshared_key.data_ptr()
    restored = restore_npu_cache_pool_layout(flat, request_width=3)
    assert restored.shape == (2, 3, 1)
    assert restored.data_ptr() == source.unshared_key.data_ptr()


def test_npu_terminal_result_keeps_batch_identity_and_output_options() -> None:
    request_ids = ("request-0", "request-1")
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.pending_plan = SimpleNamespace(execution=SimpleNamespace(request_ids=request_ids))
    stage.session_request = SimpleNamespace(
        batch_id="batch-0",
        items=tuple(
            SimpleNamespace(
                item_index=index,
                request=SimpleNamespace(request_id=request_id, prompt_token_ids=[1, 2]),
            )
            for index, request_id in enumerate(request_ids)
        ),
        beam_params=SimpleNamespace(
            stop_token_ids=(99,), ignore_stop_tokens=False, length_penalty=1.0
        ),
        output_options=SimpleNamespace(
            num_return_sequences=1,
            include_stop_token=False,
            include_cumulative_logprob=True,
            include_parent_beam_indices=True,
        ),
    )
    result = stage._build_terminal_result(
        torch.tensor([[[10, 99], [20, 21]], [[30, 31], [40, 41]]], dtype=torch.int32),
        torch.tensor([[-0.1, -1.0], [-0.2, -2.0]], dtype=torch.float32),
        torch.tensor([[[0, 0], [0, 1]], [[0, 0], [0, 1]]], dtype=torch.int32),
    )

    assert result.failure is None and result.output is not None
    assert tuple(item.request_id for item in result.output.items) == request_ids
    first = result.output.items[0].sequences[0]
    assert (first.token_ids, first.parent_beam_indices) == ((10,), (0,))
    assert (first.finish_reason, first.stop_token_id) == ("stop", 99)


@pytest.mark.parametrize(  # type: ignore[misc]
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.npu_test)]
)
@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("result_width", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "case", ["nan", "posinf", "neginf", "masked", "finite", "unsorted", "all_invalid"]
)
def test_npu_single_step_filters_candidates_before_return_limit(
    device: str, batch_size: int, result_width: int, case: str
) -> None:
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("Ascend NPU is unavailable")
    scheduled = _eager_stage_output(kind="prefill", batch_size=batch_size, beam_width=2)
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    metadata.beam_params.temperature = 1.0
    metadata.beam_params.dynamic_beam_widths = (2,)
    metadata.beam_params.max_tokens = 1
    metadata.is_last_stage = True
    metadata.output_device_state = None
    metadata.output_options = SimpleNamespace(
        num_return_sequences=result_width,
        include_stop_token=False,
        include_cumulative_logprob=True,
        include_parent_beam_indices=True,
    )
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    first_score = {
        "nan": float("nan"),
        "posinf": float("inf"),
        "neginf": -float("inf"),
        "masked": 0.0,
        "finite": -0.1,
        "unsorted": -0.9,
        "all_invalid": -0.1,
    }[case]
    token_ids = torch.tensor(
        [[11 + row * 10, 12 + row * 10] for row in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )
    scores = torch.tensor([[first_score, -0.2]] * batch_size, device=device)
    valid_mask = torch.ones_like(scores, dtype=torch.bool)
    if case == "masked":
        valid_mask[:, 0] = False
    elif case == "all_invalid":
        scores[-1] = torch.tensor([float("nan"), float("inf")], device=device)
    candidates = ConstraintOutput(token_ids=token_ids, logprobs=scores, valid_mask=valid_mask)

    def select(logits: torch.Tensor, constraint: ConstraintMetadata) -> ConstraintOutput:
        assert logits.device.type == device
        assert constraint.current_step == 0 and constraint.top_k == 2
        return candidates

    worker = SimpleNamespace(
        _current_scheduler_output=scheduled,
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(max_rows=batch_size), select=select
        ),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.pending_plan, stage.execution_error = worker, plan, None
    stage.logits = torch.zeros((batch_size, 16), device=device)
    stage.attention_batch = None
    stage.session_request = SimpleNamespace(
        batch_id="batch-0",
        items=tuple(
            SimpleNamespace(
                item_index=row,
                request=SimpleNamespace(request_id=request_id, prompt_token_ids=[1, 2]),
            )
            for row, request_id in enumerate(plan.execution.request_ids)
        ),
        beam_params=SimpleNamespace(stop_token_ids=(), ignore_stop_tokens=True, length_penalty=0.0),
        output_options=metadata.output_options,
    )
    output = stage.sample()
    receipt = output.gr_worker_results[plan.execution.session_id]
    result = receipt.terminal_result
    assert receipt.finished and receipt.resources_released
    assert stage.closed and stage.pending_plan is None
    if case == "all_invalid":
        assert result.output is None and result.failure.code == "no_valid_beams"
        assert result.failure.item_index == batch_size - 1
        return
    assert result.failure is None
    assert tuple(item.request_id for item in result.output.items) == plan.execution.request_ids
    expected = [0, 1] if case == "finite" else [1, 0] if case == "unsorted" else [1]
    for row, item in enumerate(result.output.items):
        selected = expected[:result_width]
        assert [seq.token_ids for seq in item.sequences] == [
            (11 + row * 10 + index,) for index in selected
        ]
        for sequence, index in zip(item.sequences, selected):
            assert sequence.cumulative_logprob == pytest.approx(first_score if index == 0 else -0.2)
            assert sequence.parent_beam_indices == (0,)
            assert sequence.finish_reason == "length" and sequence.stop_token_id is None


def _terminal_result_runner(
    batch: int,
    *,
    returned: int = 2,
    stop_ids: tuple[int, ...] = (),
    ignore_stops: bool = False,
    penalty: float = 1.0,
    include_stop: bool = True,
    include_score: bool = True,
    include_parents: bool = True,
) -> NPUEagerBeamStageRunner:
    request_ids = tuple(f"request-{row}" for row in range(batch))
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.pending_plan = SimpleNamespace(execution=SimpleNamespace(request_ids=request_ids))
    stage.session_request = SimpleNamespace(
        batch_id="terminal-batch",
        items=tuple(
            SimpleNamespace(
                item_index=row,
                request=SimpleNamespace(request_id=request_id, prompt_token_ids=[1] * (row + 1)),
            )
            for row, request_id in enumerate(request_ids)
        ),
        beam_params=SimpleNamespace(
            stop_token_ids=stop_ids, ignore_stop_tokens=ignore_stops, length_penalty=penalty
        ),
        output_options=SimpleNamespace(
            num_return_sequences=returned,
            include_stop_token=include_stop,
            include_cumulative_logprob=include_score,
            include_parent_beam_indices=include_parents,
        ),
    )
    return stage


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "batch,width,steps", [(1, 64, 1), (1, 128, 3), (4, 64, 2), (4, 128, 3)]
)
@pytest.mark.parametrize(  # type: ignore[misc]
    "include_score,include_parents", [(True, True), (False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("penalty", [0.0, 1.0, -1.0])  # type: ignore[misc]
def test_npu_cpu_columns_materialize_identically_to_stop_aware_rows(
    batch: int, width: int, steps: int, include_score: bool, include_parents: bool, penalty: float
) -> None:
    from vllm_gr.v1.beam.batch_contracts import (
        BeamSequenceColumns,
        materialize_beam_result,
        validate_complete_sequences,
    )

    stage = _terminal_result_runner(
        batch,
        returned=width,
        penalty=penalty,
        include_score=include_score,
        include_parents=include_parents,
    )
    stage.session_request.beam_params.max_tokens = steps
    tokens = torch.arange(batch * width * steps, dtype=torch.int32).view(batch, width, steps)
    parents = torch.zeros_like(tokens)
    # Ties, both signed zeros, and nonfinite candidates exercise the original
    # Python ranking. Scores are transported as their original float32 bits.
    scores = (torch.arange(batch * width, dtype=torch.float32).view(batch, width) % 7) * -0.125
    scores[:, :5] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0])
    columns = stage._build_terminal_result(tokens, scores, parents)
    assert columns.output is not None
    for item in columns.output.items:
        assert isinstance(item.sequences, BeamSequenceColumns)
        validate_complete_sequences(item.sequences)
    # A stop ID absent from the input selects the unchanged row path without
    # changing any sequence's length, score normalization, or return limit.
    stage.session_request.beam_params.stop_token_ids = (2**31 - 1,)
    rows = stage._build_terminal_result(tokens, scores, parents)
    materialized = materialize_beam_result(columns)
    assert materialized == rows
    if include_score:
        for left, right in zip(materialized.output.items, rows.output.items, strict=True):
            assert [struct.pack("<f", x.cumulative_logprob) for x in left.sequences] == [
                struct.pack("<f", x.cumulative_logprob) for x in right.sequences
            ]
    # Serialized columns own their bytes even after worker tensors are reused.
    tokens.fill_(99)
    parents.fill_(99)
    scores.fill_(99)
    assert materialize_beam_result(columns) == rows


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "width,returned,steps,budget,stop_ids",
    [(32, 32, 3, 3, ()), (128, 2, 3, 3, ()), (64, 64, 2, 3, ()), (64, 64, 3, 3, (99,))],
)
def test_npu_cpu_columns_keep_unsupported_results_on_rows(
    width: int, returned: int, steps: int, budget: int, stop_ids: tuple[int, ...]
) -> None:
    stage = _terminal_result_runner(1, returned=returned, stop_ids=stop_ids)
    stage.session_request.beam_params.max_tokens = budget
    tokens = torch.full((1, width, steps), 99, dtype=torch.int32)
    result = stage._build_terminal_result(tokens, torch.zeros(1, width), torch.zeros_like(tokens))
    assert result.output is not None
    assert isinstance(result.output.items[0].sequences, tuple)


@pytest.mark.l1  # type: ignore[misc]
def test_npu_cpu_columns_all_invalid_preserves_lazy_normalization_failure() -> None:
    stage = _terminal_result_runner(4, returned=64, penalty=1e6)
    stage.session_request.beam_params.max_tokens = 3
    tokens = torch.zeros(4, 64, 3, dtype=torch.int32)
    result = stage._build_terminal_result(tokens, torch.full((4, 64), -float("inf")), tokens)
    assert result.output is None
    assert (result.failure.code, result.failure.item_index) == ("no_valid_beams", 0)


@pytest.mark.parametrize(  # type: ignore[misc]
    "batch,steps,include_score,include_parents",
    [
        (1, 1, True, False),
        (1, 2, False, True),
        (1, 3, True, True),
        (4, 1, False, False),
        (4, 2, True, False),
        (4, 3, True, True),
    ],
)
def test_npu_terminal_fixed_lengths_preserve_rows_scores_and_output_flags(
    batch: int, steps: int, include_score: bool, include_parents: bool
) -> None:
    stage = _terminal_result_runner(
        batch, include_score=include_score, include_parents=include_parents
    )
    tokens = torch.tensor(
        [
            [[100 * row + 10 * beam + step for step in range(steps)] for beam in range(3)]
            for row in range(batch)
        ],
        dtype=torch.int32,
    )
    parents = torch.tensor([[[0, 1, 2][:steps]] * 3] * batch, dtype=torch.int32)
    scores = torch.tensor([[-0.25, -0.1, -0.2]] * batch, dtype=torch.float32)
    original_tokens, original_parents = tokens.clone(), parents.clone()

    result = stage._build_terminal_result(tokens, scores, parents)

    assert result.batch_id == "terminal-batch" and result.failure is None
    assert result.output is not None and len(result.output.items) == batch
    for row, output in enumerate(result.output.items):
        assert (output.item_index, output.request_id) == (row, f"request-{row}")
        assert [sequence.token_ids for sequence in output.sequences] == [
            tuple(100 * row + 10 * beam + step for step in range(steps)) for beam in (1, 2)
        ]
        for sequence, score_bits in zip(output.sequences, ("bdcccccd", "be4ccccd"), strict=True):
            if include_score:
                assert struct.pack(">f", sequence.cumulative_logprob).hex() == score_bits
            else:
                assert sequence.cumulative_logprob is None
            assert sequence.parent_beam_indices == ((0, 1, 2)[:steps] if include_parents else ())
            assert sequence.finish_reason == "length" and sequence.stop_token_id is None
    assert torch.equal(tokens, original_tokens) and torch.equal(parents, original_parents)


def test_npu_terminal_filters_nonfinite_before_limits_independently_per_request() -> None:
    stage = _terminal_result_runner(4, include_score=False, include_parents=False)
    tokens = torch.arange(20, dtype=torch.int32).view(4, 5, 1)
    scores = torch.tensor(
        [
            [float("nan"), float("inf"), -float("inf"), -0.25, -0.125],
            [-0.1, float("nan"), float("inf"), -float("inf"), float("nan")],
            [-float("inf"), -float("inf"), -float("inf"), -0.125, -0.125],
            [-0.3, float("nan"), float("inf"), -float("inf"), -0.2],
        ],
        dtype=torch.float32,
    )

    result = stage._build_terminal_result(tokens, scores, torch.zeros_like(tokens))

    assert result.failure is None and result.output is not None
    assert [[seq.token_ids for seq in item.sequences] for item in result.output.items] == [
        [(4,), (3,)],
        [(5,)],
        [(13,), (14,)],
        [(19,), (15,)],
    ]
    assert all(
        seq.cumulative_logprob is None for item in result.output.items for seq in item.sequences
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "penalty,include_stop,order",
    [
        (1.0, True, [2, 0, 1]),
        (1.0, False, [2, 0, 1]),
        (0.0, False, [2, 0, 1]),
        (-1.0, True, [0, 2, 1]),
    ],
)
def test_npu_terminal_stop_lengths_rank_with_prompt_and_keep_parent_alignment(
    penalty: float, include_stop: bool, order: list[int]
) -> None:
    stage = _terminal_result_runner(
        1, returned=3, stop_ids=(99, 98), penalty=penalty, include_stop=include_stop
    )
    tokens = torch.tensor([[[99, 97, 99], [20, 21, 22], [30, 99, 98]]], dtype=torch.int32)
    parents = torch.tensor([[[0, 1, 2], [0, 2, 1], [0, 1, 0]]], dtype=torch.int32)
    result = stage._build_terminal_result(
        tokens, torch.tensor([[-0.5, -1.0, -0.375]], dtype=torch.float32), parents
    )

    assert result.failure is None and result.output is not None
    expected_tokens = [
        (99,) if include_stop else (),
        (20, 21, 22),
        (30, 99) if include_stop else (30,),
    ]
    expected_parents = [(0,) if include_stop else (), (0, 2, 1), (0, 1) if include_stop else (0,)]
    expected_scores = [-0.5, -1.0, -0.375]
    for sequence, beam in zip(result.output.items[0].sequences, order, strict=True):
        assert sequence.token_ids == expected_tokens[beam]
        assert sequence.parent_beam_indices == expected_parents[beam]
        assert sequence.cumulative_logprob == expected_scores[beam]
        assert sequence.finish_reason == ("length" if beam == 1 else "stop")
        assert sequence.stop_token_id == (None if beam == 1 else 99)


def test_npu_terminal_ignored_stops_keep_full_history_and_same_parent_payload() -> None:
    stage = _terminal_result_runner(1, stop_ids=(99, 98), ignore_stops=True, include_stop=False)
    tokens = torch.tensor([[[99, 97, 99], [20, 21, 22], [30, 99, 98]]], dtype=torch.int32)
    parents = torch.tensor([[[0, 1, 2], [0, 2, 1], [0, 1, 0]]], dtype=torch.int32)
    scores = torch.tensor([[-0.5, -1.0, -0.375]], dtype=torch.float32)
    included = stage._build_terminal_result(tokens, scores, parents)
    stage.session_request.output_options.include_parent_beam_indices = False
    omitted = stage._build_terminal_result(tokens, scores, parents)

    assert included.output is not None and omitted.output is not None
    for left, right, expected_tokens, expected_parents in zip(
        included.output.items[0].sequences,
        omitted.output.items[0].sequences,
        [(30, 99, 98), (99, 97, 99)],
        [(0, 1, 0), (0, 1, 2)],
        strict=True,
    ):
        assert left.token_ids == right.token_ids == expected_tokens
        assert left.parent_beam_indices == expected_parents and right.parent_beam_indices == ()
        assert left.cumulative_logprob == right.cumulative_logprob
        assert left.finish_reason == right.finish_reason == "length"
        assert left.stop_token_id is right.stop_token_id is None


@pytest.mark.parametrize(  # type: ignore[misc]
    "raw_scores,penalty,expected_tokens,expected_bits",
    [
        ([-0.0, 0.0, -0.5], 1.0, [10, 20, 30], ["80000000", "00000000", "bf000000"]),
        (
            [-8 * 2.0**-149, -(2.0**-149), -(2.0**-149)],
            1023.0,
            [20, 30, 10],
            ["80000001", "80000001", "80000008"],
        ),
    ],
)
def test_npu_terminal_ties_keep_raw_score_then_original_beam_order(
    raw_scores: list[float], penalty: float, expected_tokens: list[int], expected_bits: list[str]
) -> None:
    stage = _terminal_result_runner(1, returned=3, penalty=penalty)
    tokens = torch.tensor([[[10], [20], [30]]], dtype=torch.int32)
    result = stage._build_terminal_result(
        tokens, torch.tensor([raw_scores], dtype=torch.float32), torch.zeros_like(tokens)
    )

    assert result.output is not None and result.failure is None
    sequences = result.output.items[0].sequences
    assert [sequence.token_ids for sequence in sequences] == [(token,) for token in expected_tokens]
    assert [
        struct.pack(">f", sequence.cumulative_logprob).hex() for sequence in sequences
    ] == expected_bits


@pytest.mark.parametrize("width", [0, 3])  # type: ignore[misc]
def test_npu_terminal_empty_or_nonfinite_rows_do_not_evaluate_overflowing_penalty(
    width: int,
) -> None:
    stage = _terminal_result_runner(1, penalty=1e6)
    tokens = torch.zeros((1, width, 3), dtype=torch.int32)
    scores = torch.tensor([[float("nan"), float("inf"), -float("inf")]], dtype=torch.float32)[
        :, :width
    ]

    result = stage._build_terminal_result(tokens, scores, torch.zeros_like(tokens))

    assert result.output is None and result.failure is not None
    assert (result.failure.code, result.failure.item_index) == ("no_valid_beams", 0)


def test_npu_terminal_invalid_second_request_returns_batch_failure_without_partial_output() -> None:
    stage = _terminal_result_runner(4)
    tokens = torch.arange(8, dtype=torch.int32).view(4, 2, 1)
    scores = torch.tensor(
        [[-0.1, -0.2], [float("nan"), float("inf")], [-0.3, -0.4], [-0.5, -0.6]],
        dtype=torch.float32,
    )

    result = stage._build_terminal_result(tokens, scores, torch.zeros_like(tokens))

    assert result.batch_id == "terminal-batch" and result.output is None
    assert result.failure is not None
    assert (result.failure.code, result.failure.message, result.failure.item_index) == (
        "no_valid_beams",
        "No valid final Beam candidates",
        1,
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "invalid,message",
    [
        ("session", "missing session_init"),
        ("order", "request order changed"),
        ("scores", "inconsistent shapes"),
        ("parents", "inconsistent shapes"),
        ("parent_depth", "shape"),
    ],
)
def test_npu_terminal_rejects_existing_session_order_and_tensor_errors(
    invalid: str, message: str
) -> None:
    stage = _terminal_result_runner(1, include_parents=True)
    tokens = torch.zeros((1, 2, 3), dtype=torch.int32)
    scores = torch.zeros((1, 2), dtype=torch.float32)
    parents = torch.zeros_like(tokens)
    if invalid == "session":
        stage.session_request = None
    elif invalid == "order":
        stage.pending_plan.execution.request_ids = ("different-request",)
    elif invalid == "scores":
        scores = scores[:, :1]
    elif invalid == "parents":
        parents = parents[:, :1]
    else:
        parents = parents[:, :, :2]

    with pytest.raises(RuntimeError, match=message):
        stage._build_terminal_result(tokens, scores, parents)


def _parent_history_request(batch_size: int, include_parents: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch_id="batch-0",
        items=tuple(
            SimpleNamespace(
                item_index=row,
                request=SimpleNamespace(request_id=f"request-{row}", prompt_token_ids=[1, 2]),
            )
            for row in range(batch_size)
        ),
        beam_params=SimpleNamespace(
            beam_width=3,
            max_tokens=3,
            temperature=1.0,
            stop_token_ids=(),
            ignore_stop_tokens=False,
            length_penalty=0.0,
        ),
        output_options=SimpleNamespace(
            num_return_sequences=3,
            include_stop_token=False,
            include_cumulative_logprob=True,
            include_parent_beam_indices=include_parents,
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("steps", [1, 2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("include_parents", [False, True])  # type: ignore[misc]
def test_npu_parent_history_follows_output_options_through_session(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, steps: int, include_parents: bool
) -> None:
    """Exercise the real decision, including duplicate parents and request offsets."""
    from unittest.mock import Mock

    from vllm_gr.v1.beam.attention_context import BeamAttentionContext

    context = BeamAttentionContext(batch_size, 3, 1, 3, 2, 1, torch.device("cpu"), torch.float16)
    sequence_pointer = context.decision.sequence.data_ptr()
    scores_pointer = context.decision.beam_scores.data_ptr()
    request = _parent_history_request(batch_size, include_parents)
    request.beam_params.max_tokens = steps
    token_tables = (
        torch.tensor([[10, 20, 30]], dtype=torch.int32),
        torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=torch.int32),
        torch.tensor([[41, 42, 43], [51, 52, 53], [61, 62, 63]], dtype=torch.int32),
    )
    score_tables = (
        torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32),
        torch.tensor([[-0.8, -1.0, -1.2], [-0.01, -0.02, -2.0], [-0.03, -2.0, -3.0]]),
        torch.tensor([[-0.5, -2.0, -3.0], [-0.01, -0.02, -3.0], [-0.01, -2.0, -3.0]]),
    )

    def select(logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        index = metadata.current_step
        tokens = torch.cat([token_tables[index] + row * 100 for row in range(batch_size)])
        return ConstraintOutput(
            token_ids=tokens,
            logprobs=score_tables[index].repeat(batch_size, 1),
            valid_mask=torch.ones_like(tokens, dtype=torch.bool),
        )

    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _vllm_gr_config=SimpleNamespace(constraint=table_constraint()),
        _gr_npu_graph_context=context,
        _gr_npu_beam_graph_shapes={(batch_size, 3)},
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(max_rows=batch_size * 3), select=select
        ),
        execute_model_state=None,
        kv_connector_output=None,
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.session_request = worker, request
    stage.execution_error, stage.attention_batch, stage.parent_history = None, None, None
    stage.closed = False
    advance = Mock(wraps=stage._advance_parent_history)
    if not include_parents:
        advance.side_effect = AssertionError("unrequested parent history must not be advanced")
    monkeypatch.setattr(stage, "_advance_parent_history", advance)
    zeros_like = Mock(wraps=torch.zeros_like)
    monkeypatch.setattr(torch, "zeros_like", zeros_like)
    zeros = Mock(wraps=torch.zeros)
    monkeypatch.setattr(torch, "zeros", zeros)
    terminal = None
    for index in range(steps):
        scheduled = _eager_stage_output(
            kind="prefill" if index == 0 else "decode",
            batch_size=batch_size,
            beam_width=3,
            decode_step=max(index - 1, 0),
        )
        metadata = next(iter(scheduled.gr_stage_metadata.values()))
        metadata.beam_params = request.beam_params
        metadata.is_last_stage = index == steps - 1
        metadata.output_device_state = None
        worker._current_scheduler_output = scheduled
        stage.pending_plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
        if index:
            stage._ensure_attention_batch(stage.pending_plan)
            assert (stage.parent_history is not None) is include_parents
            assert stage.attention_batch is not None
            for session in stage.attention_batch.sessions:
                session._state.is_last_step = metadata.is_last_stage
        rows = batch_size if index == 0 else batch_size * 3
        stage.logits = torch.zeros(rows, 128)
        output = stage.sample()
        terminal = output.gr_worker_results["batch-0"].terminal_result
        assert stage.closed is (index == steps - 1)

    assert terminal is not None and terminal.output is not None and terminal.failure is None
    expected_tokens = (
        ((10,), (20,), (30,)),
        ((20, 21), (20, 22), (30, 31)),
        ((20, 22, 51), (20, 22, 52), (30, 31, 61)),
    )[steps - 1]
    expected_parents = (
        ((0,), (0,), (0,)),
        ((0, 1), (0, 1), (0, 2)),
        ((0, 1, 1), (0, 1, 1), (0, 2, 2)),
    )[steps - 1]
    expected_scores = (
        score_tables[0][0],
        torch.stack(
            (
                score_tables[0][0, 1] + score_tables[1][1, 0],
                score_tables[0][0, 1] + score_tables[1][1, 1],
                score_tables[0][0, 2] + score_tables[1][2, 0],
            )
        ),
        torch.tensor([-0.2, -0.2, -0.3])
        + torch.tensor([-0.02, -0.02, -0.03])
        + torch.tensor([-0.01, -0.02, -0.01]),
    )[steps - 1]
    for row, item in enumerate(terminal.output.items):
        assert item.request_id == f"request-{row}"
        assert tuple(sequence.token_ids for sequence in item.sequences) == tuple(
            tuple(token + row * 100 for token in tokens) for tokens in expected_tokens
        )
        assert tuple(sequence.parent_beam_indices for sequence in item.sequences) == (
            expected_parents if include_parents else ((), (), ())
        )
        assert tuple(sequence.cumulative_logprob for sequence in item.sequences) == tuple(
            expected_scores.tolist()
        )
        assert all(sequence.finish_reason == "length" for sequence in item.sequences)
    assert advance.call_count == (steps - 1 if include_parents else 0)
    history_allocations = [
        call for call in zeros_like.call_args_list if call.args[0].data_ptr() == sequence_pointer
    ]
    assert len(history_allocations) == int(include_parents and steps > 1)
    prefill_history_allocations = [
        call for call in zeros.call_args_list if call.args == ((batch_size, 3, 1),)
    ]
    assert len(prefill_history_allocations) == int(include_parents and steps == 1)
    assert stage.parent_history is None
    assert context.registry.num_active_sessions == 0
    assert not getattr(worker, "_beam_sessions", {})
    assert context.decision.sequence.data_ptr() == sequence_pointer
    assert context.decision.beam_scores.data_ptr() == scores_pointer


@pytest.mark.parametrize("include_parents", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("include_stop", [False, True])  # type: ignore[misc]
def test_npu_terminal_parent_transfer_and_stop_alignment(
    monkeypatch: pytest.MonkeyPatch, include_parents: bool, include_stop: bool
) -> None:
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.session_request = _parent_history_request(1, include_parents)
    stage.session_request.beam_params.stop_token_ids = (99,)
    stage.session_request.output_options.include_stop_token = include_stop
    stage.pending_plan = SimpleNamespace(execution=SimpleNamespace(request_ids=("request-0",)))
    cat, tolist = torch.cat, torch.Tensor.tolist
    pieces: list[tuple[int, ...]] = []
    converted: list[tuple[int, ...]] = []

    def record_cat(tensors: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        pieces.append(tuple(tensor.numel() for tensor in tensors))
        return cat(tensors, *args, **kwargs)

    def record_tolist(tensor: torch.Tensor) -> Any:
        converted.append(tuple(tensor.shape))
        return tolist(tensor)

    monkeypatch.setattr(torch, "cat", record_cat)
    monkeypatch.setattr(torch.Tensor, "tolist", record_tolist)
    result = stage._build_terminal_result(
        torch.tensor([[[10, 99, 12], [20, 21, 22], [30, 31, 32]]], dtype=torch.int32),
        torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32),
        torch.tensor([[[0, 2, 1], [0, 0, 2], [0, 1, 0]]], dtype=torch.int32)
        if include_parents
        else None,
    )
    assert result.output is not None
    first = result.output.items[0].sequences[0]
    assert first.token_ids == ((10, 99) if include_stop else (10,))
    assert first.parent_beam_indices == (
        ((0, 2) if include_stop else (0,)) if include_parents else ()
    )
    assert (first.finish_reason, first.stop_token_id) == ("stop", 99)
    expected_pieces: list[tuple[int, ...]] = [(9, 9, 3)] if include_parents else [(9, 3)]
    expected_converted: list[tuple[int, ...]] = (
        [(1, 3, 3), (1, 3, 3), (1, 3)] if include_parents else [(1, 3, 3), (1, 3)]
    )
    assert pieces == expected_pieces
    assert converted == expected_converted


def test_npu_missing_parent_history_is_rejected_only_when_requested() -> None:
    stage = object.__new__(NPUEagerBeamStageRunner)
    assert stage._needs_parent_history()
    stage.session_request = None
    assert stage._needs_parent_history()
    stage.session_request = _parent_history_request(1, True)
    stage.pending_plan = SimpleNamespace(execution=SimpleNamespace(request_ids=("request-0",)))
    sequences = torch.ones(1, 3, 2, dtype=torch.int32)
    scores = torch.zeros(1, 3)
    with pytest.raises(RuntimeError, match="missing requested parent history"):
        stage._build_terminal_result(sequences, scores, None)
    with pytest.raises(RuntimeError, match="inconsistent shapes"):
        stage._build_terminal_result(sequences, scores, torch.zeros(1, 3, dtype=torch.int32))
    stage.session_request.output_options.include_parent_beam_indices = False
    # A supplied legacy history is ignored entirely when not requested.
    result = stage._build_terminal_result(sequences, scores, object())
    assert result.output is not None
    assert all(not sequence.parent_beam_indices for sequence in result.output.items[0].sequences)


@pytest.mark.parametrize("widths", [(3, 2, 1), (1, 2, 3), None])  # type: ignore[misc]
@pytest.mark.parametrize("include_parents", [False, True])  # type: ignore[misc]
def test_npu_dynamic_parent_history_is_not_advanced_twice(
    monkeypatch: pytest.MonkeyPatch,
    widths: tuple[int, ...] | None,
    include_parents: bool,
) -> None:
    """Dynamic decisions own lineage even when only the return width differs."""
    from unittest.mock import Mock

    request = _parent_history_request(2, include_parents)
    request.beam_params.dynamic_beam_widths = widths
    request.beam_params.ignore_stop_tokens = True
    result_width = 1 if widths == (3, 2, 1) else 2
    request.output_options.num_return_sequences = result_width
    scheduled = _eager_stage_output(kind="decode", batch_size=2, beam_width=3, decode_step=1)
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    metadata.beam_params = request.beam_params
    metadata.output_options = request.output_options
    metadata.is_last_stage, metadata.output_device_state = True, None
    plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
    execution = plan.execution
    candidate_shape = (execution.num_execution_rows, execution.candidate_top_k)
    candidates = ConstraintOutput(
        token_ids=torch.zeros(candidate_shape, dtype=torch.int32),
        logprobs=torch.zeros(candidate_shape),
        valid_mask=torch.ones(candidate_shape, dtype=torch.bool),
    )
    sequences = torch.tensor(
        [[[10, 11, 12], [20, 21, 22]], [[110, 111, 112], [120, 121, 122]]],
        dtype=torch.int32,
    )[:, :result_width]
    parents = torch.tensor([[[0, 0, 1], [0, 0, 0]]] * 2, dtype=torch.int32)[:, :result_width]
    scores = torch.tensor([[-0.25, -0.5]] * 2)[:, :result_width]
    decision = {
        "out_token_ids": sequences[:, :, -1],
        "out_sequence": sequences,
        "out_log_probs": scores,
        "out_parent_history": parents,
    }
    dynamic_select = Mock(return_value=decision)
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.session_request, stage.pending_plan, stage.execution_error = request, plan, None
    stage.parent_history = None
    stage.dynamic_state = SimpleNamespace(
        select=dynamic_select,
        sequence=torch.zeros(2, execution.input_width, 3, dtype=torch.int32),
    )
    stage.attention_batch = SimpleNamespace(
        sessions=tuple(SimpleNamespace(session_id=f"request-{row}") for row in range(2)),
        end_session=Mock(),
    )
    stage.worker = SimpleNamespace(
        _current_scheduler_output=scheduled,
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(max_rows=execution.num_execution_rows),
            select=Mock(return_value=candidates),
        ),
    )
    stage.logits = torch.zeros(execution.num_execution_rows, 128)
    native = Mock(side_effect=AssertionError("dynamic lineage must not use the native decision"))
    advance = Mock(side_effect=AssertionError("dynamic lineage must not be advanced twice"))
    monkeypatch.setattr(
        "vllm_gr.v1.worker.npu.npu_beam_stage_runner.run_beam_decision_candidates", native
    )
    monkeypatch.setattr(stage, "_advance_parent_history", advance)
    build_terminal = stage._build_terminal_result

    def check_terminal_history(*args: Any) -> Any:
        # Inspect before terminal cleanup clears the session's history.
        assert (stage.parent_history is not None) is include_parents
        if include_parents:
            assert stage.parent_history is parents
        return build_terminal(*args)

    terminal = Mock(side_effect=check_terminal_history)
    monkeypatch.setattr(stage, "_build_terminal_result", terminal)

    assert stage._uses_dynamic_state(plan)
    result = stage.sample().gr_worker_results["batch-0"].terminal_result

    dynamic_select.assert_called_once_with(
        candidates,
        execution.input_width,
        execution.output_width,
        2,
        is_last=True,
        result_width=result_width,
    )
    native.assert_not_called()
    advance.assert_not_called()
    terminal.assert_called_once()
    if include_parents:
        torch.testing.assert_close(terminal.call_args.args[2], parents, rtol=0, atol=0)
    else:
        assert terminal.call_args.args[2] is None
    assert result is not None and result.failure is None and result.output is not None
    for row, item in enumerate(result.output.items):
        assert item.request_id == f"request-{row}"
        assert tuple(sequence.token_ids for sequence in item.sequences) == tuple(
            tuple(tokens) for tokens in sequences[row].tolist()
        )
        assert tuple(sequence.cumulative_logprob for sequence in item.sequences) == tuple(
            scores[row].tolist()
        )
        assert tuple(sequence.parent_beam_indices for sequence in item.sequences) == (
            tuple(tuple(history) for history in parents[row].tolist())
            if include_parents
            else ((),) * result_width
        )
        assert all(
            (sequence.finish_reason, sequence.stop_token_id) == ("length", None)
            for sequence in item.sequences
        )
    assert stage.closed and stage.parent_history is None
    stage.attention_batch.end_session.assert_called_once()


@pytest.mark.parametrize(  # type: ignore[misc]
    ("rows", "top_k"), [(4, 128), (128, 128), (512, 128), (512, 64)]
)
@pytest.mark.parametrize("scalar_temperature", [False, True])  # type: ignore[misc]
def test_npu_constraint_candidates_single_call_preserves_tensors(
    rows: int, top_k: int, scalar_temperature: bool
) -> None:
    """B4 Decode and smaller stage widths use one call without repacking."""
    logits = torch.arange(rows, dtype=torch.float32)[:, None].expand(-1, 128).contiguous()
    metadata = ConstraintMetadata(
        prefix_token_ids=torch.arange(rows * 3, dtype=torch.int32).view(rows, 3),
        temperatures=torch.ones(1 if scalar_temperature else rows, dtype=torch.float32),
        current_step=0 if rows == 4 else 2,
        top_k=top_k,
    )
    expected = ConstraintOutput(
        token_ids=torch.arange(rows * top_k, dtype=torch.int32).view(rows, top_k),
        logprobs=torch.zeros(rows, top_k),
        valid_mask=torch.ones(rows, top_k, dtype=torch.bool),
    )
    calls = 0

    def select(
        actual_logits: torch.Tensor, actual_metadata: ConstraintMetadata
    ) -> ConstraintOutput:
        nonlocal calls
        calls += 1
        assert actual_logits is logits
        assert actual_metadata is metadata
        return expected

    stage = object.__new__(NPUEagerBeamStageRunner)
    result = stage._select_constraint_candidates(
        SimpleNamespace(info=SimpleNamespace(max_rows=512), select=select), logits, metadata
    )

    assert calls == 1
    assert result is expected


@pytest.mark.parametrize("rows,max_rows", [(6, 3), (512, 128), (512, 192)])  # type: ignore[misc]
@pytest.mark.parametrize("scalar_temperature", [False, True])  # type: ignore[misc]
def test_npu_constraint_candidates_chunk_multi_batch_rows(
    rows: int, max_rows: int, scalar_temperature: bool
) -> None:
    calls: list[ConstraintMetadata] = []

    def select(logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        calls.append(metadata)
        row_ids = logits[:, 0].to(torch.int32)
        return ConstraintOutput(
            token_ids=row_ids[:, None].expand(-1, metadata.top_k).contiguous(),
            logprobs=logits[:, : metadata.top_k].contiguous(),
            valid_mask=torch.ones((logits.shape[0], metadata.top_k), dtype=torch.bool),
        )

    top_k = 2 if rows == 6 else 128
    prefixes = torch.arange(rows * 3, dtype=torch.int32).view(rows, 3)
    temperatures = torch.arange(1 if scalar_temperature else rows, dtype=torch.float32)
    metadata = ConstraintMetadata(
        prefix_token_ids=prefixes,
        temperatures=temperatures,
        current_step=1,
        top_k=top_k,
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    result = stage._select_constraint_candidates(
        SimpleNamespace(info=SimpleNamespace(max_rows=max_rows), select=select),
        torch.arange(rows, dtype=torch.float32)[:, None].expand(-1, max(top_k, 4)),
        metadata,
    )

    assert [item.prefix_token_ids.shape[0] for item in calls] == [
        min(max_rows, rows - start) for start in range(0, rows, max_rows)
    ]
    torch.testing.assert_close(torch.cat([item.prefix_token_ids for item in calls]), prefixes)
    if scalar_temperature:
        assert all(item.temperatures is temperatures for item in calls)
    else:
        torch.testing.assert_close(torch.cat([item.temperatures for item in calls]), temperatures)
    assert all(item.current_step == 1 and item.top_k == top_k for item in calls)
    torch.testing.assert_close(result.token_ids[:, 0], torch.arange(rows, dtype=torch.int32))
    assert result.logprobs.is_contiguous() and result.valid_mask.all()


@pytest.mark.parametrize("ignore_stop_tokens", [False, True])  # type: ignore[misc]
def test_npu_dynamic_beam_admission_requires_fixed_horizon(ignore_stop_tokens: bool) -> None:
    from vllm_gr.v1.beam.batch_contracts import BeamBatchParams
    from vllm_gr.v1.beam.dispatch import _validate_gr_generation_params

    params = BeamBatchParams(
        beam_width=4,
        max_tokens=3,
        temperature=0.0,
        stop_token_ids=(99,),
        ignore_stop_tokens=ignore_stop_tokens,
        dynamic_beam_widths=(4, 2, 1),
    )
    if ignore_stop_tokens:
        _validate_gr_generation_params(params)
    else:
        with pytest.raises(ValueError, match="requires ignore_eos=True"):
            _validate_gr_generation_params(params)


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
def test_dynamic_npu_cases_use_public_engine_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    from tests.helpers.gr_config import npu_dynamic_engine_args
    from vllm_gr.config.parser import load_engine_gr_config
    from vllm_gr.engine.arg_utils_gr import (
        _gr_engine_args_init,
        origin_engine_args_init,
    )

    observed: list[Any] = []

    def probe(**kwargs: Any) -> None:
        args = EngineArgs.__new__(EngineArgs)
        _gr_engine_args_init(origin_engine_args_init)(args, **kwargs)
        assert args.dtype == "float16"
        config = load_engine_gr_config(None, {}, args.gr)
        assert args.attention_config.backend.name == "CUSTOM"
        assert args.enforce_eager is not config.beam_graph_enabled
        assert config.beam_max_width == 32
        assert config.deployment.candidate_width == (32, 32, 32)
        assert config.constraint is not None and config.constraint.table is not None
        assert config.constraint.table.backend.value == "npu"
        observed.append(config)

    for graph, explicit_widths in ((False, True), (True, True), (True, False)):
        probe(
            **npu_dynamic_engine_args(
                str(tmp_path),
                tmp_path / "table.safetensors",
                "sha256:" + "a" * 64,
                "sha256:" + "b" * 64,
                beam_width=32,
                graph=graph,
                explicit_widths=explicit_widths,
            )
        )
    assert [config.beam_graph_enabled for config in observed] == [False, True, True]
    assert [config.beam_graph_widths for config in observed] == [(8, 16, 32), (8, 16, 32), None]
