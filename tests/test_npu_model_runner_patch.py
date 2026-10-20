# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for NPU ModelRunner wrappers."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.v1.beam.npu_execution_batch import (
    NPUBeamExecutionBatch,
    restore_npu_cache_pool_layout,
)
from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.model_runner_common import InputBatchProxy
from vllm_gr.v1.worker.npu_beam_stage_adapter import (
    NPUEagerFirstDecodeHandoff,
    NPUEagerStagePlan,
    validate_npu_eager_capabilities,
)
from vllm_gr.v1.worker.npu_beam_stage_runner import NPUEagerBeamStageRunner


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


def test_npu_prepare_unproxies_and_preserves_four_item_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_batch = SimpleNamespace(req_ids=["request"])
    proxy = InputBatchProxy(input_batch, ["request"], [0], [0])
    spec_metadata = object()
    compressed = object()
    runner = SimpleNamespace(
        input_batch=proxy,
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )

    def original(
        self: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[object, object, int, object]:
        assert self.input_batch is input_batch
        assert self._current_beam_info["req_ids"] is input_batch.req_ids
        return object(), spec_metadata, 7, compressed

    monkeypatch.setattr(npu_patch, "_beam_remap_inputs", lambda *args: [3])
    wrapped = npu_patch.make_patched_prepare_inputs(original)
    assert npu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary[1:] == (spec_metadata, 7, compressed)

    runner.input_batch = proxy
    runner._current_beam_info["req_ids"] = []
    logits, returned_spec, total, returned_compressed = wrapped(
        runner,
        SimpleNamespace(beam_data={"request": {}}),
        [1],
    )
    assert logits.dtype is torch.int32
    assert logits.tolist() == [3]
    assert returned_spec is spec_metadata
    assert total == 7
    assert returned_compressed is compressed


def test_npu_bookkeeping_preserves_six_items_and_restores_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_batch = SimpleNamespace()
    proxy = SimpleNamespace(_obj=original_batch)
    runner = SimpleNamespace(
        input_batch=original_batch,
        _beam_logprobs_cache={"finished": {}, "active": {}},
    )
    ordinary_result = (["lp"], ["valid"], object(), ["request"], {"request": 0}, [3])
    sampler_output = object()
    reduced: list[object] = []

    def original(
        self: Any,
        scheduler_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        if getattr(scheduler_output, "fail", False):
            raise RuntimeError("boom")
        if getattr(scheduler_output, "beam_data", {}):
            assert reduced == [sampler_output]
        return ordinary_result

    def collapse(
        self: Any,
        orig_batch: Any,
        configs: Any,
        beam_data: Any,
        sampler: Any,
    ) -> tuple[list[str], dict[str, int], int]:
        assert orig_batch is original_batch
        assert sampler is sampler_output
        return ["beam"], {"beam": 0}, 1

    monkeypatch.setattr(npu_patch, "_beam_collapse_bookkeeping", collapse)
    monkeypatch.setattr(
        npu_patch,
        "_reduce_sampler_output_for_worker_decision",
        lambda _runner, output: reduced.append(output),
    )
    wrapped = npu_patch.make_patched_bookkeeping_sync(original)

    assert wrapped(runner, SimpleNamespace(beam_data={}), sampler_output) is ordinary_result

    beam_output = SimpleNamespace(
        beam_data={"beam": {}},
        _gr_request_row_configs=[("beam", 0, 1, [0])],
        finished_req_ids={"finished"},
    )
    runner.input_batch = proxy
    result = wrapped(runner, beam_output, sampler_output)
    assert len(result) == 6
    assert result[:2] == ([[]], [[]])
    assert result[3:5] == (["beam"], {"beam": 0})
    assert runner.input_batch is original_batch
    assert "finished" not in runner._beam_logprobs_cache

    runner.input_batch = proxy
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(
            runner,
            SimpleNamespace(
                beam_data={"beam": {}},
                _gr_request_row_configs=[],
                fail=True,
            ),
            sampler_output,
        )
    assert runner.input_batch is original_batch


def test_npu_bookkeeping_collapses_beam_rows_before_host_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_batch = SimpleNamespace()
    runner = SimpleNamespace(input_batch=SimpleNamespace(_obj=original_batch))
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.arange(128, dtype=torch.int32).reshape(128, 1)
    )
    scheduler_output = SimpleNamespace(
        beam_data={"beam": {}},
        _gr_request_row_configs=[("beam", 0, 128, list(range(128)))],
    )

    def original(self: Any, output: Any, sampler: Any) -> tuple[Any, ...]:
        assert sampler.sampled_token_ids.shape == (1, 1)
        return (None, [[]], None, ["beam"], {"beam": 0}, [])

    monkeypatch.setattr(
        npu_patch,
        "_beam_collapse_bookkeeping",
        lambda *_args: (["beam"], {"beam": 0}, 1),
    )
    wrapped = npu_patch.make_patched_bookkeeping_sync(original)
    result = wrapped(runner, scheduler_output, sampler_output)

    assert result[1] == [[]]
    assert sampler_output.sampled_token_ids.tolist() == [[0]]


def test_npu_sample_patch_is_signature_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NPU legacy branch tolerates a different positional signature."""
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER

    events: list[str] = []
    sample_output = object()

    def original(self: Any, logits: Any, extra: Any) -> Any:
        assert extra == "extra"
        events.append("sample")
        return sample_output

    def override(self: Any, logits: Any, sampler_output: Any) -> None:
        assert self is runner
        assert logits == "logits"
        assert sampler_output is sample_output
        events.append("override")

    monkeypatch.setattr(npu_patch, "apply_constrained_sample_override", override)
    # Short-circuit gate off: no beam decision active → legacy path.
    monkeypatch.setattr("vllm_gr.v1.beam.beam_decision.worker_decision_enabled", lambda r: False)
    wrapped = npu_patch.make_patched_sample_npu(original)
    assert getattr(wrapped, _WORKER_SAMPLE_PATCH_MARKER, False)
    assert npu_patch.make_patched_sample_npu(wrapped) is wrapped

    runner = SimpleNamespace()
    assert wrapped(runner, "logits", "extra") is sample_output
    assert events == ["sample", "override"]


def test_npu_sample_patch_parses_keyword_logits_and_spec_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any, Any]] = []
    sample_output = object()

    def direct(self: Any, logits: Any, spec_decode_metadata: Any) -> None:
        events.append(("direct", logits, spec_decode_metadata))
        return None

    def original(
        self: Any,
        *,
        logits: Any,
        spec_decode_metadata: Any = None,
    ) -> Any:
        events.append(("sample", logits, spec_decode_metadata))
        return sample_output

    def override(self: Any, logits: Any, sampler_output: Any) -> None:
        assert sampler_output is sample_output
        events.append(("override", logits, None))

    monkeypatch.setattr(npu_patch, "try_sample_with_constraint_backend", direct)
    monkeypatch.setattr(npu_patch, "apply_constrained_sample_override", override)
    wrapped = npu_patch.make_patched_sample_npu(original)

    assert (
        wrapped(
            SimpleNamespace(),
            logits="keyword-logits",
            spec_decode_metadata="keyword-spec",
        )
        is sample_output
    )
    assert events == [
        ("direct", "keyword-logits", "keyword-spec"),
        ("sample", "keyword-logits", "keyword-spec"),
        ("override", "keyword-logits", None),
    ]


def test_npu_sample_patch_returns_canonical_output_before_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_output = object()
    observed: list[tuple[Any, Any]] = []

    def direct(self: Any, logits: Any, spec_decode_metadata: Any) -> Any:
        observed.append((logits, spec_decode_metadata))
        return direct_output

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"ordinary sampler or legacy override ran: {args} {kwargs}")

    monkeypatch.setattr(npu_patch, "try_sample_with_constraint_backend", direct)
    monkeypatch.setattr(npu_patch, "apply_constrained_sample_override", forbidden)
    wrapped = npu_patch.make_patched_sample_npu(forbidden)

    assert (
        wrapped(
            SimpleNamespace(),
            logits="keyword-logits",
            spec_decode_metadata="keyword-spec",
        )
        is direct_output
    )
    assert observed == [("keyword-logits", "keyword-spec")]


# ---------------------------------------------------------------------------
# patched_sample — constrained sampler short-circuit (P1-1)
# ---------------------------------------------------------------------------


def _shortcut_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
    rows: list[Any] | None,
) -> None:
    """Stub the short-circuit gates with controllable results.

    ``rows`` non-empty simulates a homogeneous constrained step; the built
    sampler output mirrors ``build_constrained_sampler_output``'s contract
    ([rows, W] tensors, top-1 candidate in column 0).
    """
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    from vllm_gr.v1.beam import beam_decision
    from vllm_gr.v1.worker import constraint_hook

    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda r: enabled)

    def fake_build(runner: Any, logits: Any, collected_rows: list[Any]) -> Any:
        width = 4
        top_tokens = torch.arange(2 * width, dtype=torch.int32).reshape(2, width)
        top_logprobs = -torch.rand(2, width, dtype=torch.float32)
        return SamplerOutput(
            sampled_token_ids=top_tokens[:, :1].clone(),
            logprobs_tensors=LogprobsTensors(
                top_tokens, top_logprobs, torch.ones(2, dtype=torch.int32)
            ),
        )

    if rows is None:
        monkeypatch.setattr(constraint_hook, "collect_constrained_override_rows", lambda r: [])
        monkeypatch.setattr(constraint_hook, "build_constrained_sampler_output", fake_build)
    else:
        monkeypatch.setattr(constraint_hook, "collect_constrained_override_rows", lambda r: rows)
        monkeypatch.setattr(constraint_hook, "build_constrained_sampler_output", fake_build)


def test_npu_sample_shortcircuits_original_sampler_when_constrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker decision + constrained rows → _orig_sample never runs."""
    _shortcut_stub(monkeypatch, enabled=True, rows=[("s", 0, 4, {"beam_width": 4})])

    calls: list[str] = []

    def original(self: Any, logits: Any, spec: Any = None) -> Any:
        calls.append("orig_sample")
        raise AssertionError("original sampler must not run on the shortcut path")

    def override(self: Any, logits: Any, sampler_output: Any) -> None:
        calls.append("override")
        raise AssertionError("override must not run on the shortcut path")

    monkeypatch.setattr(npu_patch, "apply_constrained_sample_override", override)
    monkeypatch.setattr(
        npu_patch, "_maybe_run_worker_beam_decision", lambda r, s: calls.append("decision")
    )

    wrapped = npu_patch.make_patched_sample_npu(original)
    logits = torch.zeros(2, 16)
    out = wrapped(SimpleNamespace(), logits)

    assert calls == ["decision"]  # only the beam decision ran
    assert out.sampled_token_ids.dtype is torch.int32
    assert out.sampled_token_ids.shape == (2, 1)
    assert out.logprobs_tensors.logprob_token_ids.shape == (2, 4)
    assert out.logprobs_tensors.logprobs.dtype is torch.float32
    # Top-1 candidate sits in column 0 of both the sampled ids and the
    # candidate row (the _select_candidates `n_cols == W` contract).
    assert torch.equal(out.sampled_token_ids[:, 0], out.logprobs_tensors.logprob_token_ids[:, 0])
    assert torch.equal(out.logprobs_tensors.selected_token_ranks, torch.ones(2, dtype=torch.int32))


def test_npu_sample_shortcircuit_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every gate miss keeps the untouched legacy path."""
    logits = torch.zeros(2, 16)
    sample_output = object()
    events: list[str] = []

    def original(self: Any, logits: Any, spec: Any = None) -> Any:
        events.append("orig_sample")
        return sample_output

    def override(self: Any, logits: Any, sampler_output: Any) -> None:
        events.append("override")

    monkeypatch.setattr(npu_patch, "apply_constrained_sample_override", override)
    monkeypatch.setattr(
        npu_patch, "_maybe_run_worker_beam_decision", lambda r, s: events.append("decision")
    )
    wrapped = npu_patch.make_patched_sample_npu(original)

    # Gate 1: worker decision disabled (even with constrained rows present).
    _shortcut_stub(monkeypatch, enabled=False, rows=[("s", 0, 4, {"beam_width": 4})])
    assert wrapped(SimpleNamespace(), logits) is sample_output
    assert events == ["orig_sample", "override", "decision"]

    # Gate 2: decision enabled but no constrained rows.
    events.clear()
    _shortcut_stub(monkeypatch, enabled=True, rows=None)
    assert wrapped(SimpleNamespace(), logits) is sample_output
    assert events == ["orig_sample", "override", "decision"]

    # Gate 3: spec-decode metadata present.
    events.clear()
    _shortcut_stub(monkeypatch, enabled=True, rows=[("s", 0, 4, {"beam_width": 4})])
    assert wrapped(SimpleNamespace(), logits, object()) is sample_output
    assert events == ["orig_sample", "override", "decision"]

    # Gate 4: logits is None (no tensor to feed the operator).
    events.clear()
    _shortcut_stub(monkeypatch, enabled=True, rows=[("s", 0, 4, {"beam_width": 4})])
    assert wrapped(SimpleNamespace(), None) is sample_output
    assert events == ["orig_sample", "override", "decision"]


def test_npu_sample_shortcircuit_build_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator failure inside the shortcut falls back to the sampler."""
    from vllm_gr.v1.beam import beam_decision
    from vllm_gr.v1.worker import constraint_hook

    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda r: True)
    monkeypatch.setattr(
        constraint_hook, "collect_constrained_override_rows", lambda r: [("s", 0, 4, {})]
    )

    def boom(runner: Any, logits: Any, rows: list[Any]) -> Any:
        raise RuntimeError("operator unavailable")

    monkeypatch.setattr(constraint_hook, "build_constrained_sampler_output", boom)

    sample_output = object()
    events: list[str] = []

    def original(self: Any, logits: Any, spec: Any = None) -> Any:
        events.append("orig_sample")
        return sample_output

    monkeypatch.setattr(
        npu_patch, "apply_constrained_sample_override", lambda *a: events.append("override")
    )
    wrapped = npu_patch.make_patched_sample_npu(original)

    assert wrapped(SimpleNamespace(), torch.zeros(2, 16)) is sample_output
    assert events == ["orig_sample", "override"]


# ---------------------------------------------------------------------------
# _maybe_run_worker_beam_decision — kv_reorder record derivation
# ---------------------------------------------------------------------------


def _decision_stub(
    monkeypatch: pytest.MonkeyPatch,
    decision: dict[str, Any] | None,
) -> None:
    """Stub the beam-decision gate + runner with a canned decision dict."""
    from vllm_gr.v1.beam import beam_decision

    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda r: decision is not None)
    monkeypatch.setattr(
        beam_decision,
        "run_beam_decision",
        lambda runner,
        session,
        sampler_output,
        is_last_step,
        result_width,
        v2_layout=False: decision,
    )


def _decision_runner(is_last_step: bool) -> tuple[Any, Any]:
    state = SimpleNamespace(is_last_step=is_last_step, kv_reorder=None)
    session = SimpleNamespace(_state=state)
    runner = SimpleNamespace(
        _current_beam_info={
            "beam_session": session,
            "beam_data": {"s": {"is_beam_decode": True, "result_width": None}},
        },
        _beam_decision_result="unset",
    )
    return runner, state


def test_npu_worker_decision_derives_kv_reorder_from_host_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kv_reorder is derived from the return channel's host copy.

    ``_finalize_decision`` no longer materializes the record (its
    ``local_idx[0].tolist()`` was a per-step device→host sync, ~30ms on NPU).
    The return channel already copies ``out_token_index`` to host for the
    frontend; its request-0 row carries global offset 0, so it IS the
    request-local mapping applied by ``_reorder_kv_same_step``.
    """
    beam_width = 4
    decision = {
        "out_token_ids": torch.zeros(1, beam_width, dtype=torch.int32),
        "out_token_index": torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
        "out_log_probs": torch.zeros(1, beam_width, dtype=torch.float32),
    }
    _decision_stub(monkeypatch, decision)
    runner, state = _decision_runner(is_last_step=False)

    npu_patch._maybe_run_worker_beam_decision(runner, object())

    result = runner._beam_decision_result
    # is_last_step is always present: EngineCore engine-driven chaining
    # uses it as the stop-chaining signal; the frontend consumers ignore it.
    assert set(result) == {
        "out_token_ids",
        "out_token_index",
        "out_log_probs",
        "is_last_step",
        "stage",
        "session_id",
    }
    assert result["is_last_step"] is False
    # The return payload is nested Python lists: kv_transfer_params values
    # are decoded against ``dict[str, Any]``, so only plain containers
    # round-trip the IPC boundary (a tensor would arrive as the
    # unrecoverable ``['int64', shape, data]`` triple).
    assert result["out_token_ids"] == [[0, 0, 0, 0]]
    assert result["out_token_index"] == [[0, 1, 0, 1]]
    assert all(not torch.is_tensor(v) for v in result.values())
    assert state.kv_reorder == [0, 1, 0, 1]


def test_npu_worker_decision_skips_kv_reorder_on_last_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final step applies no reorder (Asynchronous scheduling skips select_unshared_kv)."""
    beam_width = 4
    decision = {
        "out_token_ids": torch.zeros(1, beam_width, dtype=torch.int32),
        "out_token_index": torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
        "out_log_probs": torch.zeros(1, beam_width, dtype=torch.float32),
        "out_sequence": torch.arange(2)
        .repeat(beam_width)
        .reshape(1, beam_width, 2)
        .to(torch.int32),
    }
    _decision_stub(monkeypatch, decision)
    runner, state = _decision_runner(is_last_step=True)

    npu_patch._maybe_run_worker_beam_decision(runner, object())

    result = runner._beam_decision_result
    assert "out_sequence" in result
    # Terminal payloads keep the full [B, W, steps] history as nested lists.
    assert result["out_sequence"] == [[[0, 1]] * beam_width]
    assert state.kv_reorder is None


def test_npu_worker_decision_disabled_clears_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate closed → no decision, no record."""
    _decision_stub(monkeypatch, None)
    runner, state = _decision_runner(is_last_step=False)

    npu_patch._maybe_run_worker_beam_decision(runner, object())

    assert runner._beam_decision_result is None
    assert state.kv_reorder is None


# ---------------------------------------------------------------------------
# patched_determine — FULL grant / guards (mirrors GPU-side contracts)
# ---------------------------------------------------------------------------


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
    import vllm_gr.v1.worker.model_runner_common as common

    monkeypatch.setattr(common, "_get_gr_config", lambda r: gr_cfg)


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


def test_npu_capture_reentry_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a graph for the width is already recorded, capture short-circuits True."""
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_graph_keys={4},
        _gr_npu_beam_graph_shapes={(1, 4)},
    )
    assert npu_patch.capture_beam_decode_graph(runner) is True


def test_npu_capture_forwards_run_under_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
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
    assert create_args[0][2] == 2
    assert runner._gr_npu_beam_graph_shapes == {(2, 4)}
    assert len(inference_flags) == npu_patch._BEAM_CAPTURE_NUM_WARMUPS + 1
    assert all(inference_flags), f"capture forwards ran outside inference_mode: {inference_flags}"


# ---------------------------------------------------------------------------
# _verify_and_record_captured_graph_npu — ACL entry verification
# ---------------------------------------------------------------------------
# These tests monkeypatch vllm_ascend.compilation.acl_graph.ACLGraphWrapper,
# which requires vllm-ascend to be installed (it is an optional dependency on
# GPU-only gates). Keep the skip local so the platform-neutral NPU wrapper
# contracts above still run on CPU/GPU CI.


def test_npu_verify_and_record_records_on_match(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip(
        "vllm_ascend.compilation.acl_graph",
        reason="vllm-ascend is not installed",
    )
    from unittest.mock import MagicMock

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    mock_entry = MagicMock()
    mock_entry.aclgraph = MagicMock()  # not None → captured

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = CUDAGraphMode.FULL
    mock_wrapper.concrete_aclgraph_entries = {
        BatchDescriptor(num_tokens=4, uniform=True): mock_entry
    }

    runner = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_ascend.compilation.acl_graph.ACLGraphWrapper._all_instances",
        [mock_wrapper],
    )
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=4) is True
    assert 4 in runner._beam_graph_keys


def test_npu_verify_and_record_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip(
        "vllm_ascend.compilation.acl_graph",
        reason="vllm-ascend is not installed",
    )
    from unittest.mock import MagicMock

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = MagicMock()  # not FULL
    mock_wrapper.concrete_aclgraph_entries = {}

    runner = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_ascend.compilation.acl_graph.ACLGraphWrapper._all_instances",
        [mock_wrapper],
    )
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=4) is False
    assert not hasattr(runner, "_beam_graph_keys")


def test_npu_verify_and_record_appends_to_existing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip(
        "vllm_ascend.compilation.acl_graph",
        reason="vllm-ascend is not installed",
    )
    from unittest.mock import MagicMock

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    mock_entry = MagicMock()
    mock_entry.aclgraph = MagicMock()

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = CUDAGraphMode.FULL
    mock_wrapper.concrete_aclgraph_entries = {
        BatchDescriptor(num_tokens=8, uniform=True): mock_entry
    }

    runner = SimpleNamespace(_beam_graph_keys={4})
    monkeypatch.setattr(
        "vllm_ascend.compilation.acl_graph.ACLGraphWrapper._all_instances",
        [mock_wrapper],
    )
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=8) is True
    assert runner._beam_graph_keys == {4, 8}


def test_npu_eager_decode_binds_winners_and_sampling_rows() -> None:
    prefill = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="prefill"))
    plan = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="decode"))
    input_ids = torch.full((8,), -1, dtype=torch.int32)
    positions = torch.full((8,), -1, dtype=torch.int64)
    winners = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    handoff = NPUEagerFirstDecodeHandoff.from_prefill(prefill, winners)

    handoff.bind_first_decode(plan, input_ids, positions)

    assert input_ids.tolist() == [11, 12, 13, 21, 22, 23, -1, -1]
    assert positions.tolist() == [2, 2, 2, 3, 3, 3, -1, -1]
    source = SimpleNamespace(
        temperature=torch.tensor([0.5, 0.9]),
        top_p=torch.tensor([0.8, 0.7]),
    )
    remapped = plan.remap_sampling_metadata(source)
    assert remapped is not source
    assert remapped.temperature.tolist() == pytest.approx([0.5] * 3 + [0.9] * 3)
    assert remapped.top_p.tolist() == pytest.approx([0.8] * 3 + [0.7] * 3)
    assert source.temperature.tolist() == pytest.approx([0.5, 0.9])


def test_npu_eager_attention_layout_keeps_request_ownership() -> None:
    plan = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="decode"))
    layout = plan.build_decode_attention_layout(
        block_ids_by_request=((10, 11), (20, 21)),
        pool_slots=(4, 7),
    )

    assert layout.row_request_indices == (0, 0, 0, 1, 1, 1)
    assert layout.row_beam_indices == (0, 1, 2, 0, 1, 2)
    assert layout.query_start_locations == tuple(range(7))
    assert layout.prefix_sequence_lengths == (2, 2, 2, 3, 3, 3)
    assert layout.suffix_sequence_lengths == (1,) * 6
    assert layout.prefix_block_tables[:3] == ((10, 11),) * 3
    assert layout.prefix_block_tables[3:] == ((20, 21),) * 3
    assert layout.pool_slots == (4, 7)


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


def test_npu_stage_runner_binds_physical_decode_without_mutating_scheduler() -> None:
    scheduler_output = _eager_stage_output(kind="decode")
    plan = NPUEagerStagePlan.from_scheduler_output(scheduler_output)
    prefill = NPUEagerStagePlan.from_scheduler_output(_eager_stage_output(kind="prefill"))
    winners = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    input_batch = SimpleNamespace(req_ids=list(plan.execution.request_ids))
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
    stage.handoff = NPUEagerFirstDecodeHandoff.from_prefill(prefill, winners)
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
        return object(), object(), 6, object()

    counts = np.asarray([1, 1], dtype=np.int32)
    logits, _, total, _ = stage.prepare_inputs(original, runtime_output, counts)

    assert counts.tolist() == [1, 1]
    assert scheduler_output.num_scheduled_tokens == {
        "request-0": 1,
        "request-1": 1,
    }
    assert worker.input_ids.gpu[:6].tolist() == [11, 12, 13, 21, 22, 23]
    assert worker.positions[:6].tolist() == [2, 2, 2, 3, 3, 3]
    assert logits.tolist() == list(range(6))
    assert total == 6
    assert isinstance(worker.input_batch, InputBatchProxy)


@pytest.mark.parametrize("device", ["cuda", "npu", "cpu"])  # type: ignore[misc]
def test_shared_stage_dispatch_routes_devices(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    from unittest.mock import Mock

    from vllm_gr.v1.worker import gpu_beam_stage_runner
    from vllm_gr.v1.worker import model_runner_common as common
    from vllm_gr.v1.worker import npu_beam_stage_runner

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
            common._dispatch_stage_frame(worker, scheduled, object(), hooks_type=hooks)
        cuda.assert_not_called()
        npu.assert_not_called()
    else:
        assert common._dispatch_stage_frame(worker, scheduled, object(), hooks_type=hooks) is output
        if device == "cuda":
            assert cuda.call_args.kwargs["hooks_type"] is hooks
            npu.assert_not_called()
        else:
            npu.assert_called_once()
            cuda.assert_not_called()


def test_npu_graph_context_survives_shape_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
    from vllm_gr.v1.worker import model_runner_common as common

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

    monkeypatch.setattr(common, "_create_beam_context", create)
    monkeypatch.setattr(common, "_get_gr_config", lambda worker: object())
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
    """Mirror Ascend 0.22.1rc1 call order using the real GR dispatch and NPU hooks."""
    from vllm.v1.cudagraph_dispatcher import CUDAGraphMode

    from vllm_gr.v1.worker import model_runner_common as common
    from vllm_gr.v1.worker import npu_beam_stage_runner

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
    stage.handoff = NPUEagerFirstDecodeHandoff.from_prefill(prefill, winners)
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
    monkeypatch.setattr(common, "_materialize_logits_indices", lambda _, rows: torch.tensor(rows))
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
        return object(), None, int(counts.sum()), None

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
        logits_indices, _, total, _ = self._prepare_inputs(runtime, counts)
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

    result = common._dispatch_stage_frame(worker, logical, native)
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

    from vllm_gr.v1.worker import model_runner_common as common
    from vllm_gr.v1.worker import npu_beam_stage_runner

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
        common._dispatch_stage_frame(worker, scheduled, fail)
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
    stage.handoff = None
    stage.latest_token_ids = None
    stage.attention_batch = None
    stage.closed = False

    output = stage.sample()

    result = output.gr_worker_results[plan.execution.session_id]
    assert result.owner_generation == plan.execution.owner_generation == 1


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


def test_npu_constraint_candidates_chunk_multi_batch_rows() -> None:
    calls: list[ConstraintMetadata] = []

    def select(logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        calls.append(metadata)
        row_ids = logits[:, 0].to(torch.int32)
        return ConstraintOutput(
            token_ids=row_ids[:, None].expand(-1, metadata.top_k).contiguous(),
            logprobs=logits[:, : metadata.top_k].contiguous(),
            valid_mask=torch.ones((logits.shape[0], metadata.top_k), dtype=torch.bool),
        )

    prefixes = torch.arange(18, dtype=torch.int32).view(6, 3)
    temperatures = torch.arange(6, dtype=torch.float32)
    metadata = ConstraintMetadata(
        prefix_token_ids=prefixes,
        temperatures=temperatures,
        current_step=1,
        top_k=2,
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    result = stage._select_constraint_candidates(
        SimpleNamespace(info=SimpleNamespace(max_rows=3), select=select),
        torch.arange(6, dtype=torch.float32)[:, None].expand(-1, 4),
        metadata,
    )

    assert [item.prefix_token_ids.shape[0] for item in calls] == [3, 3]
    torch.testing.assert_close(torch.cat([item.prefix_token_ids for item in calls]), prefixes)
    torch.testing.assert_close(torch.cat([item.temperatures for item in calls]), temperatures)
    assert all(item.current_step == 1 and item.top_k == 2 for item in calls)
    torch.testing.assert_close(result.token_ids[:, 0], torch.arange(6, dtype=torch.int32))
    assert result.logprobs.is_contiguous() and result.valid_mask.all()
