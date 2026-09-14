# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for NPU ModelRunner wrappers."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.model_runner_common import InputBatchProxy


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


def test_npu_bookkeeping_materializes_batched_sync_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    original_batch = SimpleNamespace()
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(_obj=original_batch),
        use_async_scheduling=False,
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[11], [21]], dtype=torch.int32),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
            logprobs=torch.tensor(
                [[-0.1, -0.2], [-0.3, -0.4]], dtype=torch.float32
            ),
            selected_token_ranks=torch.tensor([0, 0], dtype=torch.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        beam_data={"beam-0": {}, "beam-1": {}},
        _gr_request_row_configs=[
            ("beam-0", 0, 1, [0]),
            ("beam-1", 1, 1, [1]),
        ],
        finished_req_ids=set(),
    )

    def original(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return (object(), [[99], [98]], {}, ["physical"], {"physical": 0}, [])

    monkeypatch.setattr(
        npu_patch,
        "_reduce_sampler_output_for_worker_decision",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        npu_patch,
        "_beam_collapse_bookkeeping",
        lambda *_args: (["beam-0", "beam-1"], {"beam-0": 0, "beam-1": 1}, 2),
    )

    result = npu_patch.make_patched_bookkeeping_sync(original)(
        runner,
        scheduler_output,
        sampler_output,
    )

    assert result[0].slice_request(0, 1).logprob_token_ids.tolist() == [[11, 12]]
    assert result[0].slice_request(1, 1).logprob_token_ids.tolist() == [[21, 22]]
    assert result[1] == [[11], [21]]
    assert result[3:5] == (
        ["beam-0", "beam-1"],
        {"beam-0": 0, "beam-1": 1},
    )
    assert runner.input_batch is original_batch


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
        lambda runner, session, sampler_output, is_last_step, result_width: decision,
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
    """The final step applies no reorder (ACS skips select_unshared_kv)."""
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
    beam_data = {
        req_id: ({"is_beam_decode": True, "decode_step": 0} if req_id in beam_keys else {})
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
    npu_patch._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.FULL
        assert result[1].num_tokens == 4
        assert result[1].uniform is True
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


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

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=8))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},  # captured at 4, batch is 8 tokens
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
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
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

    session = SimpleNamespace(end_session=lambda: None)
    monkeypatch.setattr(
        npu_patch,
        "_capture_create_dummy_context_npu",
        lambda *args, **kwargs: (object(), session, object(), object()),
    )

    inference_flags: list[bool] = []

    def fake_forward(*args: Any, **kwargs: Any) -> None:
        inference_flags.append(torch.is_inference_mode_enabled())

    monkeypatch.setattr(npu_patch, "_dummy_beam_forward_npu", fake_forward)
    monkeypatch.setattr(npu_patch, "_verify_and_record_captured_graph_npu", lambda r, width: True)

    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_sessions={},
        _current_beam_info=None,
    )
    assert npu_patch.capture_beam_decode_graph(runner) is True
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
