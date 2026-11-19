# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Core contracts for NPU ModelRunner wrappers."""

import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests._npu_model_runner_patch_helpers import (
    _cached_prefill_output,
    _decision_runner,
    _decision_stub,
    _eager_stage_output,
    _shortcut_stub,
    _terminal_result_runner,
)
from vllm_gr.v1.beam.execution_batch import BeamExecutionContractError
from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.npu_beam_stage_adapter import (
    NPUEagerStagePlan,
)
from vllm_gr.v1.worker.npu_beam_stage_runner import NPUEagerBeamStageRunner
from vllm_gr.v1.worker.npu_model_runner_inputs import InputBatchProxy


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


def test_npu_prepare_unproxies_and_preserves_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(npu_patch, "_beam_remap_inputs", lambda *args: [3])
    wrapped = npu_patch.make_patched_prepare_inputs(original)
    assert npu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary[1:] == (spec_metadata, 7)

    runner.input_batch = proxy
    runner._current_beam_info["req_ids"] = []
    logits, returned_spec, total = wrapped(
        runner,
        SimpleNamespace(beam_data={"request": {}}),
        [1],
    )
    assert logits.dtype is torch.int32
    assert logits.tolist() == [3]
    assert returned_spec is spec_metadata
    assert total == 7


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

    def original(
        self: Any,
        scheduler_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        if getattr(scheduler_output, "fail", False):
            raise RuntimeError("boom")
        return ordinary_result

    def collapse(
        self: Any,
        configs: Any,
        beam_data: Any,
        sampler: Any,
    ) -> tuple[list[str], dict[str, int], int]:
        assert sampler is sampler_output
        return ["beam"], {"beam": 0}, 1

    monkeypatch.setattr(npu_patch, "_beam_collapse_bookkeeping", collapse)
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
    from vllm_gr.v1.worker.model_runner_contracts import (
        _WORKER_SAMPLE_PATCH_MARKER,
    )

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
    from vllm_gr.v1.beam.batch_contracts import BeamSequenceColumns
    from vllm_gr.v1.beam.output_columns import (
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


pytestmark = pytest.mark.cpu_test
