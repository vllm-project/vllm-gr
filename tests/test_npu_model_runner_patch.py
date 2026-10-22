# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for NPU ModelRunner wrappers."""

import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import NewRequestData

from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.v1.beam.execution_batch import BeamExecutionContractError
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
    NPUEagerFirstDecodeHandoff.from_prefill(plan, winners).bind_first_decode(
        decode_plan, input_ids, positions
    )
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
    spec_metadata, compressed = object(), object()

    def native(self: Any, output: Any, native_counts: Any) -> tuple[Any, ...]:
        assert self is worker and output is scheduled and native_counts is counts
        assert tuple(r.num_computed_tokens for r in output.scheduled_new_reqs) == (128, 0, 512, 256)
        return object(), spec_metadata, 900, compressed

    logits, spec, total, returned_compressed = stage.prepare_inputs(native, scheduled, counts)
    assert hidden[logits.long()].tolist() == [128, 513, 514, 899]
    assert spec is spec_metadata and returned_compressed is compressed and total == 900
    assert counts.tolist() == [129, 385, 1, 385]


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

    session = SimpleNamespace(is_active=True, slot=0, batch_size=1, request_width=4)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=4))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
        _gr_npu_beam_graph_shapes={(1, 4)},
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
    )
    request_ids = [f"beam-{index}" for index in range(4)]
    beam_info = _make_beam_info(session, request_ids, request_ids)
    native = (getattr(CUDAGraphMode, native_mode), BatchDescriptor(num_tokens=8), "native-extra")
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={8},
        _gr_npu_beam_graph_shapes={(2, 4)},
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


def _stub_beam_graph_install(
    monkeypatch: pytest.MonkeyPatch, ready_widths: tuple[int, ...] = ()
) -> list[Any]:
    import sys
    from types import ModuleType

    calls: list[Any] = []
    module = ModuleType("vllm_gr.v1.worker.npu_beam_decode_graph")

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


@pytest.mark.parametrize("first_path", ["ordinary", "beam"])  # type: ignore[misc]
def test_npu_beam_and_ordinary_graphs_isolate_matching_descriptors(
    monkeypatch: pytest.MonkeyPatch, first_path: str
) -> None:
    acl_graph = pytest.importorskip(
        "vllm_ascend.compilation.acl_graph", reason="vllm-ascend is not installed"
    )
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor

    from vllm_gr.v1.beam.context import BEAM_DATA_VAR
    from vllm_gr.v1.worker.npu_beam_decode_graph import NPUBeamGraphModel

    descriptor = BatchDescriptor(num_tokens=4, uniform=True)

    def native_forward(wrapper: Any) -> Any:
        entries = wrapper.concrete_aclgraph_entries
        if descriptor not in entries:
            entries[descriptor] = SimpleNamespace(aclgraph=object())
        return entries[descriptor].aclgraph

    monkeypatch.setattr(acl_graph.ACLGraphWrapper, "__call__", native_forward)
    ordinary = object.__new__(acl_graph.ACLGraphWrapper)
    ordinary.concrete_aclgraph_entries = {}
    beam = object.__new__(NPUBeamGraphModel)
    beam.original_model = ordinary
    beam.concrete_aclgraph_entries = {}
    runner = SimpleNamespace(model=beam, _gr_npu_graph_model=beam)

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


@pytest.mark.parametrize(  # type: ignore[misc]
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.npu_integration)]
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
    stage = _terminal_result_runner(1, include_parents=False)
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


@pytest.mark.parametrize("ignore_stop_tokens", [False, True])  # type: ignore[misc]
def test_npu_dynamic_beam_admission_requires_fixed_horizon(ignore_stop_tokens: bool) -> None:
    from vllm_gr.v1.beam.batch_contracts import BeamBatchParams
    from vllm_gr.v1.engine.beam_search_v1 import _validate_gr_generation_params

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
