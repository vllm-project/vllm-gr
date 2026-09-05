# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the persistent NPU V1 input boundary.

These tests use CPU tensors to validate ownership, geometry, and token-source
semantics.  They are not evidence for NPU attention metadata, graph replay,
data residency, or asynchronous device execution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_gr.v1.engine.types import GRSchedulerOutput, GRRequestStage
from vllm_gr.v1.worker import npu_gr_inputs
from vllm_gr.v1.worker import model_runner_common
from vllm_gr.v1.worker.npu_gr_decision import bootstrap_from_constrained_prefill


def _metadata(*, step: int = 1, width: int = 4, final: bool = False) -> dict[str, Any]:
    return {
        "persistent_runner_state": True,
        "persistent_request": True,
        "is_beam_decode": True,
        "is_constrained": True,
        "beam_width": width,
        "generation_index": step,
        "constraint_step": step,
        "beam_decode_steps": step,
        "decode_step": step - 1,
        "is_last_step": final,
        "next_token_ids": None,
        "beam_suffix": [],
        "parent_beam_ids": None,
    }


def _output(*, step: int = 1, width: int = 4, final: bool = False) -> Any:
    sidecar = GRSchedulerOutput(
        request_id="gr",
        stage=GRRequestStage.DECODE_RUNNING,
        step=step,
        device_step=step,
        is_prefill=False,
        is_final=final,
        logical_token_count=1,
        physical_row_count=width,
    )
    return SimpleNamespace(
        beam_data={"gr": _metadata(step=step, width=width, final=final)},
        gr_outputs={"gr": sidecar},
        num_scheduled_tokens={"gr": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        has_structured_output_requests=False,
        scheduled_encoder_inputs={},
    )


def _runner(*, step: int = 1, width: int = 4) -> Any:
    tokens = torch.arange(40, 40 + width, dtype=torch.int32).view(1, width)
    state = SimpleNamespace(
        request_id="gr",
        released=False,
        _prepared_step=step,
        step=step,
        device_step=step,
        next_token_ids=tokens,
    )
    return SimpleNamespace(
        use_async_scheduling=True,
        pcp_size=1,
        dcp_size=1,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        use_cp=False,
        speculative_config=None,
        enable_prompt_embeds=False,
        input_batch=SimpleNamespace(req_ids=["gr"], req_prompt_embeds={}),
        _gr_model_runner_states={"gr": state},
        input_ids=SimpleNamespace(gpu=torch.full((width + 2,), -1, dtype=torch.int32)),
        device=torch.device("cpu"),
    )


@pytest.fixture(autouse=True)
def _disable_transfer_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(npu_gr_inputs, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(npu_gr_inputs, "has_ec_transfer", lambda: False)


def _wrap_execute_without_common_context(monkeypatch: pytest.MonkeyPatch, original: Any) -> Any:
    # This module's contract is the NPU geometry frame.  Shared persistent
    # state/session lifecycle has separate tests and would obscure this unit.
    monkeypatch.setattr(
        npu_gr_inputs,
        "_make_common_patched_execute_model",
        lambda target: target,
    )
    return npu_gr_inputs.make_patched_execute_model(original)


def test_decode_frame_exposes_physical_width_once_and_restores_logical_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _output()
    runner = _runner()
    original_prepare_calls = 0
    execute_calls = 0
    observed_frames = []
    result = object()

    def original_prepare(*_args: Any) -> None:
        nonlocal original_prepare_calls
        original_prepare_calls += 1
        runner.input_ids.gpu[:4].fill_(999)  # Host control-token stand-in.

    prepare = npu_gr_inputs.make_patched_prepare_input_ids(original_prepare)

    def original_execute(self: Any, scheduled: Any) -> object:
        nonlocal execute_calls
        execute_calls += 1
        frame = self._gr_npu_input_frame
        observed_frames.append(frame)
        assert frame.scheduler_output is scheduled is output
        assert frame.sidecar is scheduled.gr_outputs["gr"]
        assert frame.metadata is scheduled.beam_data["gr"]
        assert frame.active
        assert scheduled.num_scheduled_tokens == {"gr": 4}
        assert scheduled.total_num_scheduled_tokens == 4
        assert scheduled._gr_npu_logical_geometry == ("gr", 1, 4)
        assert prepare(self, scheduled, 1, 4, np.array([4], dtype=np.int32)) is None
        return result

    execute = _wrap_execute_without_common_context(monkeypatch, original_execute)
    assert execute(runner, output) is result
    assert execute_calls == 1
    assert original_prepare_calls == 0
    assert runner.input_ids.gpu.tolist() == [40, 41, 42, 43, -1, -1]
    assert output.num_scheduled_tokens == {"gr": 1}
    assert output.total_num_scheduled_tokens == 1
    assert not hasattr(runner, "_gr_npu_input_frame")
    assert not hasattr(output, "_gr_npu_logical_geometry")
    assert len(observed_frames) == 1
    assert not observed_frames[0].active


def test_b1_cumulative_counts_match_installed_upstream_helper() -> None:
    runner = SimpleNamespace(arange_np=np.arange(16, dtype=np.int32))
    per_request = np.array([4], dtype=np.int32)
    arange_out = np.empty((4,), dtype=np.int32)

    cumulative = GPUModelRunner._get_cumsum_and_arange(
        runner,
        per_request,
        arange_out,
        cumsum_dtype=np.int32,
    )

    assert cumulative.shape == (1,)
    assert cumulative.dtype == np.int32
    assert cumulative.tolist() == [4]
    assert arange_out.tolist() == [0, 1, 2, 3]


def test_decode_frame_restores_logical_geometry_and_preserves_exception_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _output()
    runner = _runner()
    error = LookupError("execute failed")

    def original(self: Any, scheduled: Any) -> None:
        assert scheduled.num_scheduled_tokens == {"gr": 4}
        assert self._gr_npu_input_frame.active
        raise error

    execute = _wrap_execute_without_common_context(monkeypatch, original)
    with pytest.raises(LookupError) as caught:
        execute(runner, output)
    assert caught.value is error
    assert output.num_scheduled_tokens == {"gr": 1}
    assert output.total_num_scheduled_tokens == 1
    assert not hasattr(runner, "_gr_npu_input_frame")
    assert not hasattr(output, "_gr_npu_logical_geometry")


def test_ordinary_and_prefill_execute_delegate_exact_result_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = object()
    error = RuntimeError("ordinary failed")
    calls: list[Any] = []

    def original(_self: Any, output: Any, *, fail: bool = False) -> object:
        calls.append(output)
        if fail:
            raise error
        return result

    execute = _wrap_execute_without_common_context(monkeypatch, original)
    runner = _runner()
    ordinary = SimpleNamespace(beam_data={})
    prefill = SimpleNamespace(
        beam_data={"gr": {"persistent_runner_state": True, "is_beam_decode": False}}
    )
    assert execute(runner, ordinary) is result
    assert execute(runner, prefill) is result
    with pytest.raises(RuntimeError) as caught:
        execute(runner, ordinary, fail=True)
    assert caught.value is error
    assert calls == [ordinary, prefill, ordinary]


def test_real_common_wrapper_preserves_async_sample_context_without_inlining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _output()
    runner = _runner()
    session = object()
    sample_calls = 0
    launch_calls = 0

    runner._vllm_gr_config = SimpleNamespace(
        beam_engine_driven=True,
        beam_worker_decision=True,
    )

    def sample_tokens(_grammar: Any) -> None:
        nonlocal sample_calls
        sample_calls += 1

    def launch(_runner: Any) -> None:
        nonlocal launch_calls
        launch_calls += 1

    runner.sample_tokens = sample_tokens
    monkeypatch.setattr(
        model_runner_common,
        "_get_or_create_beam_session",
        lambda *_args: session,
    )
    monkeypatch.setattr(
        "vllm_gr.v1.worker.constraint_hook.launch_prepared_cuda_prefill_constraint",
        launch,
    )

    execute = npu_gr_inputs.make_patched_execute_model(lambda *_args: None)
    assert execute(runner, output) is None
    assert sample_calls == 0
    assert launch_calls == 1
    assert runner._current_beam_data is output.beam_data
    assert runner._current_beam_info["beam_data"] is output.beam_data
    assert runner._current_beam_info["beam_session"] is session
    assert runner._current_scheduler_output is output
    assert output.num_scheduled_tokens == {"gr": 1}
    assert output.total_num_scheduled_tokens == 1
    assert not hasattr(runner, "_gr_npu_input_frame")


def test_real_common_wrapper_releases_same_request_and_preserves_worker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import gr_model_runner_state

    output = _output()
    runner = _runner()
    error = ArithmeticError("worker failed")
    released: list[tuple[Any, tuple[str, ...]]] = []

    monkeypatch.setattr(
        model_runner_common,
        "_get_or_create_beam_session",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        gr_model_runner_state,
        "release_gr_model_runner_states",
        lambda target, request_ids: released.append((target, tuple(request_ids))),
    )

    def original(*_args: Any) -> None:
        raise error

    execute = npu_gr_inputs.make_patched_execute_model(original)
    with pytest.raises(ArithmeticError) as caught:
        execute(runner, output)
    assert caught.value is error
    assert released == [(runner, ("gr",))]
    assert output.num_scheduled_tokens == {"gr": 1}
    assert output.total_num_scheduled_tokens == 1
    assert not hasattr(runner, "_gr_npu_input_frame")
    assert not hasattr(output, "_gr_npu_logical_geometry")


def test_ordinary_and_prefill_input_copy_delegate_exact_result_and_exception() -> None:
    result = object()
    error = RuntimeError("copy failed")
    should_fail = False

    def original(_self: Any, *_args: Any) -> object:
        if should_fail:
            raise error
        return result

    prepare = npu_gr_inputs.make_patched_prepare_input_ids(original)
    runner = _runner()
    ordinary = SimpleNamespace(beam_data={})
    prefill = SimpleNamespace(
        beam_data={"gr": {"persistent_runner_state": True, "is_beam_decode": False}}
    )
    assert prepare(runner, ordinary, 1, 1, np.array([1])) is result
    assert prepare(runner, prefill, 1, 1, np.array([1])) is result
    should_fail = True
    with pytest.raises(RuntimeError) as caught:
        prepare(runner, ordinary, 1, 1, np.array([1]))
    assert caught.value is error


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda output: output.beam_data.update({"other": {}}), "isolated B=1"),
        (lambda output: output.num_scheduled_tokens.update({"other": 1}), "only scheduled"),
        (lambda output: setattr(output, "gr_outputs", {"gr": object()}), "typed"),
        (lambda output: setattr(output.gr_outputs["gr"], "request_id", "other"), "request_id"),
        (
            lambda output: setattr(output.gr_outputs["gr"], "stage", GRRequestStage.FINAL_READY),
            "Decode",
        ),
        (lambda output: setattr(output.gr_outputs["gr"], "step", 2), "step/device_step"),
        (
            lambda output: setattr(output.gr_outputs["gr"], "logical_token_count", 2),
            "logical token",
        ),
        (
            lambda output: setattr(output.gr_outputs["gr"], "logical_token_count", True),
            "logical token count",
        ),
        (
            lambda output: output.num_scheduled_tokens.update(gr=True),
            "scheduled logical token count",
        ),
        (
            lambda output: setattr(output, "total_num_scheduled_tokens", True),
            "total logical token count",
        ),
        (lambda output: setattr(output.gr_outputs["gr"], "physical_row_count", 3), "Beam width"),
        (lambda output: output.beam_data["gr"].update(generation_index=2), "generation_index"),
        (lambda output: output.beam_data["gr"].update(decode_step=1), "decode_step"),
        (lambda output: output.beam_data["gr"].update(is_last_step=True), "final marker"),
        (
            lambda output: output.beam_data["gr"].update(next_token_ids=[1, 2, 3, 4]),
            "Host metadata",
        ),
        (lambda output: output.beam_data["gr"].update(beam_suffix=[1]), "Host Beam state"),
    ],
)
def test_decode_frame_rejects_untyped_or_inconsistent_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    match: str,
) -> None:
    output = _output()
    mutation(output)
    execute = _wrap_execute_without_common_context(monkeypatch, lambda *_args: None)
    with pytest.raises(RuntimeError, match=match):
        execute(_runner(), output)


@pytest.mark.parametrize(
    "stale_on",
    ["runner", "runner-ordinary", "runner-prefill", "scheduler"],
)
def test_nested_or_stale_input_frames_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    stale_on: str,
) -> None:
    output = _output()
    runner = _runner()
    if stale_on.startswith("runner"):
        runner._gr_npu_input_frame = object()
        match = "nested or stale"
        if stale_on == "runner-ordinary":
            output = SimpleNamespace(beam_data={})
        elif stale_on == "runner-prefill":
            output = SimpleNamespace(
                beam_data={
                    "gr": {
                        "persistent_runner_state": True,
                        "is_beam_decode": False,
                    }
                }
            )
    else:
        output._gr_npu_logical_geometry = ("gr", 1, 4)
        match = "stale persistent"
    execute = _wrap_execute_without_common_context(monkeypatch, lambda *_args: None)
    with pytest.raises(RuntimeError, match=match):
        execute(runner, output)


def _installed_frame(output: Any, runner: Any) -> Any:
    frame = npu_gr_inputs._persistent_decode_frame(output)
    assert frame is not None
    output.num_scheduled_tokens["gr"] = frame.physical_rows
    output.total_num_scheduled_tokens = frame.physical_rows
    output._gr_npu_logical_geometry = ("gr", 1, frame.physical_rows)
    runner._gr_npu_input_frame = frame
    return frame


def test_persistent_decode_without_frame_never_falls_back_to_host_copy() -> None:
    output = _output()
    called = False

    def forbidden(*_args: Any) -> None:
        nonlocal called
        called = True

    prepare = npu_gr_inputs.make_patched_prepare_input_ids(forbidden)
    with pytest.raises(RuntimeError, match="without an active frame"):
        prepare(_runner(), output, 1, 1, np.array([1], dtype=np.int32))
    assert not called


def test_persistent_frame_allows_exactly_one_input_copy() -> None:
    output = _output()
    runner = _runner()
    frame = _installed_frame(output, runner)
    prepare = npu_gr_inputs.make_patched_prepare_input_ids(
        lambda *_args: pytest.fail("persistent input must not use Host copy")
    )

    assert prepare(runner, output, 1, 4, np.array([4], dtype=np.int32)) is None
    assert frame.input_copy_done
    with pytest.raises(RuntimeError, match="already copied"):
        prepare(runner, output, 1, 4, np.array([4], dtype=np.int32))


def test_real_bootstrap_strided_token_view_feeds_persistent_input_copy() -> None:
    width = 4
    sequence = torch.zeros((1, width, 3), dtype=torch.int32)
    decision = SimpleNamespace(
        sequence=sequence,
        beam_scores=torch.zeros((1, width), dtype=torch.float32),
        group_token_num=torch.zeros((1, width), dtype=torch.int32),
        worker_decode_steps=0,
    )
    session = SimpleNamespace(
        beam_width=width,
        is_active=True,
        slot=0,
        decode_step=0,
        context=SimpleNamespace(decision=decision),
    )
    fake_ops = SimpleNamespace(
        is_available=lambda: True,
        beam_search_group=lambda *_args: None,
        beam_search_rec_final_select=lambda *_args: None,
    )
    bootstrap = bootstrap_from_constrained_prefill(
        session,
        torch.tensor([[71, 72, 73, 74]], dtype=torch.int32),
        torch.tensor([[-1.0, -2.0, -3.0, -4.0]], dtype=torch.float32),
        expected_width=width,
        ops=fake_ops,
    )
    assert bootstrap.token_ids is not sequence
    assert bootstrap.token_ids.data_ptr() == sequence.data_ptr()
    assert not bootstrap.token_ids.is_contiguous()

    output = _output(width=width)
    runner = _runner(width=width)
    runner._gr_model_runner_states["gr"].next_token_ids = bootstrap.token_ids
    _installed_frame(output, runner)
    prepare = npu_gr_inputs.make_patched_prepare_input_ids(
        lambda *_args: pytest.fail("persistent input must not use Host copy")
    )
    assert prepare(runner, output, 1, width, np.array([width], dtype=np.int32)) is None
    assert runner.input_ids.gpu[:width].tolist() == [71, 72, 73, 74]


@pytest.mark.parametrize(
    ("mutate", "args", "match"),
    [
        (lambda _r, _o: None, (2, 4, np.array([4])), "B=1"),
        (lambda _r, _o: None, (True, 4, np.array([4])), "B=1"),
        (lambda _r, _o: None, (1, 3, np.array([4])), "physical input count"),
        (lambda _r, _o: None, (1, True, np.array([4])), "physical input count"),
        (lambda _r, _o: None, (1, 4, [4]), "NumPy array"),
        (lambda _r, _o: None, (1, 4, np.array([0, 4])), r"integer \[1\]"),
        (lambda _r, _o: None, (1, 4, np.array([3])), r"equal \[W\]"),
        (lambda _r, _o: None, (1, 4, np.array([4], dtype=np.float32)), "integer"),
        (
            lambda r, _o: setattr(r.input_batch, "req_ids", ["other"]),
            (1, 4, np.array([4])),
            "identity",
        ),
        (
            lambda r, _o: setattr(r._gr_model_runner_states["gr"], "released", True),
            (1, 4, np.array([4])),
            "released",
        ),
        (
            lambda r, _o: setattr(r._gr_model_runner_states["gr"], "_prepared_step", 0),
            (1, 4, np.array([4])),
            "prepared/device step",
        ),
        (
            lambda r, _o: setattr(
                r._gr_model_runner_states["gr"],
                "next_token_ids",
                torch.zeros((4,), dtype=torch.int32),
            ),
            (1, 4, np.array([4])),
            "shape",
        ),
        (
            lambda r, _o: setattr(
                r._gr_model_runner_states["gr"],
                "next_token_ids",
                torch.zeros((1, 4), dtype=torch.int64),
            ),
            (1, 4, np.array([4])),
            "dtype/device",
        ),
        (
            lambda r, _o: setattr(
                r._gr_model_runner_states["gr"],
                "next_token_ids",
                torch.zeros((1, 4), dtype=torch.int32, device="meta"),
            ),
            (1, 4, np.array([4])),
            "dtype/device",
        ),
        (
            lambda r, _o: setattr(r.input_ids, "gpu", torch.zeros((2,), dtype=torch.int32)),
            (1, 4, np.array([4])),
            "smaller",
        ),
        (
            lambda r, _o: setattr(r.input_ids, "gpu", torch.zeros((6,), dtype=torch.int64)),
            (1, 4, np.array([4])),
            "int32",
        ),
        (
            lambda r, _o: setattr(r, "device", torch.device("meta")),
            (1, 4, np.array([4])),
            "wrong device",
        ),
    ],
)
def test_persistent_input_copy_fails_fast_before_calling_native_host_copy(
    mutate: Any,
    args: tuple[Any, ...],
    match: str,
) -> None:
    output = _output()
    runner = _runner()
    _installed_frame(output, runner)
    mutate(runner, output)
    called = False

    def forbidden(*_args: Any) -> None:
        nonlocal called
        called = True

    prepare = npu_gr_inputs.make_patched_prepare_input_ids(forbidden)
    with pytest.raises(RuntimeError, match=match):
        prepare(runner, output, *args)
    assert not called


@pytest.mark.parametrize(
    ("mutate", "transfer", "match"),
    [
        (lambda r, _o: setattr(r, "use_async_scheduling", False), None, "async"),
        (lambda r, _o: setattr(r, "pcp_size", 2), None, "PCP"),
        (lambda r, _o: setattr(r, "dcp_size", 2), None, "DCP"),
        (lambda r, _o: setattr(r.parallel_config, "pipeline_parallel_size", 2), None, "pipeline"),
        (lambda r, _o: setattr(r, "use_cp", True), None, "CP"),
        (lambda r, _o: setattr(r, "speculative_config", object()), None, "speculative"),
        (
            lambda _r, o: setattr(o, "scheduled_spec_decode_tokens", {"gr": [1]}),
            None,
            "draft-token",
        ),
        (lambda _r, o: setattr(o, "has_structured_output_requests", True), None, "grammar"),
        (lambda _r, o: setattr(o, "scheduled_encoder_inputs", {"gr": [1]}), None, "encoder"),
        (lambda r, _o: setattr(r, "enable_prompt_embeds", True), None, "prompt embeddings"),
        (lambda _r, _o: None, "kv", "KV connector"),
        (lambda _r, _o: None, "ec", "EC transfer"),
    ],
)
def test_execute_rejects_unsupported_parallel_or_input_rebuild_paths(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    transfer: str | None,
    match: str,
) -> None:
    output = _output()
    runner = _runner()
    mutate(runner, output)
    if transfer == "kv":
        monkeypatch.setattr(npu_gr_inputs, "has_kv_transfer_group", lambda: True)
    if transfer == "ec":
        monkeypatch.setattr(npu_gr_inputs, "has_ec_transfer", lambda: True)
    execute = _wrap_execute_without_common_context(monkeypatch, lambda *_args: None)
    with pytest.raises(RuntimeError, match=match):
        execute(runner, output)
