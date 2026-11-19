# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Startup graph publication is conditional on capture, release and GPU health."""

from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.platforms.cuda.stage import create_v1_stage, create_v2_stage
from vllm_gr.v1.worker import gr_startup as startup
from vllm_gr.v1.worker.cuda import gpu_beam_stage_runner as stages
from vllm_gr.v1.worker.cuda.beam_decode_graph import BeamDecodeCaptureUnsupported
from vllm_gr.v1.worker.cuda.gpu_stage_hooks import CudaStageHooks
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1
from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2
from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_resources import (
    capture_multibatch_forward,
    get_v2_runtime,
)


def _capture_failed(worker: SimpleNamespace) -> bool:
    runtime = getattr(worker, "_gr_v2_runtime", None)
    return bool(
        runtime.capture_failed
        if runtime is not None
        else getattr(worker, "_gr_async_decode_graph_capture_failed", False)
    )


def _capture_error(worker: SimpleNamespace) -> str | None:
    runtime = getattr(worker, "_gr_v2_runtime", None)
    return (
        runtime.capture_error
        if runtime is not None
        else getattr(worker, "_gr_async_decode_graph_capture_error", None)
    )


CaptureRuntime = tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, Mock, Mock]


@pytest.fixture  # type: ignore[misc]
def capture_runtime(monkeypatch: pytest.MonkeyPatch) -> CaptureRuntime:
    monkeypatch.delenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, raising=False)
    worker = SimpleNamespace(
        device="cuda",
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=True, max_num_seqs=1)
        ),
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(
            beam_max_width=4,
            beam_max_decode_steps=3,
            beam_worker_decision=True,
            beam_graph_enabled=True,
            beam_online_batch_sizes=(1, 2, 4),
            constraint_table=SimpleNamespace(enabled=True),
        ),
        _gr_async_decode_graph_capture_failed=False,
        _gr_async_decode_graph_capture_error=None,
    )
    worker._gr_unplanned_stage = create_v1_stage(plan=None, runner=worker)
    graph = SimpleNamespace(ready=True, clear=Mock())
    owner = SimpleNamespace(
        graph=None,
        closed=False,
        binding=object(),
        prepare_cpu=Mock(),
        _bind_attention=Mock(),
        input_ids=Mock(),
        positions=Mock(),
        execution_mask=Mock(),
        beam=SimpleNamespace(
            session=SimpleNamespace(context=SimpleNamespace(decode_step_buf=Mock()))
        ),
        capture_forward=Mock(return_value=graph),
    )

    def release(*args: Any, **kwargs: Any) -> None:
        assert owner.graph is None, "graph published before release"
        assert not _capture_failed(worker)
        owner.closed = True

    owner.release = Mock(side_effect=release)
    stream = Mock()

    def synchronize() -> None:
        assert owner.closed
        assert owner.graph is None, "graph published before synchronization"
        assert not _capture_failed(worker)

    stream.synchronize.side_effect = synchronize
    monkeypatch.setattr(stages, "get_stage_runner", Mock(return_value=owner))
    monkeypatch.setattr(torch, "Event", Mock(return_value=Mock()))
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=stream))
    health = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", health)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return worker, owner, graph, stream, health


def _capture(worker: SimpleNamespace, *, allow: bool = True) -> bool:
    return bool(capture_multibatch_forward(worker, allow_eager_fallback=allow))


@pytest.mark.l1  # type: ignore[misc]
def test_startup_requires_a_bound_stage_or_installed_factory(
    capture_runtime: CaptureRuntime,
) -> None:
    with pytest.raises(ValueError, match="requires explicit stage hooks"):
        startup.capture_async_forward(capture_runtime[0])
    cast(Mock, stages.get_stage_runner).assert_not_called()


@pytest.mark.parametrize("hooks_type", [CudaStageHooksV1, CudaStageHooksV2])  # type: ignore[misc]
def test_success_publishes_only_after_release_and_sync(
    capture_runtime: CaptureRuntime,
    hooks_type: type[CudaStageHooks],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, raising=False)
    worker, owner, graph, stream, health = capture_runtime
    factory = create_v2_stage if hooks_type is CudaStageHooksV2 else create_v1_stage
    worker._gr_unplanned_stage = factory(plan=None, runner=worker)
    assert startup.capture_async_forward(worker)
    assert cast(Mock, stages.get_stage_runner).call_args.kwargs == {
        "hooks_type": hooks_type,
        "for_startup_capture": True,
    }
    decode = owner.prepare_cpu.call_args.args[0]
    assert decode.kv_blocks == ((0,),)
    assert decode.beam_params.max_tokens == 2
    assert decode.is_last_stage
    assert decode.input_device_state == startup.GRDeviceStateRef(decode.session_id, 0)
    assert decode.output_device_state == startup.GRDeviceStateRef(decode.session_id, 1)
    owner._bind_attention.assert_called_once_with(decode)
    assert not hasattr(worker, "requests")
    assert owner.graph is graph
    owner.release.assert_called_once()
    stream.synchronize.assert_called_once()
    health.assert_not_called()
    graph.clear.assert_not_called()


@pytest.mark.parametrize("batch_size", range(1, 9))  # type: ignore[misc]
def test_opt_in_startup_capture_uses_requested_rows(
    capture_runtime: CaptureRuntime, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    monkeypatch.setenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, str(batch_size))
    worker, owner, _graph, _stream, _health = capture_runtime
    assert startup.capture_async_forward(worker)
    scheduled = cast(Mock, stages.get_stage_runner).call_args.args[1]
    prefill = scheduled.gr_stage_metadata["__async_scheduling_startup_capture__"]
    decode = owner.prepare_cpu.call_args.args[0]
    assert not prefill.is_last_stage
    assert len(prefill.request_ids) == batch_size
    assert prefill.num_prompt_tokens == batch_size
    assert prefill.num_prompt_tokens_by_request == (1,) * batch_size
    assert scheduled.num_scheduled_tokens == dict.fromkeys(prefill.request_ids, 1)
    assert decode.kv_blocks == ((0,),) * batch_size
    owner.release.assert_called_once()


def test_v1_startup_batch_override_does_not_change_v2(
    capture_runtime: CaptureRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, "4")
    worker, owner, _graph, _stream, _health = capture_runtime
    assert capture_multibatch_forward(worker)
    scheduled = cast(Mock, stages.get_stage_runner).call_args.args[1]
    prefill = scheduled.gr_stage_metadata["__async_scheduling_startup_capture__"]
    decode = owner.prepare_cpu.call_args.args[0]
    assert len(prefill.request_ids) == 1
    assert prefill.num_prompt_tokens == 1
    assert prefill.num_prompt_tokens_by_request == (1,)
    assert scheduled.num_scheduled_tokens == dict.fromkeys(prefill.request_ids, 1)
    assert decode.kv_blocks == ((0,),)
    owner.release.assert_called_once()


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
def test_prefix_calibration_precedes_decode_capture(
    capture_runtime: CaptureRuntime, monkeypatch: pytest.MonkeyPatch, v2: bool
) -> None:
    from vllm_gr.v1.attention.backends import beam_prefix_calibrate as calibrate

    events: list[str] = []
    initialize = Mock(side_effect=lambda worker: events.append("calibration"))
    capture = Mock(side_effect=lambda *args, **kwargs: events.append("capture"))
    monkeypatch.setattr(calibrate, "ensure_prefix_calibration", initialize)
    monkeypatch.setattr(startup, "_capture_batch", capture)
    worker = capture_runtime[0]
    entry = (
        capture_multibatch_forward
        if v2
        else partial(startup.capture_async_forward, hooks_type=CudaStageHooksV1)
    )

    assert entry(worker)
    initialize.assert_called_once_with(worker)
    assert events == ["calibration", "capture"]


@pytest.mark.parametrize("raw", ["0", "9", "-1", "1.5", "invalid", ""])  # type: ignore[misc]
def test_invalid_startup_capture_batch_size(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, raw)
    with pytest.raises(ValueError) as error:
        startup._startup_capture_batch_size()
    assert str(error.value) == (
        f"{startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE} must be one of [1, 2, 3, 4, 5, 6, 7, 8]"
    )


def test_default_startup_capture_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, raising=False)
    assert startup._startup_capture_batch_size() == 1


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
def test_single_step_skips_decode_capture(capture_runtime: CaptureRuntime, v2: bool) -> None:
    worker, owner, _graph, _stream, _health = capture_runtime
    worker._vllm_gr_config.beam_max_decode_steps = 1
    entry = (
        capture_multibatch_forward
        if v2
        else partial(startup.capture_async_forward, hooks_type=CudaStageHooksV1)
    )

    assert not entry(worker)
    assert not entry(worker)
    cast(Mock, stages.get_stage_runner).assert_not_called()
    owner.capture_forward.assert_not_called()
    owner.release.assert_not_called()
    assert not hasattr(worker, "_gr_gpu_stage_runner")
    assert getattr(getattr(worker, "_gr_v2_runtime", None), "startup_failures", None) is None


def test_recoverable_failure_records_reason_only_after_cleanup(
    capture_runtime: CaptureRuntime,
) -> None:
    worker, owner, _graph, stream, health = capture_runtime
    owner.capture_forward.side_effect = BeamDecodeCaptureUnsupported("capture unsupported")
    assert not _capture(worker)
    assert _capture_failed(worker)
    assert _capture_error(worker) == "capture unsupported"
    assert owner.graph is None and owner.closed
    owner.release.assert_called_once()
    health.assert_called_once_with(worker.device)
    stream.synchronize.assert_called_once()
    assert not stages.should_replay_decode_graph(
        worker, owner.graph, capture_failed=get_v2_runtime(worker).capture_failed
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "error",
    [
        ValueError("invalid input"),
        RuntimeError("operation not supported when stream is capturing"),
    ],
)
def test_unclassified_failure_cannot_enable_fallback(
    capture_runtime: CaptureRuntime, error: Exception
) -> None:
    worker, owner, _graph, _stream, health = capture_runtime
    owner.capture_forward.side_effect = error
    with pytest.raises(type(error)) as raised:
        _capture(worker)
    assert raised.value is error
    assert not _capture_failed(worker)
    assert owner.graph is None
    owner.release.assert_called_once()
    health.assert_not_called()


@pytest.mark.parametrize("recoverable", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("phase", ["release", "sync"])  # type: ignore[misc]
def test_cleanup_failure_never_publishes_graph_or_fallback(
    capture_runtime: CaptureRuntime, recoverable: bool, phase: str
) -> None:
    worker, owner, graph, stream, _health = capture_runtime
    if recoverable:
        owner.capture_forward.side_effect = BeamDecodeCaptureUnsupported("capture unsupported")
    error = RuntimeError("operation not supported when stream is capturing")
    (owner.release if phase == "release" else stream.synchronize).side_effect = error
    with pytest.raises(RuntimeError) as raised:
        _capture(worker)
    assert raised.value is error
    assert not _capture_failed(worker)
    assert _capture_error(worker) is None
    assert owner.graph is None
    assert graph.clear.call_count == int(not recoverable)


def test_failed_device_health_check_cannot_enable_fallback(capture_runtime: CaptureRuntime) -> None:
    worker, owner, _graph, _stream, health = capture_runtime
    owner.capture_forward.side_effect = BeamDecodeCaptureUnsupported("capture unsupported")
    health.side_effect = RuntimeError("CUDA illegal memory access")
    with pytest.raises(RuntimeError, match="illegal memory"):
        _capture(worker)
    assert not _capture_failed(worker)
    assert owner.graph is None
    owner.release.assert_called_once()


def test_capture_mode_must_have_ended(
    capture_runtime: CaptureRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, owner, _graph, _stream, health = capture_runtime
    owner.capture_forward.side_effect = BeamDecodeCaptureUnsupported("capture unsupported")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="did not leave capture mode"):
        _capture(worker)
    assert not _capture_failed(worker)
    health.assert_not_called()


def test_v1_does_not_opt_into_optional_fallback(capture_runtime: CaptureRuntime) -> None:
    worker, owner, _graph, _stream, _health = capture_runtime
    owner.capture_forward.side_effect = BeamDecodeCaptureUnsupported("capture unsupported")
    with pytest.raises(BeamDecodeCaptureUnsupported):
        startup.capture_async_forward(worker)
    assert not _capture_failed(worker)


def test_successful_release_must_close_owner(capture_runtime: CaptureRuntime) -> None:
    worker, owner, graph, _stream, _health = capture_runtime
    owner.release.side_effect = None
    with pytest.raises(RuntimeError, match="resources were not released"):
        _capture(worker)
    assert owner.graph is None
    graph.clear.assert_called_once()


@pytest.mark.parametrize("ready", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("use_v2", [False, True])  # type: ignore[misc]
def test_repeated_startup_accepts_only_completed_capture(
    capture_runtime: CaptureRuntime, ready: bool, use_v2: bool
) -> None:
    worker, owner, graph, _stream, _health = capture_runtime
    worker._gr_gpu_stage_runner = owner
    get_v2_runtime(worker).startup_failures = {}
    worker._beam_decode_graph = graph
    graph.select = Mock(return_value=graph)
    owner.hooks = CudaStageHooksV2(worker) if use_v2 else CudaStageHooksV1(worker)
    owner.graph = graph
    owner.closed = ready
    capture: Callable[[SimpleNamespace], bool] = _capture
    if not use_v2:
        capture = partial(startup.capture_async_forward, hooks_type=CudaStageHooksV1)
    if ready:
        assert capture(worker)
    else:
        with pytest.raises(RuntimeError, match="incomplete capture state"):
            capture(worker)
    owner.capture_forward.assert_not_called()
    cast(Mock, stages.get_stage_runner).assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("use_v2", [False, True])  # type: ignore[misc]
def test_repeated_startup_rejects_different_hooks_without_recapture(
    capture_runtime: CaptureRuntime, use_v2: bool
) -> None:
    worker, owner, graph, _stream, _health = capture_runtime
    worker._gr_gpu_stage_runner = owner
    get_v2_runtime(worker).startup_failures = {}
    worker._beam_decode_graph = graph
    graph.select = Mock(return_value=graph)
    hooks = CudaStageHooksV1(worker) if use_v2 else CudaStageHooksV2(worker)
    owner.hooks = hooks
    owner.graph = graph
    owner.closed = True
    capture: Callable[[SimpleNamespace], bool] = _capture
    if not use_v2:
        capture = partial(startup.capture_async_forward, hooks_type=CudaStageHooksV1)
    with pytest.raises(ValueError, match="hooks cannot change"):
        capture(worker)
    assert owner.hooks is hooks
    assert owner.graph is graph
    assert owner.closed
    assert get_v2_runtime(worker).startup_failures == {}
    graph.select.assert_not_called()
    owner.capture_forward.assert_not_called()
    cast(Mock, stages.get_stage_runner).assert_not_called()


def test_decode_graph_capture_failure_falls_back_only_when_recorded() -> None:
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True, beam_online_batch_sizes=(1, 2, 4)),
        _gr_async_decode_graph_capture_failed=True,
    )
    assert stages.should_replay_decode_graph(worker, graph=None) is False

    worker._gr_async_decode_graph_capture_failed = False
    with pytest.raises(RuntimeError, match="requires successful startup capture"):
        stages.should_replay_decode_graph(worker, graph=None)

    assert stages.should_replay_decode_graph(worker, graph=SimpleNamespace(ready=True)) is True
    with pytest.raises(RuntimeError, match="not ready"):
        stages.should_replay_decode_graph(worker, graph=SimpleNamespace(ready=False))
    worker._gr_async_decode_graph_capture_failed = True
    with pytest.raises(RuntimeError, match="failed.*published"):
        stages.should_replay_decode_graph(worker, graph=SimpleNamespace(ready=True))
    worker.model_config.enforce_eager = True
    assert stages.should_replay_decode_graph(worker, graph=None) is False


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
def test_startup_batch_plan_is_selected_by_entry_not_environment(
    capture_runtime: CaptureRuntime, monkeypatch: pytest.MonkeyPatch, v2: bool
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0" if v2 else "1")
    monkeypatch.setenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, "4")
    worker = capture_runtime[0]
    worker.vllm_config.scheduler_config.max_num_seqs = 4
    capture = Mock(return_value=None)
    monkeypatch.setattr(startup, "_capture_batch", capture)
    entry = (
        capture_multibatch_forward
        if v2
        else partial(startup.capture_async_forward, hooks_type=CudaStageHooksV1)
    )
    assert entry(worker)
    assert [call.args[1] for call in capture.call_args_list] == ([1, 2, 4] if v2 else [4])
    assert (
        getattr(getattr(worker, "_gr_v2_runtime", None), "startup_failures", None) is not None
    ) is v2


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
def test_missing_decode_graph_is_lazy_only_for_v1(v2: bool) -> None:
    graph = SimpleNamespace(ready=False, replay=Mock(return_value="hidden"))
    graph.capture = Mock(side_effect=lambda *args, **kwargs: setattr(graph, "ready", True))
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True, beam_online_batch_sizes=(1, 2, 4)),
        model=SimpleNamespace(compute_logits=Mock(return_value="logits")),
    )
    owner = SimpleNamespace(
        bindings=(object(), object()),
        beam=SimpleNamespace(state=SimpleNamespace(beam_width=4)),
        # ``prepare_cpu`` always publishes the stage buffers before
        # ``decode_logits`` runs; the graph geometry is keyed by the live
        # input width, not the configured beam width.
        inputs=SimpleNamespace(width=4),
        dynamic=None,
        graph=None,
        # Vocabulary slicing is opt-in; a stage without a request keeps the
        # monolithic LM head.
        _head_slice_request=0,
        _head_buffers=SimpleNamespace(release_full=Mock(), release_slice=Mock()),
        _forward_hidden=Mock(),
        _decode_logits=Mock(return_value="logits"),
        _forward=Mock(),
    )
    hooks = (CudaStageHooksV2 if v2 else CudaStageHooksV1)(worker)
    hooks.decode_graph = Mock(return_value=graph)
    if v2:
        with pytest.raises(RuntimeError, match="not ready"):
            hooks.decode_logits(owner)
        graph.capture.assert_not_called()
        graph.replay.assert_not_called()
    else:
        assert hooks.decode_logits(owner) == "logits"
        graph.capture.assert_called_once_with(owner._forward_hidden, persistent_install=True)
        graph.replay.assert_called_once()
        owner._decode_logits.assert_called_once_with("hidden")


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
def test_eager_decode_keeps_forward_entry_and_slices_before_logits(v2: bool) -> None:
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True, beam_online_batch_sizes=(1,)),
    )
    owner = SimpleNamespace(
        bindings=(object(),),
        inputs=SimpleNamespace(width=4),
        dynamic=None,
        graph=None,
        _head_slice_request=0,
        _head_buffers=SimpleNamespace(release_full=Mock(), release_slice=Mock()),
        _forward=Mock(return_value="logits"),
        _forward_hidden=Mock(),
        _decode_logits=Mock(return_value="logits"),
    )
    hooks = (CudaStageHooksV2 if v2 else CudaStageHooksV1)(worker)
    assert hooks.supports_transposed_head

    assert hooks.decode_logits(owner) == "logits"
    owner._forward.assert_called_once_with()
    owner._forward_hidden.assert_not_called()

    # The shared head helper stages hidden states in owner storage instead of
    # aliasing the graph's replay buffer, then skips the monolithic logits.
    owner._hidden_staging = None
    hidden = torch.zeros((2, 4))
    owner._head_slice_request = 16
    owner.head_slice_cols = Mock(return_value=16)
    assert stages._hidden_logits(owner, hidden) is None
    assert owner.hidden is not hidden
    assert torch.equal(owner.hidden, hidden)
    assert owner._active_slice_cols == 16
    owner._decode_logits.assert_not_called()

    # A later replay writes the graph output in place; the staged copy must not
    # change, and a smaller stage reuses the same high-water-mark buffer.
    staged_buffer = owner._hidden_staging
    hidden.fill_(7)
    assert torch.equal(owner.hidden, torch.zeros((2, 4)))
    smaller = torch.ones((1, 4))
    assert stages._hidden_logits(owner, smaller) is None
    assert owner._hidden_staging is staged_buffer
    assert torch.equal(owner.hidden, smaller)


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("sliced", [False, True])  # type: ignore[misc]
def test_graph_replay_consumes_hidden_through_shared_logits(v2: bool, sliced: bool) -> None:
    hidden = torch.zeros((2, 4))
    graph = SimpleNamespace(ready=True, replay=Mock(return_value=hidden))
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True, beam_online_batch_sizes=(1,)),
    )
    owner = SimpleNamespace(
        bindings=(object(),),
        inputs=SimpleNamespace(width=4),
        dynamic=None,
        graph=None,
        _head_slice_request=16 if sliced else 0,
        _head_buffers=SimpleNamespace(release_full=Mock(), release_slice=Mock()),
        _hidden_staging=None,
        head_slice_cols=Mock(return_value=16),
        _forward_hidden=Mock(),
        _decode_logits=Mock(return_value="logits"),
        _forward=Mock(return_value="eager"),
    )
    hooks = (CudaStageHooksV2 if v2 else CudaStageHooksV1)(worker)
    hooks.decode_graph = Mock(return_value=graph)

    result = hooks.decode_logits(owner)
    if sliced:
        assert result is None
        assert owner.hidden is not hidden
        assert torch.equal(owner.hidden, hidden)
        assert owner._active_slice_cols == 16
        owner._decode_logits.assert_not_called()
        # A subsequent replay may overwrite the graph's output buffer.
        hidden.fill_(7)
        assert torch.equal(owner.hidden, torch.zeros((2, 4)))
    else:
        assert result == "logits"
        owner._decode_logits.assert_called_once_with(hidden)
    graph.replay.assert_called_once_with()
    owner._forward.assert_not_called()


def test_mode_switch_releases_the_inactive_logits_buffer() -> None:
    layout = (torch.empty((4, 8)), torch.Event())
    buffers = stages._HeadLogits(layout, torch.device("cpu"))
    assert buffers.full_buffer(3) is not None
    assert buffers.slice_buffer(3, 4) is not None
    owner = SimpleNamespace(
        _head_slice_request=4,
        _head_buffers=buffers,
        _hidden_staging=None,
        hidden=None,
        _active_slice_cols=0,
        head_slice_cols=Mock(return_value=4),
        _decode_logits=Mock(return_value="logits"),
    )

    # A slicing stage drops the full-vocabulary buffer; it used ~344 MiB at
    # 1023 rows and must not survive beside the slice buffer.
    assert stages._hidden_logits(owner, torch.zeros((3, 4))) is None
    assert buffers.full is None
    assert buffers.slice is not None

    # A monolithic stage drops the slice buffer, so only one mode is retained.
    owner.head_slice_cols.return_value = 0
    assert stages._hidden_logits(owner, torch.zeros((3, 4))) == "logits"
    assert buffers.slice is None


@pytest.mark.parametrize("sizes", [(1, 2, 4), (1, 3, 7), (1, 6, 8)])  # type: ignore[misc]
def test_v2_decode_captures_only_configured_batches(
    capture_runtime: CaptureRuntime, monkeypatch: pytest.MonkeyPatch, sizes: tuple[int, ...]
) -> None:
    from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import (
        should_replay_decode_graph_v2,
    )

    worker = capture_runtime[0]
    worker.vllm_config.scheduler_config.max_num_seqs = 8
    worker._vllm_gr_config.beam_online_batch_sizes = sizes
    capture = Mock(return_value=None)
    monkeypatch.setattr(startup, "_capture_batch", capture)
    assert capture_multibatch_forward(worker)
    assert [call.args[1] for call in capture.call_args_list] == list(sizes)
    for batch in range(1, 9):
        if batch in sizes:
            with pytest.raises(RuntimeError, match="successful startup capture"):
                should_replay_decode_graph_v2(worker, None, batch_size=batch)
        else:
            assert not should_replay_decode_graph_v2(worker, None, batch_size=batch)
            with pytest.raises(RuntimeError, match="Unconfigured Decode shape"):
                should_replay_decode_graph_v2(worker, object(), batch_size=batch)


def test_startup_failure_is_local_to_batch(
    capture_runtime: CaptureRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import (
        should_replay_decode_graph_v2 as should_replay_decode_graph,
    )

    worker = capture_runtime[0]
    worker.vllm_config.scheduler_config.max_num_seqs = 4
    captured = []

    def capture(worker: SimpleNamespace, b: int, **kwargs: Any) -> str | None:
        captured.append(b)
        return "capture unsupported" if b == 2 else None

    monkeypatch.setattr(startup, "_capture_batch", capture)
    assert capture_multibatch_forward(worker)
    assert captured == [1, 2, 4]
    assert get_v2_runtime(worker).startup_failures == {2: "capture unsupported"}
    assert not _capture_failed(worker)
    assert not should_replay_decode_graph(worker, None, batch_size=2)
    assert should_replay_decode_graph(worker, SimpleNamespace(ready=True), batch_size=4)
    with pytest.raises(RuntimeError, match="not ready"):
        should_replay_decode_graph(worker, SimpleNamespace(ready=False), batch_size=4)


pytestmark = pytest.mark.cpu_test
