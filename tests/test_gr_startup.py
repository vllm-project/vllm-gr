# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup graph publication is conditional on capture, release and GPU health."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.v1.worker import gpu_beam_stage_runner as stages
from vllm_gr.v1.worker import gr_startup as startup
from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeCaptureUnsupported
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks

CaptureRuntime = tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, Mock, Mock]


@pytest.fixture  # type: ignore[misc]
def capture_runtime(monkeypatch: pytest.MonkeyPatch) -> CaptureRuntime:
    worker = SimpleNamespace(
        device="cuda",
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True)),
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(
            beam_max_width=4,
            beam_worker_decision=True,
            beam_graph_enabled=True,
            constraint_table=SimpleNamespace(enabled=True),
        ),
        _gr_async_decode_graph_capture_failed=False,
        _gr_async_decode_graph_capture_error=None,
    )
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
        assert not worker._gr_async_decode_graph_capture_failed
        owner.closed = True

    owner.release = Mock(side_effect=release)
    stream = Mock()

    def synchronize() -> None:
        assert owner.closed
        assert owner.graph is None, "graph published before synchronization"
        assert not worker._gr_async_decode_graph_capture_failed

    stream.synchronize.side_effect = synchronize
    monkeypatch.setattr(stages, "get_stage_runner", Mock(return_value=owner))
    monkeypatch.setattr(torch, "Event", Mock(return_value=Mock()))
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=stream))
    health = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", health)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return worker, owner, graph, stream, health


def _capture(worker: SimpleNamespace, *, allow: bool = True) -> bool:
    return bool(
        startup.capture_async_forward(
            worker, hooks_type=V2StageExecutionHooks, allow_eager_fallback=allow
        )
    )


@pytest.mark.parametrize("hooks_type", [stages._StageExecutionHooks, V2StageExecutionHooks])  # type: ignore[misc]
def test_success_publishes_only_after_release_and_sync(
    capture_runtime: CaptureRuntime, hooks_type: type[stages._StageExecutionHooks]
) -> None:
    worker, owner, graph, stream, health = capture_runtime
    assert startup.capture_async_forward(worker, hooks_type=hooks_type)
    assert cast(Mock, stages.get_stage_runner).call_args.kwargs == {"hooks_type": hooks_type}
    decode = owner.prepare_cpu.call_args.args[0]
    assert decode.kv_blocks == ((0,),)
    owner._bind_attention.assert_called_once_with(decode)
    assert not hasattr(worker, "requests")
    assert owner.graph is graph
    owner.release.assert_called_once()
    stream.synchronize.assert_called_once()
    health.assert_not_called()
    graph.clear.assert_not_called()


def test_opt_in_b4_startup_capture_uses_four_request_rows(
    capture_runtime: CaptureRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, "4")
    worker, owner, _graph, _stream, _health = capture_runtime
    assert startup.capture_async_forward(worker)
    scheduled = cast(Mock, stages.get_stage_runner).call_args.args[1]
    prefill = scheduled.gr_stage_metadata["__async_scheduling_startup_capture__"]
    decode = owner.prepare_cpu.call_args.args[0]
    assert len(prefill.request_ids) == 4
    assert prefill.num_prompt_tokens == 4
    assert prefill.num_prompt_tokens_by_request == (1, 1, 1, 1)
    assert scheduled.num_scheduled_tokens == dict.fromkeys(prefill.request_ids, 1)
    assert decode.kv_blocks == ((0,),) * 4
    owner.release.assert_called_once()


def test_invalid_startup_capture_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(startup.ENV_V1_STARTUP_CAPTURE_BATCH_SIZE, "3")
    with pytest.raises(ValueError, match="must be 1, 2, or 4"):
        startup._startup_capture_batch_size()


def test_recoverable_failure_records_reason_only_after_cleanup(
    capture_runtime: CaptureRuntime,
) -> None:
    worker, owner, _graph, stream, health = capture_runtime
    owner.capture_forward.side_effect = BeamDecodeCaptureUnsupported("capture unsupported")
    assert not _capture(worker)
    assert worker._gr_async_decode_graph_capture_failed
    assert worker._gr_async_decode_graph_capture_error == "capture unsupported"
    assert owner.graph is None and owner.closed
    owner.release.assert_called_once()
    health.assert_called_once_with(worker.device)
    stream.synchronize.assert_called_once()
    assert not stages.should_replay_decode_graph(worker, owner.graph)


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
    assert not worker._gr_async_decode_graph_capture_failed
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
    assert not worker._gr_async_decode_graph_capture_failed
    assert worker._gr_async_decode_graph_capture_error is None
    assert owner.graph is None
    assert graph.clear.call_count == int(not recoverable)


def test_failed_device_health_check_cannot_enable_fallback(capture_runtime: CaptureRuntime) -> None:
    worker, owner, _graph, _stream, health = capture_runtime
    owner.capture_forward.side_effect = BeamDecodeCaptureUnsupported("capture unsupported")
    health.side_effect = RuntimeError("CUDA illegal memory access")
    with pytest.raises(RuntimeError, match="illegal memory"):
        _capture(worker)
    assert not worker._gr_async_decode_graph_capture_failed
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
    assert not worker._gr_async_decode_graph_capture_failed
    health.assert_not_called()


def test_v1_does_not_opt_into_optional_fallback(capture_runtime: CaptureRuntime) -> None:
    worker, owner, _graph, _stream, _health = capture_runtime
    owner.capture_forward.side_effect = BeamDecodeCaptureUnsupported("capture unsupported")
    with pytest.raises(BeamDecodeCaptureUnsupported):
        startup.capture_async_forward(worker)
    assert not worker._gr_async_decode_graph_capture_failed


def test_successful_release_must_close_owner(capture_runtime: CaptureRuntime) -> None:
    worker, owner, graph, _stream, _health = capture_runtime
    owner.release.side_effect = None
    with pytest.raises(RuntimeError, match="resources were not released"):
        _capture(worker)
    assert owner.graph is None
    graph.clear.assert_called_once()


@pytest.mark.parametrize("ready", [False, True])  # type: ignore[misc]
def test_repeated_startup_accepts_only_completed_capture(
    capture_runtime: CaptureRuntime, ready: bool
) -> None:
    worker, owner, graph, _stream, _health = capture_runtime
    worker._gr_gpu_stage_runner = owner
    owner.graph = graph
    owner.closed = ready
    if ready:
        assert _capture(worker)
    else:
        with pytest.raises(RuntimeError, match="incomplete capture state"):
            _capture(worker)
    owner.capture_forward.assert_not_called()


def test_decode_graph_capture_failure_falls_back_only_when_recorded() -> None:
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True),
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
