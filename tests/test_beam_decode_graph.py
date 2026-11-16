# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.config import CUDAGraphMode

from vllm_gr.v1.worker import beam_decode_graph as shared


def test_equal_token_counts_do_not_alias_different_beam_geometries() -> None:
    graph = object.__new__(shared.BeamDecodeGraph)
    graph.wrapper = SimpleNamespace(concrete_cudagraph_entries={})
    graph._registries = {}
    graph.select(2, 4)
    descriptor = graph.descriptor
    captured = object()
    graph.wrapper.concrete_cudagraph_entries[descriptor] = captured
    graph.select(4, 2)
    assert graph.descriptor == descriptor
    assert not graph.wrapper.concrete_cudagraph_entries
    graph.select(2, 4)
    assert graph.wrapper.concrete_cudagraph_entries[descriptor] is captured


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])  # type: ignore[misc]
@pytest.mark.parametrize("is_beam", [False, True])  # type: ignore[misc]
def test_only_beam_full_uses_beam_registry(
    monkeypatch: pytest.MonkeyPatch, mode: CUDAGraphMode, is_beam: bool
) -> None:
    graph = SimpleNamespace(
        original_model=Mock(return_value="native"), model_forward=Mock(return_value="beam")
    )
    context = SimpleNamespace(
        cudagraph_runtime_mode=mode,
        attn_metadata={"layer": SimpleNamespace(is_beam_decode=is_beam)},
    )
    monkeypatch.setattr(shared, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(shared, "get_forward_context", lambda: context)
    model = object.__new__(shared._BeamDecodeModel)
    model.graph = graph
    expected_beam = is_beam and mode is CUDAGraphMode.FULL
    assert model(input_ids="fixed") == ("beam" if expected_beam else "native")
    assert graph.model_forward.call_count == int(expected_beam)
    assert graph.original_model.call_count == int(not expected_beam)


def test_context_replacement_fails_before_graph_launch() -> None:
    graph = object.__new__(shared.BeamDecodeGraph)
    graph.context = object()
    graph.signature = shared.BeamDecodeGraphSignature(1, 1, 1)
    graph._bindings = {graph.signature: (graph.context, None, None)}
    graph.worker = SimpleNamespace(_beam_context=object())
    with pytest.raises(RuntimeError, match="context changed"):
        graph.replay()


@pytest.mark.parametrize(  # type: ignore[misc]
    "installed,replacement,message",
    [
        (False, None, None),
        (True, None, None),
        (False, "_gr_input_buffers", "input storage changed"),
        (False, "model", "model changed"),
    ],
)
def test_replay_checks_captured_storage_and_model_before_launch(
    installed: bool, replacement: str | None, message: str | None
) -> None:
    from vllm_gr.v1.worker.gpu_model_runner_v2_resources import V2BeamDecodeGraph

    graph = object.__new__(V2BeamDecodeGraph)
    graph.context, graph.input_buffers, graph.original_model = object(), object(), object()
    graph.signature = shared.BeamDecodeGraphSignature(2, 4, 8)
    graph.descriptor = object()
    graph._bindings = {graph.signature: (graph.context, None, None)}
    entry = SimpleNamespace(cudagraph=Mock(), output=object())
    graph.wrapper = SimpleNamespace(concrete_cudagraph_entries={graph.descriptor: entry})
    graph.worker = SimpleNamespace(
        _beam_context=graph.context,
        _gr_input_buffers=graph.input_buffers,
        model=graph.wrapper if installed else graph.original_model,
    )
    if replacement is not None:
        setattr(graph.worker, replacement, object())
        with pytest.raises(RuntimeError, match=message):
            graph.replay()
        entry.cudagraph.replay.assert_not_called()
    else:
        assert graph.replay() is entry.output
        entry.cudagraph.replay.assert_called_once_with()


def test_native_model_discovery_unwraps_beam_wrapper() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    model = object.__new__(shared._BeamDecodeModel)
    model.runnable = object()
    assert GPUModelRunner.get_model(SimpleNamespace(model=model)) is model.runnable


def test_install_is_idempotent_and_uninstall_is_idempotent() -> None:
    original = object()
    wrapper = object()
    graph = object.__new__(shared.BeamDecodeGraph)
    graph.original_model = original
    graph.wrapper = wrapper
    graph.installed = False
    graph.worker = SimpleNamespace(model=original)

    graph.install()
    graph.install()  # no-op
    assert graph.worker.model is wrapper
    assert graph.installed

    graph.uninstall()
    graph.uninstall()  # no-op
    assert graph.worker.model is original
    assert not graph.installed


def test_install_conflicts_with_foreign_model_owner() -> None:
    original = object()
    wrapper = object()
    foreign = object()
    graph = object.__new__(shared.BeamDecodeGraph)
    graph.original_model = original
    graph.wrapper = wrapper
    graph.installed = False
    graph.worker = SimpleNamespace(model=foreign)

    with pytest.raises(RuntimeError, match="foreign model owner"):
        graph.install()


@pytest.fixture  # type: ignore[misc]
def capture_graph(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Mock, Mock]:
    graph = object.__new__(shared.BeamDecodeGraph)
    graph.width = 4
    graph.signature = shared.BeamDecodeGraphSignature(1, 4, 4)
    graph._bindings = {}
    graph._registries = {}
    graph.capture_inputs = None
    graph.capture_attention = None
    graph.descriptor = object()
    graph.original_model = object()
    graph.worker = SimpleNamespace(
        model=graph.original_model, _beam_context=object(), device="cuda"
    )
    entries: dict[Any, Any] = {}
    graph.wrapper = SimpleNamespace(
        concrete_cudagraph_entries=entries, clear_graphs=Mock(side_effect=entries.clear)
    )
    graph.installed = graph.capturing = False
    stream, capturing = Mock(), Mock()
    monkeypatch.setattr(shared.torch.cuda, "current_stream", lambda _: stream)
    monkeypatch.setattr(shared.torch.cuda, "Stream", lambda **kw: Mock())
    monkeypatch.setattr(shared.torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(shared.torch.cuda, "graph_pool_handle", lambda: (0, 123))
    monkeypatch.setattr(shared, "set_cudagraph_capturing_enabled", capturing)
    return graph, stream, capturing


@pytest.mark.l1  # type: ignore[misc]
def test_capture_rotates_private_pool_only_after_last_graph_is_discarded(
    monkeypatch: pytest.MonkeyPatch, capture_graph: tuple[Any, Mock, Mock]
) -> None:
    graph, _stream, _capturing = capture_graph
    new_pool = Mock(side_effect=[(0, 101), (0, 102)])
    monkeypatch.setattr(shared.torch.cuda, "graph_pool_handle", new_pool)

    def capture(_flag: bool) -> None:
        if _flag:
            graph.wrapper.concrete_cudagraph_entries[graph.descriptor] = SimpleNamespace(
                cudagraph=Mock(), output="hidden"
            )

    graph.select(1, 4)
    graph.capture(capture, persistent_install=False)
    assert graph.wrapper.graph_pool == (0, 101)
    graph.select(2, 4)
    graph.capture(capture, persistent_install=False)
    assert graph.wrapper.graph_pool == (0, 101)
    graph.select(1, 4)
    graph.discard_selected()
    graph.select(2, 4)
    survivor = graph.wrapper.concrete_cudagraph_entries[graph.descriptor].cudagraph
    assert graph.replay() == "hidden"
    survivor.replay.assert_called_once_with()
    graph.discard_selected()
    graph.select(1, 4)
    graph.capture(capture, persistent_install=False)
    assert graph.wrapper.graph_pool == (0, 102)
    assert new_pool.call_count == 2


@pytest.mark.l1  # type: ignore[misc]
def test_failed_first_capture_uses_a_fresh_pool_on_retry(
    monkeypatch: pytest.MonkeyPatch, capture_graph: tuple[Any, Mock, Mock]
) -> None:
    graph, _stream, _capturing = capture_graph
    new_pool = Mock(side_effect=[(0, 201), (0, 202)])
    monkeypatch.setattr(shared.torch.cuda, "graph_pool_handle", new_pool)
    graph.select(1, 4)

    def fail(capturing: bool) -> None:
        if capturing:
            raise RuntimeError("unexpected capture failure")

    with pytest.raises(RuntimeError, match="unexpected capture failure"):
        graph.capture(fail, persistent_install=False)
    graph.select(1, 4)

    def succeed(capturing: bool) -> None:
        if capturing:
            graph.wrapper.concrete_cudagraph_entries[graph.descriptor] = SimpleNamespace(
                cudagraph=Mock(), output="hidden"
            )

    graph.capture(succeed, persistent_install=False)
    assert graph.wrapper.graph_pool == (0, 202)
    assert new_pool.call_count == 2


@pytest.mark.parametrize("persistent_install", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("fail_capture", [False, True])  # type: ignore[misc]
def test_capture_model_install_policy_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    capture_graph: tuple[Any, Mock, Mock],
    persistent_install: bool,
    fail_capture: bool,
) -> None:
    # An opposite environment setting must not affect the caller's policy.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if persistent_install else "0")
    graph, _stream, capturing = capture_graph
    entries = graph.wrapper.concrete_cudagraph_entries
    calls = []

    def forward(capture: bool) -> None:
        assert graph.worker.model is graph.wrapper
        calls.append(capture)
        if capture:
            if fail_capture:
                raise RuntimeError("capture failed")
            entries[graph.descriptor] = SimpleNamespace(cudagraph=object())

    if fail_capture:
        with pytest.raises(RuntimeError, match="capture failed"):
            graph.capture(forward, persistent_install=persistent_install)
        graph.wrapper.clear_graphs.assert_called_once()
    else:
        graph.capture(forward, persistent_install=persistent_install)
        assert graph.ready
    assert calls == [False, False, True]
    assert graph.worker.model is (graph.wrapper if persistent_install else graph.original_model)
    assert graph.installed is persistent_install
    assert not graph.capturing
    assert [call.args[0] for call in capturing.call_args_list] == [True, False]


@pytest.mark.parametrize(  # type: ignore[misc]
    "phase,message,recoverable",
    [
        ("capture", "operation not supported when stream is capturing", True),
        ("warmup", "operation not supported when stream is capturing", False),
        ("sync", "operation not supported when stream is capturing", False),
        ("restore", "operation not supported when stream is capturing", False),
        ("capture", "CUDA out of memory", False),
        ("capture", "CUDA illegal memory access", False),
        ("capture", "operation failed due to a previous error during capture", False),
        ("capture", "unexpected shape", False),
    ],
)
def test_only_supported_capture_phase_errors_can_request_recovery(
    monkeypatch: pytest.MonkeyPatch,
    capture_graph: tuple[Any, Mock, Mock],
    phase: str,
    message: str,
    recoverable: bool,
) -> None:
    graph, stream, _capturing = capture_graph
    entries = graph.wrapper.concrete_cudagraph_entries
    error = RuntimeError(message)
    if phase == "sync":
        stream.synchronize.side_effect = error
    elif phase == "restore":
        monkeypatch.setattr(graph, "uninstall", Mock(side_effect=error))

    def forward(capture: bool) -> None:
        if (capture and phase in ("capture", "restore")) or (not capture and phase == "warmup"):
            raise RuntimeError(message)
        if capture:
            entries[graph.descriptor] = SimpleNamespace(cudagraph=object())

    with pytest.raises(RuntimeError) as raised:
        graph.capture(forward, persistent_install=False)
    assert isinstance(raised.value, shared.BeamDecodeCaptureUnsupported) is recoverable
    assert not graph.ready
    graph.wrapper.clear_graphs.assert_called_once()
    assert not graph.capturing
    assert graph.worker._beam_graph_keys == set()
    if phase != "restore":
        assert graph.worker.model is graph.original_model
        assert not graph.installed


pytestmark = pytest.mark.cpu_test
