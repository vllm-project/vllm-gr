# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the V2 prefill-dispatch patch + capture wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.v1.worker import gpu_model_runner_v2_graph as v2g
from vllm_gr.v1.worker.gpu_beam_stage_runner import should_replay_decode_graph


def _patch_capture_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    execution_mode: Any,
) -> None:
    import vllm_gr.v1.worker.gr_startup as startup

    monkeypatch.setattr(
        v2g,
        "_get_gr_config",
        lambda _runner: SimpleNamespace(beam_execution_mode=execution_mode),
    )
    monkeypatch.setattr(startup, "capture_async_forward", lambda _runner: events.append("async"))


def test_capture_wrapper_orders_all_v1_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.config import BeamExecutionMode

    events: list[str] = []
    _patch_capture_dependencies(
        monkeypatch,
        events,
        execution_mode=BeamExecutionMode.V1,
    )

    def native(_runner: Any) -> int:
        events.append("native")
        return 17

    runner = SimpleNamespace()
    wrapped = v2g.make_patched_capture_model_v2(native)
    assert wrapped(runner) == 17
    assert events == ["native", "async"]
    assert runner._gr_async_decode_graph_capture_failed is False


def test_capture_wrapper_preserves_native_only_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _patch_capture_dependencies(monkeypatch, events, execution_mode=None)

    def native(_runner: Any) -> int:
        events.append("native")
        return 23

    wrapped = v2g.make_patched_capture_model_v2(native)
    assert wrapped(object()) == 23
    assert events == ["native"]


@pytest.mark.parametrize("failing_capture", ["async"])  # type: ignore[misc]
def test_capture_wrapper_preserves_optional_capture_fallback(
    monkeypatch: pytest.MonkeyPatch,
    failing_capture: str,
) -> None:
    import vllm_gr.v1.worker.gr_startup as startup
    from vllm_gr.config import BeamExecutionMode

    events: list[str] = []
    _patch_capture_dependencies(
        monkeypatch,
        events,
        execution_mode=BeamExecutionMode.V1,
    )

    def fail(_runner: Any) -> None:
        events.append(failing_capture)
        raise RuntimeError(failing_capture)

    monkeypatch.setattr(
        startup,
        "capture_async_forward",
        fail,
    )

    def native(_runner: Any) -> int:
        events.append("native")
        return 29

    runner = SimpleNamespace()
    wrapped = v2g.make_patched_capture_model_v2(native)
    assert wrapped(runner) == 29
    assert events == ["native", "async"]
    assert runner._gr_async_decode_graph_capture_failed is (failing_capture == "async")


def test_capture_wrapper_propagates_native_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _patch_capture_dependencies(monkeypatch, events, execution_mode=None)

    def native(_runner: Any) -> None:
        events.append("native")
        raise RuntimeError("native")

    wrapped = v2g.make_patched_capture_model_v2(native)
    with pytest.raises(RuntimeError, match="native"):
        wrapped(object())
    assert events == ["native"]


def test_decode_graph_capture_failure_falls_back_only_when_recorded() -> None:
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True),
        _gr_async_decode_graph_capture_failed=True,
    )
    assert should_replay_decode_graph(worker, graph=None) is False

    worker._gr_async_decode_graph_capture_failed = False
    with pytest.raises(RuntimeError, match="requires successful startup capture"):
        should_replay_decode_graph(worker, graph=None)

    assert should_replay_decode_graph(worker, graph=object()) is True
    worker.model_config.enforce_eager = True
    assert should_replay_decode_graph(worker, graph=None) is False
