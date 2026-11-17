# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def"
"""Focused tests for shared ModelRunner patch and lifecycle helpers."""

from __future__ import annotations

import gc
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.worker import gr_gc as _gc_runtime
from vllm_gr.v1.worker import model_runner_common as common
from vllm_gr.v1.worker.cuda.v2 import gpu_model_runner_v2_patch as _v2_integration


def test_execute_wrapper_is_idempotent_and_restores_outer_context() -> None:
    class ExpectedError(RuntimeError):
        pass

    scheduler_output = SimpleNamespace(beam_data={})
    runner = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))
    should_raise = False

    def execute_model(self: Any, output: Any) -> str:
        assert BEAM_DATA_VAR.get() is self._current_beam_info
        assert self._current_beam_info == {
            "beam_data": output.beam_data,
            "req_ids": [],
        }
        # Prefill graph interception needs the current scheduler output.
        assert self._current_scheduler_output is output
        if should_raise:
            raise ExpectedError
        return "result"

    wrapped = common.make_patched_execute_model(execute_model)
    assert common.make_patched_execute_model(wrapped) is wrapped

    outer = {"level": "outer"}
    token = BEAM_DATA_VAR.set(outer)
    try:
        assert wrapped(runner, scheduler_output) == "result"
        assert BEAM_DATA_VAR.get() is outer

        should_raise = True
        with pytest.raises(ExpectedError):
            wrapped(runner, scheduler_output)
        assert BEAM_DATA_VAR.get() is outer
        should_raise = False
    finally:
        BEAM_DATA_VAR.reset(token)

    # Ordinary execution retains its scheduler output, with no stale Beam data,
    # including after an exception. The outer ContextVar has been restored.
    assert runner._current_beam_data == {}
    assert runner._current_scheduler_output is scheduler_output
    assert runner._current_beam_info == {
        "beam_data": scheduler_output.beam_data,
        "req_ids": [],
    }

    # A subsequent non-beam step overwrites the stale context at entry.
    plain_output = SimpleNamespace(beam_data={})
    assert wrapped(runner, plain_output) == "result"
    assert runner._current_beam_data == {}
    assert runner._current_scheduler_output is plain_output
    assert runner._current_beam_info == {"beam_data": {}, "req_ids": []}


# ---------------------------------------------------------------------------
# M2: inlined sample_tokens (merge the second collective_rpc into
# execute_model; upstream core.py None short-circuit does the skipping)
# ---------------------------------------------------------------------------


def _m2_runner(
    *,
    beam_worker_decision: bool = True,
    use_async_scheduling: bool = False,
    pipeline_parallel_size: int = 1,
    sample_tokens_return: Any = "mro",
    sample_tokens_side_effect: Any = None,
    state: Any = "state",
) -> SimpleNamespace:
    """Runner stub exercising the M2 gate + inline path in isolation."""
    calls: list[Any] = []

    def sample_tokens(grammar_output: Any) -> Any:
        calls.append(grammar_output)
        if sample_tokens_side_effect is not None:
            raise sample_tokens_side_effect
        return sample_tokens_return

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config={}),
        _vllm_gr_config=SimpleNamespace(
            beam_worker_decision=beam_worker_decision,
        ),
        use_async_scheduling=use_async_scheduling,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
        sample_tokens=sample_tokens,
        execute_model_state=state,
        _sample_tokens_calls=calls,
    )
    return runner


def _orig_execute_model_none(self: Any, scheduler_output: Any) -> None:
    """Upstream async-scheduling-style execute_model: stashes state, returns None."""
    self.execute_model_state = "state"
    return None


def test_execute_wrapper_does_not_inline_when_gate_closed() -> None:
    """Gate closed (async scheduling): execute_model returns None and never
    calls sample_tokens — the engine issues the second RPC itself."""
    wrapped = common.make_patched_execute_model(_orig_execute_model_none)
    runner = _m2_runner(use_async_scheduling=True)
    scheduler_output = SimpleNamespace(beam_data={})

    out = wrapped(runner, scheduler_output)

    assert out is None
    assert runner._sample_tokens_calls == []
    # State survives for the engine-initiated sample_tokens RPC.
    assert runner.execute_model_state == "state"


def test_execute_wrapper_does_not_inline_non_none_output() -> None:
    """A runner that already returned a value (e.g. pooling) is untouched."""
    wrapped = common.make_patched_execute_model(lambda self, output: "direct")
    runner = _m2_runner()
    scheduler_output = SimpleNamespace(beam_data={})

    assert wrapped(runner, scheduler_output) == "direct"
    assert runner._sample_tokens_calls == []


# ---------------------------------------------------------------------------
# Warmup GC freeze: quiescent boundary + RPC completion barrier
# ---------------------------------------------------------------------------


def _reset_warmup_gc_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the process-level freeze state per test."""
    monkeypatch.setattr(_gc_runtime, "_GC_WARMUP_FREEZE_DONE", False)
    monkeypatch.setattr(_gc_runtime, "_GC_WARMUP_FREEZE_COMPLETE", False)
    monkeypatch.setattr(_gc_runtime, "_WORKER_EXECUTE_DEPTH", 0)
    monkeypatch.setattr(_gc_runtime, "_GC_FREEZE_IN_PROGRESS", False)
    # Keep the quiescent fallback wait short in tests.
    monkeypatch.setattr(_gc_runtime, "_QUIESCENT_WAIT_TIMEOUT_S", 2.0)


def _freeze_daemon_alive() -> bool:
    return any(t.name == "vllm-gr-gc-freeze" for t in threading.enumerate())


def _wait_freeze_daemon_done(timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _freeze_daemon_alive():
            return True
        time.sleep(0.02)
    return False


def test_warmup_rpc_does_not_acknowledge_scheduled_pending_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review regression: frozen=True must imply a *completed* freeze.

    The automatic daemon claims the one-shot before its collection runs; an
    RPC racing that collector must not return frozen=True while the heap is
    still unfrozen (probe observed thresholds still [700, 10, 10]).
    """
    _reset_warmup_gc_state(monkeypatch)
    calls: list[bool] = []

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        calls.append(raise_thresholds)
        return [700, 100_000, 100_000]

    monkeypatch.setattr(_gc_runtime, "_freeze_serving_gc", fake_freeze)
    # The automatic daemon claimed the one-shot; collection still pending.
    monkeypatch.setattr(_gc_runtime, "_GC_WARMUP_FREEZE_DONE", True)

    result = _gc_runtime.freeze_gc_after_warmup(SimpleNamespace(model_runner=None))

    assert result["frozen"] is True
    assert "deduped" not in result
    # The RPC performed the freeze itself, so completion is guaranteed
    # before frozen=True was returned.
    assert calls == [True]
    assert _gc_runtime._GC_WARMUP_FREEZE_COMPLETE is True


def test_warmup_rpc_dedupes_only_completed_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_warmup_gc_state(monkeypatch)
    calls: list[bool] = []

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        calls.append(raise_thresholds)
        return [700, 100_000, 100_000]

    monkeypatch.setattr(_gc_runtime, "_freeze_serving_gc", fake_freeze)
    monkeypatch.setattr(_gc_runtime, "_GC_WARMUP_FREEZE_COMPLETE", True)

    result = _gc_runtime.freeze_gc_after_warmup(SimpleNamespace(model_runner=None))

    assert result["frozen"] is True
    assert result["deduped"] is True
    assert result["thresholds"] == list(gc.get_threshold())
    assert calls == []


def test_warmup_rpc_failure_rolls_back_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed acknowledged collector must not strand the one-shot."""
    _reset_warmup_gc_state(monkeypatch)

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        raise RuntimeError("collect failed")

    monkeypatch.setattr(_gc_runtime, "_freeze_serving_gc", fake_freeze)

    result = _gc_runtime.freeze_gc_after_warmup(SimpleNamespace(model_runner=None))

    assert result["frozen"] is False
    assert "error" in result
    assert _gc_runtime._GC_WARMUP_FREEZE_DONE is False
    assert _gc_runtime._GC_WARMUP_FREEZE_COMPLETE is False


def test_warmup_daemon_waits_for_quiescent_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review regression: no collection while execute_model is in flight.

    The teardown signal is consumed at the START of the next execute_model;
    the freeze daemon must wait for that step to finish before collecting
    (gc.collect holds the GIL and would stall the request).
    """
    _reset_warmup_gc_state(monkeypatch)
    started = threading.Event()

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        started.set()
        return [700, 100_000, 100_000]

    monkeypatch.setattr(_gc_runtime, "_freeze_serving_gc", fake_freeze)

    _gc_runtime._enter_worker_execute()
    try:
        assert _gc_runtime.schedule_warmup_gc_freeze() is True
        assert _gc_runtime.schedule_warmup_gc_freeze() is False
        assert not started.wait(0.2), "freeze ran while execute_model was in flight"
    finally:
        _gc_runtime._exit_worker_execute()

    assert started.wait(5.0), "freeze did not run at the quiescent boundary"
    assert _gc_runtime._GC_WARMUP_FREEZE_COMPLETE is True


def test_warmup_daemon_blocks_execute_reentry_until_freeze_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-depth observation and freeze form one admission boundary."""
    _reset_warmup_gc_state(monkeypatch)
    freeze_started = threading.Event()
    release_freeze = threading.Event()
    execute_entered = threading.Event()

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        freeze_started.set()
        assert release_freeze.wait(5.0)
        return [700, 100_000, 100_000]

    def enter_execute() -> None:
        _gc_runtime._enter_worker_execute()
        try:
            execute_entered.set()
        finally:
            _gc_runtime._exit_worker_execute()

    monkeypatch.setattr(_gc_runtime, "_freeze_serving_gc", fake_freeze)
    assert _gc_runtime.schedule_warmup_gc_freeze() is True
    assert freeze_started.wait(5.0)

    thread = threading.Thread(target=enter_execute)
    thread.start()
    assert not execute_entered.wait(0.2), "execute entered during the claimed GC boundary"

    release_freeze.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert execute_entered.is_set()
    assert _wait_freeze_daemon_done()
    assert _gc_runtime._GC_FREEZE_IN_PROGRESS is False


def test_warmup_daemon_skips_when_rpc_already_froze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduled daemon must not re-collect after an RPC freeze."""
    _reset_warmup_gc_state(monkeypatch)
    calls: list[bool] = []

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        calls.append(raise_thresholds)
        return [700, 100_000, 100_000]

    monkeypatch.setattr(_gc_runtime, "_freeze_serving_gc", fake_freeze)
    # Park the daemon behind the quiescence wait (a step is in flight), then
    # let the explicit RPC barrier complete a freeze while it waits.
    _gc_runtime._enter_worker_execute()
    try:
        assert _gc_runtime.schedule_warmup_gc_freeze() is True
        monkeypatch.setattr(_gc_runtime, "_GC_WARMUP_FREEZE_COMPLETE", True)
    finally:
        _gc_runtime._exit_worker_execute()

    assert _wait_freeze_daemon_done()
    assert calls == []


def test_execute_wrapper_tracks_inflight_depth_for_gc_freeze() -> None:
    depth_seen: dict[str, int] = {}

    def execute_model(self: Any, scheduler_output: Any) -> str:
        depth_seen["during"] = _gc_runtime._WORKER_EXECUTE_DEPTH
        return "result"

    wrapped = common.make_patched_execute_model(execute_model)
    runner = _m2_runner()
    scheduler_output = SimpleNamespace(beam_data={})

    assert _gc_runtime._WORKER_EXECUTE_DEPTH == 0
    assert wrapped(runner, scheduler_output) == "result"
    assert depth_seen["during"] == 1
    assert _gc_runtime._WORKER_EXECUTE_DEPTH == 0


def test_execute_wrapper_releases_depth_on_exception() -> None:
    class ExpectedError(Exception):
        pass

    def execute_model(self: Any, scheduler_output: Any) -> None:
        assert _gc_runtime._WORKER_EXECUTE_DEPTH == 1
        raise ExpectedError

    wrapped = common.make_patched_execute_model(execute_model)
    runner = _m2_runner()
    scheduler_output = SimpleNamespace(beam_data={})

    with pytest.raises(ExpectedError):
        wrapped(runner, scheduler_output)
    assert _gc_runtime._WORKER_EXECUTE_DEPTH == 0


# ---------------------------------------------------------------------------
# V2 ordinary frames remain native; no global dispatch context is needed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raises", [False, True])  # type: ignore[misc]
def test_v2_plain_frame_native_execution_and_gc_cleanup(raises: bool) -> None:
    runner = SimpleNamespace()
    output = SimpleNamespace(beam_data={})

    def native(self: Any, scheduled: Any) -> str:
        assert self is runner and scheduled is output
        assert _gc_runtime._WORKER_EXECUTE_DEPTH == 1
        if raises:
            raise RuntimeError("native failed")
        return "result"

    wrapped = _v2_integration.make_patched_execute_model_v2(native)
    if raises:
        with pytest.raises(RuntimeError, match="native failed"):
            wrapped(runner, output)
    else:
        assert wrapped(runner, output) == "result"
    assert _gc_runtime._WORKER_EXECUTE_DEPTH == 0


def test_slim_v2_wrapper_fail_fast_on_legacy_beam_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refactor §4.3: the slim V2 execute_model wrapper must raise loudly on a
    legacy beam_data frame instead of silently running the (deleted) LEGACY
    stack under a runner whose patch layer no longer supports it."""
    scheduler_output = SimpleNamespace(beam_data={"beam": {"is_beam_decode": True}})
    runner = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))

    def execute_model(self: Any, output: Any) -> str:
        raise AssertionError("native execute_model must not run for a beam_data frame")

    wrapped = _v2_integration.make_patched_execute_model_v2(execute_model)
    with pytest.raises(ValueError, match="legacy beam_data frames are not supported"):
        wrapped(runner, scheduler_output)


def test_slim_v2_wrapper_is_idempotent() -> None:
    def execute_model(self: Any, output: Any) -> str:
        return "result"

    wrapped = _v2_integration.make_patched_execute_model_v2(execute_model)
    assert _v2_integration.make_patched_execute_model_v2(wrapped) is wrapped


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("raises", [False, True])  # type: ignore[misc]
def test_v1_v2_wrappers_share_one_gc_execute_depth(
    monkeypatch: pytest.MonkeyPatch, raises: bool
) -> None:
    _reset_warmup_gc_state(monkeypatch)
    v1_runner, v2_runner = _m2_runner(), SimpleNamespace()
    scheduled = SimpleNamespace(beam_data={})
    error = RuntimeError("nested native failure")

    def native_v1(runner: Any, output: Any) -> str:
        assert runner is v1_runner and output is scheduled
        assert _gc_runtime._WORKER_EXECUTE_DEPTH == 2
        if raises:
            raise error
        return "native result"

    v1 = common.make_patched_execute_model(native_v1)

    def native_v2(runner: Any, output: Any) -> Any:
        assert runner is v2_runner and output is scheduled
        assert _gc_runtime._WORKER_EXECUTE_DEPTH == 1
        return v1(v1_runner, output)

    v2 = _v2_integration.make_patched_execute_model_v2(native_v2)
    if raises:
        with pytest.raises(RuntimeError) as caught:
            v2(v2_runner, scheduled)
        assert caught.value is error
    else:
        assert v2(v2_runner, scheduled) == "native result"
    assert _gc_runtime._WORKER_EXECUTE_DEPTH == 0
    assert not _gc_runtime._GC_FREEZE_IN_PROGRESS
    assert not hasattr(v2_runner, "_current_beam_info")
