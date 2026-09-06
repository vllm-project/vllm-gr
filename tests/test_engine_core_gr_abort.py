# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from unittest import mock

import pytest
from vllm.v1.executor.multiproc_executor import FutureWrapper, MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_gr.v1.engine.engine_core_patch import make_patched_abort_requests


class _Worker:
    def __init__(self, release_error: BaseException | None = None) -> None:
        self.release_error = release_error
        self.release_calls: list[list[str]] = []
        self.shutdown_calls = 0

    def release_gr_model_runner_states(self, request_ids: list[str]):
        self.release_calls.append(request_ids)
        if self.release_error is not None:
            raise self.release_error
        return "released"

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _uniproc(worker: _Worker) -> UniProcExecutor:
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = worker
    return executor


def _request(*, persistent: bool):
    return SimpleNamespace(gr_persistent_runner_state=persistent)


def _engine(executor, requests):
    return SimpleNamespace(
        scheduler=SimpleNamespace(requests=dict(requests)),
        model_executor=executor,
        add_request=mock.Mock(name="add_request"),
        step=mock.Mock(name="step"),
        step_with_batch_queue=mock.Mock(name="step_with_batch_queue"),
        step_fn=mock.Mock(name="step_fn"),
        shutdown=mock.Mock(name="shutdown", side_effect=executor.shutdown),
    )


def _removing_abort(calls: list[list[str]], result="done"):
    def original(engine, request_ids):
        calls.append(request_ids)
        for request_id in request_ids:
            engine.scheduler.requests.pop(request_id, None)
        return result

    return original


def test_abort_observes_actual_uniproc_collective_rpc_success() -> None:
    worker = _Worker()
    executor = _uniproc(worker)
    calls: list[list[str]] = []
    abort = make_patched_abort_requests(_removing_abort(calls))
    engine = _engine(
        executor,
        {
            "gr-0": _request(persistent=True),
            "plain-0": _request(persistent=False),
        },
    )
    healthy_methods = tuple(
        getattr(engine, name)
        for name in ("add_request", "step", "step_with_batch_queue", "step_fn")
    )
    shutdown = engine.shutdown
    rpc_function = executor.collective_rpc.__func__

    assert abort(engine, ["gr-0", "plain-0", "missing"]) == "done"

    assert calls == [["gr-0", "plain-0", "missing"]]
    assert worker.release_calls == [["gr-0"]]
    assert not hasattr(engine, "_gr_lifecycle_error")
    assert tuple(
        getattr(engine, name)
        for name in ("add_request", "step", "step_with_batch_queue", "step_fn")
    ) == healthy_methods
    assert engine.shutdown is shutdown
    assert executor.collective_rpc.__func__ is rpc_function


def test_abort_observes_actual_uniproc_failed_future_and_retires_engine() -> None:
    error = RuntimeError("worker release failed")
    worker = _Worker(error)
    executor = _uniproc(worker)
    calls: list[list[str]] = []
    abort = make_patched_abort_requests(_removing_abort(calls))
    engine = _engine(executor, {"gr-0": _request(persistent=True)})
    shutdown = engine.shutdown

    with pytest.raises(RuntimeError, match="worker release failed") as caught:
        abort(engine, "gr-0")

    assert caught.value is error
    assert calls == [["gr-0"]]
    assert worker.release_calls == [["gr-0"]]
    assert engine._gr_lifecycle_error is error
    assert caught.value.__notes__ == [
        "Persistent GR abort release failed for ['gr-0']"
    ]
    assert engine.shutdown is shutdown


def test_direct_release_submission_error_is_propagated_and_retires_engine() -> None:
    error = RuntimeError("release submission failed")
    executor = SimpleNamespace(
        collective_rpc=mock.Mock(side_effect=error), shutdown=mock.Mock()
    )
    calls: list[list[str]] = []
    abort = make_patched_abort_requests(_removing_abort(calls))
    engine = _engine(executor, {"gr-0": _request(persistent=True)})

    with pytest.raises(RuntimeError, match="release submission failed") as caught:
        abort(engine, ["gr-0"])

    assert caught.value is error
    assert calls == [["gr-0"]]
    assert engine._gr_lifecycle_error is error
    assert caught.value.__notes__ == [
        "Persistent GR abort release failed for ['gr-0']"
    ]


def test_abort_waits_for_actual_multiproc_future_fifo_without_losing_prior_output() -> None:
    fifo: deque[FutureWrapper] = deque()
    response_order: list[str] = []

    def prior_response():
        response_order.append("prior-output")
        return "model-output"

    prior = FutureWrapper(fifo, prior_response)

    def collective_rpc(method, *, args, non_block):
        assert method == "release_gr_model_runner_states"
        assert args == (["gr-0"],)
        assert non_block is True

        def release_response():
            response_order.append("release")
            return ["released"]

        return FutureWrapper(fifo, release_response)

    executor = SimpleNamespace(
        collective_rpc=collective_rpc,
        shutdown=mock.Mock(),
    )
    abort = make_patched_abort_requests(_removing_abort([]))
    engine = _engine(executor, {"gr-0": _request(persistent=True)})

    assert abort(engine, ["gr-0"]) == "done"
    assert response_order == ["prior-output", "release"]
    assert prior.done() and prior.result() == "model-output"
    assert not fifo
    assert not hasattr(engine, "_gr_lifecycle_error")


def test_delayed_multiproc_release_failure_preserves_prior_output_and_retires() -> None:
    fifo: deque[FutureWrapper] = deque()
    response_order: list[str] = []
    error = RuntimeError("delayed release failed")

    def prior_response():
        response_order.append("prior-output")
        return "model-output"

    prior = FutureWrapper(fifo, prior_response)

    def collective_rpc(method, *, args, non_block):
        assert (method, args, non_block) == (
            "release_gr_model_runner_states",
            (["gr-0"],),
            True,
        )

        def release_response():
            response_order.append("release")
            raise error

        return FutureWrapper(fifo, release_response)

    executor = SimpleNamespace(
        collective_rpc=collective_rpc,
        shutdown=mock.Mock(),
    )
    abort = make_patched_abort_requests(_removing_abort([]))
    engine = _engine(executor, {"gr-0": _request(persistent=True)})

    with pytest.raises(RuntimeError, match="delayed release failed") as caught:
        abort(engine, ["gr-0"])

    assert caught.value is error
    assert response_order == ["prior-output", "release"]
    assert prior.done() and prior.result() == "model-output"
    assert not fifo
    assert engine._gr_lifecycle_error is error


def test_retired_engine_rejects_work_and_rpc_but_keeps_shutdown_reachable() -> None:
    error = RuntimeError("worker release failed")
    worker = _Worker(error)
    executor = _uniproc(worker)
    abort = make_patched_abort_requests(_removing_abort([]))
    engine = _engine(executor, {"gr-0": _request(persistent=True)})
    shutdown = engine.shutdown

    with pytest.raises(RuntimeError, match="worker release failed"):
        abort(engine, ["gr-0"])

    for name in ("add_request", "step", "step_with_batch_queue", "step_fn"):
        with pytest.raises(RuntimeError, match="EngineCore retired") as caught:
            getattr(engine, name)(object())
        assert caught.value.__cause__ is error

    rejected_rpc_calls = (
        lambda: executor.determine_available_memory(),
        lambda: executor.execute_dummy_batch(),
        lambda: executor.collective_rpc("utility_rpc"),
    )
    for invoke in rejected_rpc_calls:
        with pytest.raises(RuntimeError, match="EngineCore retired") as caught:
            invoke()
        assert caught.value.__cause__ is error

    assert engine.shutdown is shutdown
    engine.shutdown()
    assert worker.shutdown_calls == 1


def test_plain_repeated_and_unknown_aborts_do_not_dispatch_extra_release_rpc() -> None:
    worker = _Worker()
    executor = _uniproc(worker)
    calls: list[list[str]] = []
    abort = make_patched_abort_requests(_removing_abort(calls))
    engine = _engine(
        executor,
        {
            "gr-0": _request(persistent=True),
            "plain-0": _request(persistent=False),
        },
    )

    assert abort(engine, ["plain-0", "missing"]) == "done"
    assert abort(engine, ["gr-0"]) == "done"
    assert abort(engine, ["gr-0", "plain-0", "missing"]) == "done"

    assert calls == [
        ["plain-0", "missing"],
        ["gr-0"],
        ["gr-0", "plain-0", "missing"],
    ]
    assert worker.release_calls == [["gr-0"]]


def test_release_future_result_is_called_without_timeout() -> None:
    class ResultSpy(Future):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[tuple, dict]] = []

        def result(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "released"

    release = ResultSpy()
    executor = SimpleNamespace(
        collective_rpc=mock.Mock(return_value=release),
        shutdown=mock.Mock(),
    )
    abort = make_patched_abort_requests(_removing_abort([]))
    engine = _engine(executor, {"gr-0": _request(persistent=True)})

    assert abort(engine, ["gr-0"]) == "done"
    executor.collective_rpc.assert_called_once_with(
        "release_gr_model_runner_states",
        args=(["gr-0"],),
        non_block=True,
    )
    assert release.calls == [((), {})]


def test_second_abort_preserves_first_failure_while_another_engine_stays_healthy() -> None:
    first_error = RuntimeError("first engine release failed")
    first_worker = _Worker(first_error)
    calls: list[list[str]] = []
    abort = make_patched_abort_requests(_removing_abort(calls))
    retired = _engine(
        _uniproc(first_worker), {"gr-first": _request(persistent=True)}
    )

    with pytest.raises(RuntimeError, match="first engine release failed"):
        abort(retired, ["gr-first"])
    first_notes = tuple(first_error.__notes__)

    with pytest.raises(RuntimeError, match="EngineCore retired") as caught:
        abort(retired, ["gr-never-submitted"])

    assert caught.value.__cause__ is first_error
    assert retired._gr_lifecycle_error is first_error
    assert tuple(first_error.__notes__) == first_notes
    assert calls == [["gr-first"]]
    assert first_worker.release_calls == [["gr-first"]]

    healthy_worker = _Worker()
    healthy = _engine(
        _uniproc(healthy_worker), {"gr-healthy": _request(persistent=True)}
    )
    assert abort(healthy, ["gr-healthy"]) == "done"
    assert calls == [["gr-first"], ["gr-healthy"]]
    assert healthy_worker.release_calls == [["gr-healthy"]]
    assert not hasattr(healthy, "_gr_lifecycle_error")


def test_retirement_replaces_prebound_step_fn_instance_pointer() -> None:
    error = RuntimeError("release failed after step_fn capture")
    executor = SimpleNamespace(
        collective_rpc=mock.Mock(side_effect=error), shutdown=mock.Mock()
    )
    abort = make_patched_abort_requests(_removing_abort([]))
    engine = _engine(executor, {"gr-0": _request(persistent=True)})
    old_operation = mock.Mock(name="old_captured_step")

    def old_step(_self):
        return old_operation()

    captured_step_fn = MethodType(old_step, engine)
    engine.step_fn = captured_step_fn

    with pytest.raises(RuntimeError, match="release failed after step_fn capture"):
        abort(engine, ["gr-0"])

    assert engine.step_fn is not captured_step_fn
    with pytest.raises(RuntimeError, match="EngineCore retired") as caught:
        engine.step_fn()
    assert caught.value.__cause__ is error
    old_operation.assert_not_called()


def test_actual_multiproc_shutdown_remains_reachable_after_retirement() -> None:
    error = RuntimeError("multiproc release submission failed")
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    rpc_queue = SimpleNamespace(shutdown=mock.Mock(name="rpc_queue_shutdown"))
    response_0 = SimpleNamespace(shutdown=mock.Mock(name="response_0_shutdown"))
    response_1 = SimpleNamespace(shutdown=mock.Mock(name="response_1_shutdown"))
    executor.workers = []
    executor.rpc_broadcast_mq = rpc_queue
    executor.response_mqs = [response_0, response_1]
    executor.collective_rpc = mock.Mock(side_effect=error)
    engine = _engine(executor, {"gr-0": _request(persistent=True)})
    abort = make_patched_abort_requests(_removing_abort([]))
    shutdown = engine.shutdown

    with pytest.raises(RuntimeError, match="multiproc release submission failed"):
        abort(engine, ["gr-0"])

    assert engine.shutdown is shutdown
    with pytest.raises(RuntimeError, match="EngineCore retired"):
        executor.collective_rpc("utility_rpc")
    engine.shutdown()

    rpc_queue.shutdown.assert_called_once_with()
    response_0.shutdown.assert_called_once_with()
    response_1.shutdown.assert_called_once_with()
    assert executor.rpc_broadcast_mq is None
    assert executor.response_mqs == []
    assert executor.shutting_down is True
