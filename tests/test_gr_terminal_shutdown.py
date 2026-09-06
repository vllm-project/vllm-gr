# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for persistent-GR terminal shutdown ownership."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

import pytest
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus

from vllm_gr.entrypoints import gr as offline_gr
from vllm_gr.v1.core.sched.async_scheduler import GRAsyncScheduler


def _request(*, persistent: bool, gr: bool = True):
    return SimpleNamespace(
        is_gr_request=gr,
        gr_persistent_runner_state=persistent,
    )


def _scheduler(requests, events: list[str] | None = None):
    scheduler = GRAsyncScheduler.__new__(GRAsyncScheduler)
    scheduler.requests = dict(requests)
    scheduler.kv_event_publisher = None
    scheduler.connector = SimpleNamespace(
        shutdown=mock.Mock(
            side_effect=(
                (lambda: events.append("scheduler_shutdown"))
                if events is not None
                else None
            )
        )
    )
    finish_calls = []

    def finish_requests(request_ids, status):
        ids = list(request_ids)
        finish_calls.append((ids, status))
        if events is not None:
            events.append("native_finish")
        for request_id in ids:
            scheduler.requests.pop(request_id, None)
        return [(request_id, 0) for request_id in ids]

    scheduler.finish_requests = finish_requests
    return scheduler, finish_calls


class _OutputProcessor:
    def __init__(
        self,
        *,
        abort_failures: int = 0,
        events: list[str] | None = None,
    ) -> None:
        self.request_states = {
            # This marked request models shutdown before EngineCore admission.
            "frontend-only-persistent": SimpleNamespace(
                external_req_id="mixed-external",
                gr_persistent_request=True,
            ),
            "second-persistent": SimpleNamespace(
                external_req_id="persistent-only-external",
                gr_persistent_request=True,
            ),
            # Its name resembles the legacy frontend convention, but the explicit
            # marker says it is ordinary and therefore shutdown must retain it.
            "offline-beam-7-0": SimpleNamespace(
                external_req_id="ordinary-external",
                gr_persistent_request=False,
            ),
            "unmarked-ordinary": SimpleNamespace(
                external_req_id="mixed-external",
            ),
        }
        self.external_req_ids = defaultdict(
            list,
            {
                "mixed-external": [
                    "frontend-only-persistent",
                    "unmarked-ordinary",
                ],
                "persistent-only-external": ["second-persistent"],
                "ordinary-external": ["offline-beam-7-0"],
            },
        )
        self.abort_failures = abort_failures
        self.events = events
        self.abort_error = RuntimeError("frontend abort failed")
        self.abort_calls: list[tuple[list[str], bool]] = []

    def abort_requests(self, request_ids, internal):
        request_ids = list(request_ids)
        self.abort_calls.append((request_ids, internal))
        if self.events is not None:
            self.events.append("frontend_abort")
        if self.abort_failures:
            self.abort_failures -= 1
            raise self.abort_error
        assert internal is True
        for request_id in request_ids:
            state = self.request_states.pop(request_id)
            internal_ids = self.external_req_ids[state.external_req_id]
            internal_ids.remove(request_id)
            if not internal_ids:
                del self.external_req_ids[state.external_req_id]
        return request_ids


def _output_processor(
    *, abort_failures: int = 0, events: list[str] | None = None
):
    return _OutputProcessor(abort_failures=abort_failures, events=events)


def _grllm(output_processor, engine=None):
    llm = offline_gr.GRLLM.__new__(offline_gr.GRLLM)
    llm._shutdown_complete = False
    llm.llm_engine = (
        SimpleNamespace(output_processor=output_processor)
        if engine is None
        else engine
    )
    return llm


def _patch_terminal_cleanup(monkeypatch, shutdown):
    from vllm_gr.utils import engine_lifecycle

    collect = mock.Mock(name="gc_collect")
    release = mock.Mock(name="release_accelerator_memory")
    monkeypatch.setattr(engine_lifecycle, "shutdown_sync_engine", shutdown)
    monkeypatch.setattr(offline_gr.gc, "collect", collect)
    monkeypatch.setattr(offline_gr, "release_accelerator_memory", release)
    return collect, release


def _real_core_engine(
    monkeypatch,
    output_processor,
    events: list[str],
    *,
    drain_error: BaseException | None = None,
):
    from vllm.v1.engine import core as core_module

    scheduler, finish_calls = _scheduler(
        {"persistent": _request(persistent=True)}, events
    )

    def drain_worker() -> None:
        events.append("worker_drain")
        if drain_error is not None:
            raise drain_error

    core = EngineCore.__new__(EngineCore)
    core.structured_output_manager = SimpleNamespace(clear_backend=mock.Mock())
    core.model_executor = SimpleNamespace(shutdown=mock.Mock(side_effect=drain_worker))
    core.scheduler = scheduler
    unfreeze = mock.Mock(name="gc_unfreeze")
    cleanup_dist = mock.Mock(name="cleanup_dist_env_and_memory")
    monkeypatch.setattr(core_module.gc, "unfreeze", unfreeze)
    monkeypatch.setattr(core_module, "cleanup_dist_env_and_memory", cleanup_dist)

    engine = SimpleNamespace(
        output_processor=output_processor,
        shutdown=lambda: EngineCore.shutdown(core),
    )
    return engine, core, scheduler, finish_calls, unfreeze, cleanup_dist


def test_scheduler_shutdown_finishes_only_live_persistent_native_requests() -> None:
    scheduler, finish_calls = _scheduler(
        {
            "persistent": _request(persistent=True),
            "legacy-gr": _request(persistent=False),
            "ordinary": _request(persistent=False, gr=False),
        }
    )

    scheduler.shutdown()

    assert finish_calls == [
        (["persistent"], RequestStatus.FINISHED_ABORTED)
    ]
    assert set(scheduler.requests) == {"legacy-gr", "ordinary"}
    scheduler.connector.shutdown.assert_called_once_with()






def test_grllm_failed_shutdown_preserves_engine_and_records_for_retry(
    monkeypatch,
) -> None:
    output_processor = _output_processor()
    llm = _grllm(output_processor)
    engine = llm.llm_engine
    failure = RuntimeError("engine shutdown failed")
    shutdown = mock.Mock(side_effect=(failure, None))
    collect, release = _patch_terminal_cleanup(monkeypatch, shutdown)
    states_before = dict(output_processor.request_states)
    mappings_before = {
        external: list(internal)
        for external, internal in output_processor.external_req_ids.items()
    }

    with pytest.raises(RuntimeError, match="engine shutdown failed") as caught:
        llm.shutdown()

    assert caught.value is failure
    assert llm._shutdown_complete is False
    assert not getattr(llm, "_engine_shutdown_complete", False)
    assert llm.llm_engine is engine
    assert output_processor.request_states == states_before
    assert dict(output_processor.external_req_ids) == mappings_before
    collect.assert_not_called()
    release.assert_not_called()
    assert output_processor.abort_calls == []

    llm.shutdown()
    assert shutdown.call_args_list == [mock.call(engine), mock.call(engine)]
    assert llm._shutdown_complete is True
    assert llm._engine_shutdown_complete is True
    assert llm.llm_engine is None
    assert set(output_processor.request_states) == {
        "offline-beam-7-0",
        "unmarked-ordinary",
    }
    assert dict(output_processor.external_req_ids) == {
        "mixed-external": ["unmarked-ordinary"],
        "ordinary-external": ["offline-beam-7-0"],
    }
    collect.assert_called_once_with()
    release.assert_called_once_with()




def test_real_engine_core_shutdown_orders_worker_native_and_frontend_release(
    monkeypatch,
) -> None:
    events: list[str] = []
    output_processor = _output_processor(events=events)
    engine, core, scheduler, finish_calls, unfreeze, cleanup_dist = (
        _real_core_engine(monkeypatch, output_processor, events)
    )
    llm = _grllm(output_processor, engine)
    collect = mock.Mock(name="outer_gc_collect")
    release = mock.Mock(name="release_accelerator_memory")
    monkeypatch.setattr(offline_gr.gc, "collect", collect)
    monkeypatch.setattr(offline_gr, "release_accelerator_memory", release)

    llm.shutdown()

    assert events == [
        "worker_drain",
        "native_finish",
        "scheduler_shutdown",
        "frontend_abort",
    ]
    assert finish_calls == [
        (["persistent"], RequestStatus.FINISHED_ABORTED)
    ]
    core.model_executor.shutdown.assert_called_once_with()
    scheduler.connector.shutdown.assert_called_once_with()
    unfreeze.assert_called_once_with()
    cleanup_dist.assert_called_once_with()
    collect.assert_called_once_with()
    release.assert_called_once_with()
    assert llm._engine_shutdown_complete is True
    assert llm._shutdown_complete is True


def test_real_engine_core_drain_failure_preserves_native_and_frontend_ownership(
    monkeypatch,
) -> None:
    events: list[str] = []
    error = RuntimeError("worker drain failed")
    output_processor = _output_processor(events=events)
    engine, core, scheduler, finish_calls, unfreeze, cleanup_dist = (
        _real_core_engine(
            monkeypatch,
            output_processor,
            events,
            drain_error=error,
        )
    )
    llm = _grllm(output_processor, engine)
    collect = mock.Mock(name="outer_gc_collect")
    release = mock.Mock(name="release_accelerator_memory")
    monkeypatch.setattr(offline_gr.gc, "collect", collect)
    monkeypatch.setattr(offline_gr, "release_accelerator_memory", release)
    states_before = dict(output_processor.request_states)

    with pytest.raises(RuntimeError, match="worker drain failed") as caught:
        llm.shutdown()

    assert caught.value is error
    assert events == ["worker_drain"]
    assert finish_calls == []
    assert "persistent" in scheduler.requests
    scheduler.connector.shutdown.assert_not_called()
    assert output_processor.request_states == states_before
    assert output_processor.abort_calls == []
    assert llm.llm_engine is engine
    assert not getattr(llm, "_engine_shutdown_complete", False)
    assert llm._shutdown_complete is False
    unfreeze.assert_not_called()
    cleanup_dist.assert_not_called()
    collect.assert_not_called()
    release.assert_not_called()
    core.model_executor.shutdown.assert_called_once_with()
