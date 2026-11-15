# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-call"
"""Production control routes and Core publication/failure invariants."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from vllm_gr.config import BeamExecutionMode
from vllm_gr.constraints.versioned_table import (
    INITIAL_CONSTRAINT_VERSION,
    VersionedConstraintTable,
)
from vllm_gr.entrypoints.management import control_app
from vllm_gr.update_status import ServiceState, ServiceStatus
from vllm_gr.v1.engine.hot_update import (
    pause,
    reload_trie,
    resume,
    service_status,
)


class Core:
    def __init__(self) -> None:
        self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="/models/v1"))
        self.vllm_config._vllm_gr_parsed_config = SimpleNamespace(
            beam_execution_mode=BeamExecutionMode.V1,
            constraint_table=SimpleNamespace(
                enabled=True, path="/tries/v1/table", backend=SimpleNamespace(value="cuda")
            ),
        )
        self.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True)
        self.scheduler = SimpleNamespace(running=[], _gr_v1_sessions={})
        self.batch_queue: list[Any] = []
        self.paused = False
        self.worker_failure: str | None = None
        self.calls: list[tuple[str, Any]] = []
        self.model_executor = SimpleNamespace(collective_rpc=self.rpc)
        self.reset_prefix_cache = Mock(return_value=True)
        self.set_weight_version = Mock()

    def rpc(self, method: str, args: tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        self.calls.append((method, args if args else kwargs))
        if self.worker_failure == "worker":
            raise RuntimeError("load failed")
        if method == "reload_weights":
            return [None, None]
        return [{"version": args[-1], "ready": self.worker_failure != "ack"}] * 2

    def is_scheduler_paused(self) -> bool:
        return self.paused

    def pause_scheduler(self, **kwargs: Any) -> None:
        assert kwargs == {"mode": "wait", "clear_cache": True}
        self.paused = True

    def resume_scheduler(self) -> None:
        self.paused = False


@pytest.mark.l1
def test_weights_endpoint_calls_original_rpc_without_core_reload_wrapper() -> None:
    async def run() -> None:
        core = AsyncMock()
        core.call_utility_async.return_value = ServiceStatus(
            ServiceState.PAUSED, "v1", None, "/models/v1", None, True
        )
        engine = SimpleNamespace(engine_core=core, collective_rpc=AsyncMock())
        app = control_app(engine)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            reply = await client.post(
                "/reload_weights", json={"weights_path": "/v2", "version": "v2"}
            )
            assert reply.status_code == 200
        engine.collective_rpc.assert_awaited_once_with(
            "reload_weights", kwargs={"weights_path": "/v2"}
        )
        core.call_utility_async.assert_awaited_once_with("service_status")
        assert app.state.gr_checkpoint == ("v2", "/v2")

    asyncio.run(run())


@pytest.mark.l1
def test_failed_native_load_does_not_record_success() -> None:
    async def run() -> None:
        engine = SimpleNamespace(
            engine_core=SimpleNamespace(
                call_utility_async=AsyncMock(
                    return_value=ServiceStatus(
                        ServiceState.PAUSED, "v1", None, "/models/v1", None, True
                    )
                )
            ),
            collective_rpc=AsyncMock(side_effect=RuntimeError("load failed")),
        )
        app = control_app(engine)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/reload_weights", json={"weights_path": "/v2", "version": "v2"}
            )
            assert response.status_code == 500
            assert "load failed" in response.json()["detail"]
        assert not hasattr(app.state, "gr_checkpoint")

    asyncio.run(run())


@pytest.mark.l1
def test_trie_has_only_one_worker_reload_operation() -> None:
    core = Core()
    assert reload_trie(core, "/trie/v2", "v2")["ok"]
    assert core.calls == [("reload_trie", ("/trie/v2", "v2"))]
    assert not core.paused
    assert service_status(core).trie_path == "/trie/v2"


@pytest.mark.l1
def test_trie_worker_failure_does_not_report_a_new_version() -> None:
    core = Core()
    core.worker_failure = "worker"
    assert not reload_trie(core, "/trie/v2", "v2")["ok"]
    assert core.calls == [("reload_trie", ("/trie/v2", "v2"))]
    assert service_status(core).trie_version == "v1"


def _backend() -> Any:
    return Mock(
        info=SimpleNamespace(table_metadata=SimpleNamespace(trie_count=1, artifact_digest="same")),
        vocab_slice=None,
    )


@pytest.mark.l1
def test_versioned_table_reload_switches_without_prepare_or_publish() -> None:
    old, new = _backend(), _backend()
    table = VersionedConstraintTable(old, loader=lambda _: new, retention_seconds=60)
    assert table.reload("new", "v2") is new
    assert table.get() is new
    assert table.get(INITIAL_CONSTRAINT_VERSION) is old
    old.close.assert_not_called()
    table.close()
    old.close.assert_called_once()
    new.close.assert_called_once()


@pytest.mark.l1
def test_production_routes_forward_arguments_and_map_errors() -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    async def utility(method: str, *args: Any) -> Any:
        calls.append((method, args))
        if method == "service_status":
            return ServiceStatus(ServiceState.PAUSED, "v1", "t1", "/models/v1", "/tries/t1", True)
        if args and args[-1] == "busy":
            return {"ok": False, "error": "busy: updating", "error_type": "busy"}
        return {"ok": True}

    async def run() -> None:
        app = control_app(
            SimpleNamespace(
                engine_core=SimpleNamespace(call_utility_async=utility), collective_rpc=AsyncMock()
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/service/status")).json() == {
                "service_state": "paused",
                "checkpoint_valid": True,
                "checkpoint_version": "v1",
                "trie_version": "t1",
                "checkpoint_path": "/models/v1",
                "trie_path": "/tries/t1",
            }
            assert (
                await client.post("/pause?mode=wait", json={"version": "v2"})
            ).status_code == 200
            assert (
                await client.post("/reload_weights", json={"version": "v2", "weights_path": "/v2"})
            ).status_code == 200
            assert (
                await client.post("/reload_trie", json={"version": "v2", "trie_path": "/t"})
            ).status_code == 200
            assert (await client.post("/resume", json={"version": "v2"})).status_code == 200
            assert (
                await client.post("/reload_trie", json={"version": "busy", "trie_path": "/v2"})
            ).status_code == 409
            assert (await client.post("/reload_weights", json={"version": "v2"})).status_code == 422
        assert calls == [
            ("service_status", ()),
            ("pause", ("checkpoint",)),
            ("service_status", ()),
            ("reload_trie", ("/t", "v2")),
            ("resume", ()),
            ("reload_trie", ("/v2", "busy")),
        ]

    asyncio.run(run())


@pytest.mark.l1
def test_runner_keeps_the_same_wrapper_across_reload() -> None:
    from vllm_gr.v1.worker import gpu_beam_stage_runner as stages
    from vllm_gr.v1.worker.gpu_beam_stage_runner_v1 import CudaStageHooksV1

    old, new = _backend(), _backend()
    table = VersionedConstraintTable(old, loader=lambda _: new)
    table.reload("new", "v2")
    owner = SimpleNamespace(
        closed=False,
        backend=table,
        hooks=CudaStageHooksV1(SimpleNamespace()),
    )
    runner = SimpleNamespace(constraint_backend=table, _gr_gpu_stage_runner=owner)
    assert stages.get_stage_runner(runner, object(), hooks_type=CudaStageHooksV1) is owner
    assert owner.backend is runner.constraint_backend
    assert owner.backend.info is new.info
    assert not hasattr(runner, "_gr_versioned_constraint_table")


@pytest.mark.l2
@pytest.mark.parametrize("drain", ["idle", "pending", "failed"])
def test_sidecar_calls_production_admin_listener(tmp_path: Any, drain: str) -> None:
    import socket

    from vllm_gr.entrypoints.management import AdminServer
    from vllm_gr.sidecar import (
        ControlClient,
        Instance,
        PausedUpdateProcess,
        UpdateContext,
        UpdateError,
    )
    from vllm_gr.v1.engine.hot_update import (
        pause,
        service_status,
    )

    core = Core()
    functions = {
        "pause": pause,
        "resume": resume,
        "service_status": service_status,
    }

    async def utility(method: str, *args: Any) -> Any:
        result = functions[method](core, *args)
        return await asyncio.wrap_future(result) if isinstance(result, Future) else result

    async def run() -> None:
        started = asyncio.Event()
        draining: Future[Any] = Future()
        if drain != "idle":

            def pause_scheduler(**kwargs: Any) -> Future[Any]:
                assert kwargs == {"mode": "wait", "clear_cache": True}
                core.paused = True
                started.set()
                return draining

            setattr(core, "pause_scheduler", pause_scheduler)

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]

        async def native_rpc(method: str, **kwargs: Any) -> Any:
            return core.rpc(method, **kwargs)

        engine = SimpleNamespace(
            engine_core=SimpleNamespace(call_utility_async=utility), collective_rpc=native_rpc
        )
        listener = AdminServer(engine, port)
        await listener.start()
        try:
            process = PausedUpdateProcess(ControlClient())
            instance = Instance("worker", f"http://127.0.0.1:{port}")
            context = UpdateContext("weights", "checkpoint", "v2", tmp_path)
            update = asyncio.create_task(
                process.run(instance, context, checker=lambda old, new: True)
            )
            if drain != "idle":
                await asyncio.wait_for(started.wait(), timeout=5)
                assert not update.done()
                assert not core.calls
                assert core.paused
                if drain == "failed":
                    draining.set_exception(RuntimeError("drain failed"))
                    with pytest.raises(UpdateError, match="HTTP Error 503"):
                        await asyncio.wait_for(update, timeout=5)
                    assert not core.calls
                    assert core.paused
                    assert not hasattr(listener.server.config.app.state, "gr_checkpoint")
                    return
                draining.set_result(None)
            assert await asyncio.wait_for(update, timeout=5)
            assert listener.server.config.app.state.gr_checkpoint == ("v2", str(tmp_path))
            assert service_status(core).service_state is ServiceState.RUNNING
            assert not await process.run(instance, context, checker=lambda old, new: True)
            assert len(core.calls) == 1
        finally:
            await listener.stop()
        assert listener.task is not None and listener.task.done()

    asyncio.run(run())


@pytest.mark.l1
def test_pause_returns_upstream_drain_future_without_custom_polling() -> None:
    core = Core()
    draining: Future[Any] = Future()
    setattr(core, "pause_scheduler", lambda **_: draining)
    assert pause(core) is draining
    assert not draining.done()
    draining.set_result(None)
    assert draining.result() is None
    assert not core.calls


@pytest.mark.l1
@pytest.mark.parametrize("unsupported", ["disabled", "npu", "sync"])
def test_trie_rejects_unsupported_execution_before_worker_rpc(unsupported: str) -> None:
    core = Core()
    config = core.vllm_config._vllm_gr_parsed_config
    if unsupported == "disabled":
        config.constraint_table.enabled = False
    elif unsupported == "npu":
        config.constraint_table.backend.value = "npu"
    else:
        core.vllm_config.scheduler_config.async_scheduling = False
    result = reload_trie(core, "/v2", "v2")
    assert not result["ok"] and result["error_type"] == "invalid"
    assert not core.calls


@pytest.mark.l1
def test_versioned_table_lazy_expiry_and_default_access(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.constraints import versioned_table

    now = [100.0]
    monkeypatch.setattr(versioned_table, "time", SimpleNamespace(monotonic=lambda: now[0]))
    old, new = _backend(), _backend()
    release = Mock()
    table = VersionedConstraintTable(
        old, loader=lambda _: new, retention_seconds=10, before_release=release
    )
    table.reload("new", "v2")
    assert table.info is new.info
    now[0] = 109.0
    assert table.get(INITIAL_CONSTRAINT_VERSION) is old
    now[0] = 110.0
    assert table.get() is new
    old.close.assert_called_once()
    release.assert_called_once()
    assert INITIAL_CONSTRAINT_VERSION not in table.tables


@pytest.mark.l1
def test_expired_table_is_released_by_api_without_session_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.constraints import versioned_table

    now = [0.0]
    monkeypatch.setattr(versioned_table, "time", SimpleNamespace(monotonic=lambda: now[0]))
    old, new = _backend(), _backend()
    table = VersionedConstraintTable(old, loader=lambda _: new, retention_seconds=5)
    table.reload("new", "v2")
    now[0] = 10.0
    assert table.get() is new
    old.close.assert_called_once()
    with pytest.raises(KeyError):
        table.get(INITIAL_CONSTRAINT_VERSION)


@pytest.mark.l1
@pytest.mark.parametrize("cleanup", ["expiry", "close"])
def test_versioned_table_failed_close_is_retryable_and_other_versions_are_released(
    monkeypatch: pytest.MonkeyPatch, cleanup: str
) -> None:
    from vllm_gr.constraints import versioned_table

    now = [0.0]
    monkeypatch.setattr(versioned_table, "time", SimpleNamespace(monotonic=lambda: now[0]))
    old, middle, current = _backend(), _backend(), _backend()
    backends = iter((middle, current))
    release = Mock()
    table = VersionedConstraintTable(
        old, loader=lambda _: next(backends), retention_seconds=10, before_release=release
    )
    table.reload("middle", "v2")
    table.reload("current", "v3")
    failure = RuntimeError("old table close failed")
    old.close.side_effect = [failure, None]
    now[0] = 20.0
    collect = table.get if cleanup == "expiry" else table.close

    with pytest.raises(RuntimeError) as error:
        collect()
    assert error.value is failure
    assert table.tables[INITIAL_CONSTRAINT_VERSION] is old
    assert INITIAL_CONSTRAINT_VERSION in table._retired_at
    assert "v2" not in table.tables and "v2" not in table._retired_at
    middle.close.assert_called_once()
    release.assert_called_once()

    if cleanup == "expiry":
        current.close.assert_not_called()
        assert collect() is current
        assert table.tables == {"v3": current}
    else:
        current.close.assert_called_once()
        collect()
        assert not table.tables
        current.close.assert_called_once()
        collect()  # Successful repeated close does not synchronize or release again.
    assert old.close.call_count == 2
    assert release.call_count == 2
    assert not table._retired_at


@pytest.mark.l1
@pytest.mark.parametrize("cleanup", ["expiry", "close"])
def test_versioned_table_sync_failure_preserves_all_versions_for_retry(
    monkeypatch: pytest.MonkeyPatch, cleanup: str
) -> None:
    from vllm_gr.constraints import versioned_table

    now = [0.0]
    monkeypatch.setattr(versioned_table, "time", SimpleNamespace(monotonic=lambda: now[0]))
    old, new = _backend(), _backend()
    release = Mock(side_effect=[RuntimeError("sync failed"), None])
    table = VersionedConstraintTable(
        old, loader=lambda _: new, retention_seconds=10, before_release=release
    )
    table.reload("new", "v2")
    versions, retired = dict(table.tables), dict(table._retired_at)
    now[0] = 20.0
    collect = table.get if cleanup == "expiry" else table.close

    with pytest.raises(RuntimeError, match="sync failed"):
        collect()
    assert table.tables == versions and table._retired_at == retired
    old.close.assert_not_called()
    new.close.assert_not_called()

    collect()
    old.close.assert_called_once()
    assert not table._retired_at
    if cleanup == "expiry":
        assert table.get() is new
        new.close.assert_not_called()
    else:
        assert not table.tables
        new.close.assert_called_once()


@pytest.mark.l1
def test_versioned_table_multiple_close_failures_are_reported_and_retried() -> None:
    old, new = _backend(), _backend()
    table = VersionedConstraintTable(old, loader=lambda _: new)
    table.reload("new", "v2")
    failures = [RuntimeError("old failed"), ValueError("new failed")]
    old.close.side_effect = [failures[0], None]
    new.close.side_effect = [failures[1], None]

    with pytest.raises(ExceptionGroup) as error:
        table.close()
    assert error.value.exceptions == tuple(failures)
    assert len(table.tables) == 2

    table.close()
    assert old.close.call_count == new.close.call_count == 2
    assert not table.tables and not table._retired_at


@pytest.mark.l1
def test_status_has_one_enum_state_and_both_resource_versions() -> None:
    core = Core()
    current = service_status(core)
    assert current == ServiceStatus(
        ServiceState.RUNNING, "v1", "v1", "/models/v1", "/tries/v1/table", True
    )
    core.paused = True
    assert service_status(core).service_state is ServiceState.PAUSED
    core.vllm_config._vllm_gr_parsed_config.constraint_table = None
    delattr(core, "_gr_serving_resources")
    current = service_status(core)
    assert current.service_state is ServiceState.PAUSED
    assert current.trie_version is None and current.trie_path is None


@pytest.mark.l1
def test_status_enum_restored_at_engine_core_ipc_boundary() -> None:
    from vllm.v1.engine import EngineCoreOutputs, UtilityOutput
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    value = ServiceStatus(ServiceState.PAUSED, "a", "t", "/a", "/t", True)
    payload = EngineCoreOutputs(
        utility_output=UtilityOutput(call_id=1, result=UtilityResult(value))
    )
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(MsgpackEncoder().encode(payload))
    from pydantic import TypeAdapter

    result = TypeAdapter(ServiceStatus).validate_python(decoded.utility_output.result.result)
    assert result == value
    assert result.service_state is ServiceState.PAUSED


@pytest.mark.l1
@pytest.mark.parametrize("fail", [False, True])
def test_status_publishes_checkpoint_only_after_successful_reload(fail: bool) -> None:
    async def run() -> None:
        started, finish = asyncio.Event(), asyncio.Event()

        async def native_reload(*args: Any, **kwargs: Any) -> None:
            started.set()
            await finish.wait()
            if fail:
                raise RuntimeError("load failed")

        async def utility(method: str, *args: Any) -> ServiceStatus:
            assert method == "service_status" and not args
            return ServiceStatus(
                ServiceState.PAUSED, "initial", "trie-new", "/initial", "/trie-new", True
            )

        app = control_app(
            SimpleNamespace(
                engine_core=SimpleNamespace(call_utility_async=utility),
                collective_rpc=native_reload,
            )
        )
        app.state.gr_checkpoint = ("old", "/old")
        assert not app.openapi()["paths"]["/service/status"]["get"].get("parameters")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(
                client.post("/reload_weights", json={"version": "new", "weights_path": "/new"})
            )
            await asyncio.wait_for(started.wait(), 1)
            pending = (await client.get("/service/status")).json()
            assert pending == {
                "service_state": "paused",
                "checkpoint_valid": False,
                "checkpoint_version": "old",
                "trie_version": "trie-new",
                "checkpoint_path": "/old",
                "trie_path": "/trie-new",
            }
            finish.set()
            reply = await request
            assert reply.status_code == (500 if fail else 200)
            current = (await client.get("/service/status")).json()
            assert current["checkpoint_valid"] is not fail
            assert current["checkpoint_version"] == ("old" if fail else "new")
            assert current["checkpoint_path"] == ("/old" if fail else "/new")
            assert current["service_state"] == "paused" and current["trie_version"] == "trie-new"

    asyncio.run(run())


@pytest.mark.l1
@pytest.mark.parametrize("cancel", [False, True])
def test_reload_in_progress_rejects_resume_and_concurrent_reload(cancel: bool) -> None:
    async def run() -> None:
        entered, finish = asyncio.Event(), asyncio.Event()

        async def reload(*args: Any, **kwargs: Any) -> None:
            entered.set()
            await finish.wait()

        core = Core()
        core.paused = True

        async def utility(method: str, *args: Any) -> Any:
            return service_status(core)

        app = control_app(
            SimpleNamespace(
                engine_core=SimpleNamespace(call_utility_async=utility),
                collective_rpc=reload,
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            payload = {"version": "B", "weights_path": "/B"}
            pending = asyncio.create_task(client.post("/reload_weights", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            assert (await client.post("/resume")).status_code == 409
            assert (await client.post("/reload_weights", json=payload)).status_code == 409
            if cancel:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert not app.state.gr_checkpoint_valid
                assert (await client.post("/resume")).status_code == 409
            else:
                finish.set()
                assert (await pending).status_code == 200
                assert app.state.gr_checkpoint_valid

    asyncio.run(run())


@pytest.mark.l1
def test_running_service_cannot_reload_checkpoint() -> None:
    async def run() -> None:
        core = Core()

        async def utility(method: str, *args: Any) -> Any:
            return service_status(core)

        engine = SimpleNamespace(
            engine_core=SimpleNamespace(call_utility_async=utility), collective_rpc=AsyncMock()
        )
        app = control_app(engine)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (
                await client.post("/reload_weights", json={"version": "B", "weights_path": "/B"})
            ).status_code == 409
        engine.collective_rpc.assert_not_awaited()
        assert app.state.gr_checkpoint_valid

    asyncio.run(run())


@pytest.mark.l1
def test_partial_native_reload_remains_invalid_after_sidecar_restart_and_withdrawal() -> None:
    from pathlib import Path

    import torch
    from pydantic import TypeAdapter
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_gr.sidecar import (
        Instance,
        OnlineUpdateProcess,
        PausedUpdateProcess,
        UpdateContext,
        UpdateError,
    )

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(2, 2, bias=False)
            self.second = torch.nn.Linear(2, 2, bias=False)

        def load_weights(self, iterator: Any) -> set[str]:
            loaded = set()
            for name, value in iterator:
                param = self.get_parameter(name)
                param.weight_loader(param, value)
                loaded.add(name)
            return loaded

    async def run() -> None:
        model = Model()
        record_metadata_for_reloading(model)
        runner = SimpleNamespace(
            get_model=lambda: model,
            lora_config=None,
            model_config=SimpleNamespace(dtype=torch.float32, quantization=None),
        )
        core = Core()
        native_calls: list[str] = []

        async def utility(method: str, *args: Any) -> Any:
            return {"service_status": service_status, "pause": pause, "resume": resume}[method](
                core, *args
            )

        def broken_weights() -> Any:
            yield "first.weight", torch.full((2, 2), 11.0)
            raise OSError("interrupted after first layer")

        async def rpc(method: str, kwargs: Any) -> None:
            native_calls.append(kwargs["weights_path"])
            GPUModelRunner.reload_weights(runner, weights_iterator=broken_weights())

        app = control_app(
            SimpleNamespace(
                engine_core=SimpleNamespace(call_utility_async=utility), collective_rpc=rpc
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:

            class Control:
                async def status(self, instance: Any) -> ServiceStatus:
                    reply = await client.get("/service/status")
                    reply.raise_for_status()
                    return TypeAdapter(ServiceStatus).validate_python(reply.json())

                async def post(self, path: str, body: Any) -> None:
                    reply = await client.post(path, json=body)
                    if reply.status_code >= 400:
                        raise UpdateError(reply.text)

                async def pause_and_drain(self, instance: Any, context: Any) -> None:
                    await self.post("/pause", {})

                async def reload(self, instance: Any, context: Any) -> None:
                    await self.post(
                        "/reload_weights",
                        {"version": context.version, "weights_path": str(context.local_path)},
                    )

                async def resume(self, instance: Any, context: Any) -> None:
                    await self.post("/resume", {})

            instance = Instance("node", "http://test")
            control = Control()
            with pytest.raises(UpdateError, match="interrupted"):
                await PausedUpdateProcess(control).run(
                    instance,
                    UpdateContext("ckpt", "checkpoint", "B", Path("/B")),
                    checker=lambda old, new: True,
                )
            assert torch.all(model.first.weight == 11) and model.second.weight.is_meta
            status = await control.status(instance)
            assert status.service_state is ServiceState.PAUSED
            assert status.checkpoint_version == "v1" and not status.checkpoint_valid
            # Fresh process objects model a Sidecar restart. Even withdrawing B
            # back to the last committed version must not resume corrupted A.
            for version in ("v1", "C"):
                with pytest.raises(UpdateError, match="manual recovery"):
                    await PausedUpdateProcess(Control()).run(
                        instance, UpdateContext("ckpt", "checkpoint", version, Path("/" + version))
                    )
            with pytest.raises(UpdateError, match="manual recovery"):
                await OnlineUpdateProcess(Control()).run(
                    instance, UpdateContext("trie", "trie", "new", Path("/trie"))
                )
            assert (await client.post("/resume")).status_code == 409
            assert (
                await client.post("/reload_weights", json={"version": "C", "weights_path": "/C"})
            ).status_code == 409
            assert native_calls == ["/B"] and core.paused
            assert not core.calls  # No other Worker operation was admitted.

    asyncio.run(run())


pytestmark = pytest.mark.cpu_test
