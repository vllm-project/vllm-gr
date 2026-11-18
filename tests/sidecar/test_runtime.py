# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Version convergence, stateless restart, grouping, and resource validation."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from vllm_gr.sidecar import (
    Instance,
    OnlineUpdateProcess,
    PausedUpdateProcess,
    RollingUpdate,
    Sidecar,
    SidecarConfig,
    SyncUpdate,
    UpdateContext,
    UpdateError,
    UpdateTask,
)
from vllm_gr.sidecar.__main__ import load_config
from vllm_gr.update_status import ServiceState, ServiceStatus


class Files:
    def __init__(self) -> None:
        self.latest: str | None = "v1"
        self.downloads: list[str] = []
        self.invalid = False

    async def get_latest_version(self, address: str) -> str | None:
        return self.latest

    async def download(self, address: str, version: str, target_dir: Path) -> None:
        self.downloads.append(version)
        target_dir.mkdir()
        (target_dir / "config.json").write_text("{}")
        if not self.invalid:
            (target_dir / "model.bin").write_bytes(version.encode())


class Control:
    """The simulated model service owns versions, pause state, and loading."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []
        self.statuses: dict[str, dict[str, Any]] = {}
        self.checks: list[str] = []
        self.reloads: list[tuple[str, str]] = []
        self.paused: set[str] = set()
        self.fail: str | None = None
        self.active = self.peak = 0
        self.resource_path: str | None = None

    def state(self, instance_id: str) -> dict[str, Any]:
        return self.statuses.setdefault(
            instance_id,
            {
                "service_state": ServiceState.RUNNING,
                "checkpoint_valid": True,
                "checkpoint_version": "old",
                "trie_version": "old",
                "checkpoint_path": self.resource_path,
                "trie_path": self.resource_path,
            },
        )

    def record(self, step: str, instance: Instance, context: UpdateContext) -> None:
        self.events.append((instance.id, context.version, step))
        if self.fail == step:
            self.fail = None
            raise UpdateError("reply lost after server completed step")
        if self.fail == "cancel_reload" and step == "reload":
            self.fail = None
            raise asyncio.CancelledError()

    async def pause_and_drain(self, instance: Instance, context: UpdateContext) -> None:
        self.paused.add(instance.id)
        self.state(instance.id).update(service_state=ServiceState.PAUSED)
        self.record("pause", instance, context)

    async def reload(self, instance: Instance, context: UpdateContext) -> None:
        self.active += 1
        self.peak = max(self.active, self.peak)
        try:
            await asyncio.sleep(0.01)
            self.reloads.append((instance.id, context.version))
            self.state(instance.id)[f"{context.resource_type}_version"] = context.version
            self.state(instance.id)[f"{context.resource_type}_path"] = str(context.local_path)
            self.record("reload", instance, context)
        finally:
            self.active -= 1

    async def resume(self, instance: Instance, context: UpdateContext) -> None:
        self.paused.discard(instance.id)
        self.state(instance.id).update(service_state=ServiceState.RUNNING)
        self.record("resume", instance, context)

    async def status(self, instance: Instance) -> ServiceStatus:
        self.checks.append(instance.id)
        return ServiceStatus(**self.state(instance.id))


def task(tmp: Path, fs: Files, process: Any, mode: Any, name: str = "weights") -> UpdateTask:
    old = tmp / "initial"
    old.mkdir(exist_ok=True)
    (old / "config.json").write_text("{}")
    (old / "model.bin").write_bytes(b"v0")
    process.control.resource_path = str(old)
    for state in process.control.statuses.values():
        if state.get("checkpoint_path") is None and state.get("checkpoint_version") is not None:
            state["checkpoint_path"] = str(old)
    return UpdateTask(
        name=name,
        filesystem=fs,
        address="hdfs://models",
        target_dir=tmp / name,
        update_mode=mode,
        update_process=process,
        resource_type="checkpoint",
    )


def config(tmp: Path, tasks: list[UpdateTask], count: int = 4) -> SidecarConfig:
    return SidecarConfig(
        tasks=tasks,
        instances=[Instance(str(i), f"http://localhost:{9000 + i}") for i in range(count)],
    )


@pytest.mark.parametrize("parallel", [1, 2, 4])  # type: ignore[misc]
def test_rolling_groups_and_reload_ordering(tmp_path: Path, parallel: int) -> None:
    fs, control = Files(), Control()
    cfg = config(
        tmp_path, [task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate(parallel))]
    )

    async def run() -> None:
        runtime = Sidecar(cfg)
        assert await runtime.run_once("weights")
        assert not await runtime.run_once("weights")

    asyncio.run(run())
    assert control.peak == parallel
    assert fs.downloads == ["v1"]
    assert len(control.reloads) == 4
    for i in range(4):
        assert [e[2] for e in control.events if e[0] == str(i)] == ["pause", "reload", "resume"]
    assert not (tmp_path / "state").exists()


def test_sync_online_only_reloads(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    cfg = config(tmp_path, [task(tmp_path, fs, OnlineUpdateProcess(control), SyncUpdate())])
    assert asyncio.run(Sidecar(cfg).run_once("weights"))
    assert control.peak == 4
    assert [event[2] for event in control.events] == ["reload"] * 4


def test_restart_updates_only_mismatched_instances(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    for instance_id in ["1", "3"]:
        control.state(instance_id)["checkpoint_version"] = "v1"
    cfg = config(tmp_path, [task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate())])
    assert asyncio.run(Sidecar(cfg).run_once("weights"))
    assert {event[0] for event in control.events} == {"0", "2"}
    control.events.clear()
    assert not asyncio.run(Sidecar(cfg).run_once("weights"))
    assert control.events == []
    assert fs.downloads == ["v1"]
    assert set(control.checks) == {"0", "1", "2", "3"}


def test_instance_version_drift_is_detected_without_sidecar_restart(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    cfg = config(tmp_path, [task(tmp_path, fs, OnlineUpdateProcess(control), RollingUpdate())], 2)

    async def run() -> None:
        runtime = Sidecar(cfg)
        await runtime.run_once("weights")
        control.events.clear()
        control.state("1")["checkpoint_version"] = "old"
        assert await runtime.run_once("weights")

    asyncio.run(run())
    assert control.events == [("1", "v1", "reload")]


def test_lost_reload_reply_resumes_from_live_version_without_reloading(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    control.fail = "reload"
    cfg = config(tmp_path, [task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate())], 2)
    with pytest.raises(UpdateError):
        asyncio.run(Sidecar(cfg).run_once("weights"))
    assert control.state("0")["checkpoint_version"] == "v1"
    assert asyncio.run(Sidecar(cfg).run_once("weights"))
    assert [event[2] for event in control.events if event[0] == "0"] == [
        "pause",
        "reload",
        "resume",
    ]
    assert fs.downloads == ["v1"]


def test_restart_uses_latest_version_without_replaying_old_progress(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    control.fail = "reload"
    cfg = config(tmp_path, [task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate())], 2)
    with pytest.raises(UpdateError):
        asyncio.run(Sidecar(cfg).run_once("weights"))
    fs.latest = "v2"
    assert asyncio.run(Sidecar(cfg).run_once("weights"))
    assert all(control.state(str(i))["checkpoint_version"] == "v2" for i in range(2))
    assert fs.downloads == ["v1", "v2"]


def test_default_checker_requires_an_old_resource_path(tmp_path: Path) -> None:
    control = Control()
    control.state("0").update(checkpoint_version=None, checkpoint_path=None)
    cfg = config(
        tmp_path, [task(tmp_path, Files(), OnlineUpdateProcess(control), RollingUpdate())], 1
    )
    with pytest.raises(UpdateError, match="requires the current resource_path"):
        asyncio.run(Sidecar(cfg).run_once("weights"))
    assert not control.events


def test_paused_service_blocks_online_task_until_checkpoint_resumes(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    control.fail = "reload"
    cfg = config(
        tmp_path,
        [
            task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()),
            task(tmp_path, fs, OnlineUpdateProcess(control), SyncUpdate(), "other"),
        ],
        1,
    )
    runtime = Sidecar(cfg)
    with pytest.raises(UpdateError):
        asyncio.run(runtime.run_once("weights"))
    with pytest.raises(UpdateError, match="paused"):
        asyncio.run(runtime.run_once("other"))
    assert asyncio.run(runtime.run_once("weights"))
    assert not asyncio.run(runtime.run_once("other"))


def test_tasks_share_instance_lock_and_recheck_status(tmp_path: Path) -> None:
    fs, other, control = Files(), Files(), Control()
    other.latest = "v2"
    cfg = config(
        tmp_path,
        [
            task(tmp_path, fs, OnlineUpdateProcess(control), SyncUpdate(), "one"),
            task(tmp_path, other, OnlineUpdateProcess(control), SyncUpdate(), "two"),
        ],
        1,
    )

    async def run() -> None:
        runtime = Sidecar(cfg)
        await asyncio.gather(runtime.run_once("one"), runtime.run_once("two"))

    asyncio.run(run())
    assert control.peak == 1
    assert len(control.events) == 2


def test_invalid_download_never_calls_server(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    fs.invalid = True
    cfg = config(tmp_path, [task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate())])
    with pytest.raises(UpdateError, match="checker rejected"):
        asyncio.run(Sidecar(cfg).run_once("weights"))
    assert not control.events
    assert (tmp_path / "weights/v1").exists()
    assert not list((tmp_path / "weights/.partial").iterdir())


def test_invalid_mode_is_rejected_before_mutation(tmp_path: Path) -> None:
    class IncompleteMode:
        def groups(self, instances: Any) -> Any:
            return [instances[:1]]

    control = Control()
    cfg = config(
        tmp_path, [task(tmp_path, Files(), OnlineUpdateProcess(control), IncompleteMode())]
    )
    with pytest.raises(UpdateError, match="every target instance"):
        asyncio.run(Sidecar(cfg).run_once("weights"))
    assert not control.events


def test_cancelled_sidecar_restarts_without_persistent_progress(tmp_path: Path) -> None:
    control = Control()
    control.fail = "cancel_reload"
    cfg = config(
        tmp_path, [task(tmp_path, Files(), PausedUpdateProcess(control), RollingUpdate())], 1
    )
    with pytest.raises(UpdateError):
        asyncio.run(Sidecar(cfg).run_once("weights"))
    assert asyncio.run(Sidecar(cfg).run_once("weights"))
    assert [event[2] for event in control.events] == ["pause", "reload", "resume"]
    assert not list(tmp_path.glob("*.json"))


def test_no_publication_does_not_touch_instances(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    fs.latest = None
    cfg = config(tmp_path, [task(tmp_path, fs, OnlineUpdateProcess(control), RollingUpdate())])
    assert not asyncio.run(Sidecar(cfg).run_once("weights"))
    assert not control.events and not control.checks


def test_changed_instance_list_needs_no_local_state_migration(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    tasks = [task(tmp_path, fs, OnlineUpdateProcess(control), RollingUpdate())]
    asyncio.run(Sidecar(config(tmp_path, tasks, 1)).run_once("weights"))
    control.events.clear()
    assert asyncio.run(Sidecar(config(tmp_path, tasks, 2)).run_once("weights"))
    assert control.events == [("1", "v1", "reload")]


def test_python_config_export_and_lightweight_import(tmp_path: Path) -> None:
    source = tmp_path / "config.py"
    source.write_text(
        'from pathlib import Path\nfrom vllm_gr.sidecar import *\nconfig=SidecarConfig(instances=[Instance("one","http://localhost:1")],tasks=[UpdateTask(name="weights",filesystem=HDFSDownloader(),address="hdfs://models",target_dir=Path("/updates/weights"),resource_type="checkpoint",update_mode=RollingUpdate(),update_process=PausedUpdateProcess())])\n'
    )
    loaded = load_config(source)
    assert isinstance(loaded, SidecarConfig)
    source.write_text("config = {}")
    with pytest.raises(ValueError, match="SidecarConfig"):
        load_config(source)


@pytest.mark.parametrize("asynchronous", [False, True])  # type: ignore[misc]
def test_task_custom_checker_receives_old_and_new_paths(tmp_path: Path, asynchronous: bool) -> None:
    from dataclasses import replace

    fs, control = Files(), Control()
    fs.invalid = True  # A custom checker replaces the default file-list comparison.
    calls: list[tuple[Path | None, Path]] = []

    def check(old_path: Path | None, new_path: Path) -> bool:
        calls.append((old_path, new_path))
        return True

    async def async_check(old_path: Path | None, new_path: Path) -> bool:
        return check(old_path, new_path)

    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()),
        checker=async_check if asynchronous else check,
    )
    assert asyncio.run(Sidecar(config(tmp_path, [configured], 1)).run_once("weights"))
    assert calls == [(tmp_path / "initial", tmp_path / "weights/v1")]
    assert [event[2] for event in control.events] == ["pause", "reload", "resume"]


def test_custom_checker_failure_never_pauses_or_reloads(tmp_path: Path) -> None:
    from dataclasses import replace

    control = Control()
    configured = replace(
        task(tmp_path, Files(), PausedUpdateProcess(control), RollingUpdate()),
        checker=lambda old, new: False,
    )
    with pytest.raises(UpdateError, match="checker rejected"):
        asyncio.run(Sidecar(config(tmp_path, [configured], 1)).run_once("weights"))
    assert not control.events


@pytest.mark.parametrize("version", ["old", "v1"])  # type: ignore[misc]
def test_online_update_rejects_paused_service(tmp_path: Path, version: str) -> None:
    control = Control()
    control.state("0").update(service_state=ServiceState.PAUSED, checkpoint_version=version)
    cfg = config(
        tmp_path, [task(tmp_path, Files(), OnlineUpdateProcess(control), RollingUpdate())], 1
    )
    with pytest.raises(UpdateError, match="paused"):
        asyncio.run(Sidecar(cfg).run_once("weights"))
    assert not control.events


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
