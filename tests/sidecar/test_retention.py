# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound checkpoint files while protecting live and in-progress resource paths."""

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.sidecar.test_runtime import Control, Files, config, task
from vllm_gr.sidecar import PausedUpdateProcess, RollingUpdate, Sidecar, UpdateError


def cached(root: Path, version: str, age: int) -> Path:
    path = root / version
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.bin").write_bytes(b"v0")
    os.utime(path, (age, age))
    return path


def versions(root: Path) -> set[str]:
    return {p.name for p in root.iterdir() if not p.name.startswith(".") and not p.is_symlink()}


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])  # type: ignore[misc]
def test_checkpoint_retention_requires_a_positive_integer(tmp_path: Path, limit: Any) -> None:
    configured = task(tmp_path, Files(), PausedUpdateProcess(Control()), RollingUpdate())
    with pytest.raises(ValueError, match="max_local_checkpoints must be a positive integer"):
        replace(configured, max_local_checkpoints=limit)


def test_checkpoint_retention_uses_publication_age_after_restart(tmp_path: Path) -> None:
    class PreservingFiles(Files):
        async def download(self, address: str, version: str, target_dir: Path) -> None:
            await super().download(address, version, target_dir)
            # Remote directory mtimes deliberately run backwards.
            stamp = 100 - len(self.downloads)
            os.utime(target_dir, (stamp, stamp))

    fs, control = PreservingFiles(), Control()
    configured = task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate())
    assert configured.max_local_checkpoints == 2
    cfg = config(tmp_path, [configured], 1)

    async def run() -> None:
        # Names deliberately do not sort in publication order.
        for version in ["v9", "v2", "v1"]:
            fs.latest = version
            assert await Sidecar(cfg, control=control).run_once("weights")

    asyncio.run(run())
    assert versions(configured.target_dir) == {"v2", "v1"}
    assert fs.downloads == ["v9", "v2", "v1"]
    assert control.state("0")["checkpoint_version"] == "v1"


def test_retention_runs_without_a_new_publication(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    fs.latest = None
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()), max_local_checkpoints=1
    )
    for i in range(3):
        cached(configured.target_dir, f"v{i}", i)
    control.state("0").update(
        checkpoint_version="v2", checkpoint_path=str(configured.target_dir / "v2")
    )
    assert not asyncio.run(
        Sidecar(config(tmp_path, [configured], 1), control=control).run_once("weights")
    )
    assert versions(configured.target_dir) == {"v2"}
    assert not fs.downloads and not control.events


def test_retention_preserves_current_path_until_single_copy_update_finishes(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()), max_local_checkpoints=1
    )
    old = cached(configured.target_dir, "old-files", 1)
    control.state("0").update(checkpoint_version="serving-alias", checkpoint_path=str(old))
    observed: list[Path | None] = []

    def check(old_path: Path | None, new_path: Path) -> bool:
        assert old_path == old and old.is_dir() and new_path.is_dir()
        observed.append(old_path)
        return True

    configured = replace(configured, checker=check)
    assert asyncio.run(
        Sidecar(config(tmp_path, [configured], 1), control=control).run_once("weights")
    )
    assert observed == [old]
    assert versions(configured.target_dir) == {"v1"}


def test_non_target_reference_keeps_oldest_until_a_no_change_poll(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="vllm_gr.sidecar.controller")
    fs, control = Files(), Control()
    fs.latest = "v2"
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()),
        max_local_checkpoints=1,
        target_instances=["0"],
    )
    oldest = cached(configured.target_dir, "v0", 1)
    current = cached(configured.target_dir, "v1", 2)
    control.state("0").update(checkpoint_version="v1", checkpoint_path=str(current))
    control.state("1").update(checkpoint_version="v0", checkpoint_path=str(oldest))
    cfg = config(tmp_path, [configured], 2)
    assert asyncio.run(Sidecar(cfg, control=control).run_once("weights"))
    assert versions(configured.target_dir) == {"v0", "v2"}
    assert "exceeds retention" in caplog.text
    control.state("1").update(checkpoint_version="external", checkpoint_path=control.resource_path)
    assert not asyncio.run(Sidecar(cfg, control=control).run_once("weights"))
    assert versions(configured.target_dir) == {"v2"}
    assert fs.downloads == ["v2"]


def test_partial_rollout_retains_both_versions_until_restart_reconciles(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()), max_local_checkpoints=1
    )
    old = cached(configured.target_dir, "v0", 1)
    for key in ["0", "1"]:
        control.state(key).update(checkpoint_version="v0", checkpoint_path=str(old))
    cfg = config(tmp_path, [configured], 2)
    control.fail = "reload"  # The server commits, but the reload reply is lost.
    with pytest.raises(UpdateError, match="reply lost"):
        asyncio.run(Sidecar(cfg, control=control).run_once("weights"))
    assert versions(configured.target_dir) == {"v0", "v1"}
    assert asyncio.run(Sidecar(cfg, control=control).run_once("weights"))
    assert versions(configured.target_dir) == {"v1"}
    assert fs.downloads == ["v1"]


def test_unreachable_instance_skips_cleanup_without_failing_update(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="vllm_gr.sidecar.controller")

    class Unreachable(Control):
        async def status(self, instance: Any) -> Any:
            if instance.id == "1":
                raise UpdateError("unreachable")
            return await super().status(instance)

    fs, control = Files(), Unreachable()
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()),
        max_local_checkpoints=1,
        target_instances=["0"],
    )
    for i in range(3):
        cached(configured.target_dir, f"old{i}", i + 1)
    assert asyncio.run(
        Sidecar(config(tmp_path, [configured], 2), control=control).run_once("weights")
    )
    assert versions(configured.target_dir) == {"old0", "old1", "old2", "v1"}
    assert "status unavailable" in caplog.text
    assert control.state("0")["checkpoint_version"] == "v1"


def test_invalid_checkpoint_state_keeps_uncommitted_downloads(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()), max_local_checkpoints=1
    )
    old = cached(configured.target_dir, "v0", 1)
    cached(configured.target_dir, "uncommitted", 2)
    control.state("0").update(checkpoint_valid=False, checkpoint_path=str(old))
    with pytest.raises(UpdateError, match="invalid checkpoint"):
        asyncio.run(Sidecar(config(tmp_path, [configured], 1), control=control).run_once("weights"))
    assert versions(configured.target_dir) == {"v0", "uncommitted", "v1"}
    assert not control.events


def test_cleanup_leaves_staging_and_symlinks_untouched(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()), max_local_checkpoints=1
    )
    cached(configured.target_dir, "v0", 1)
    staging = cached(configured.target_dir / ".partial", "pending", 2)
    outside = cached(tmp_path, "outside", 3)
    (configured.target_dir / "linked").symlink_to(outside, target_is_directory=True)
    assert asyncio.run(
        Sidecar(config(tmp_path, [configured], 1), control=control).run_once("weights")
    )
    assert versions(configured.target_dir) == {"v1"}
    assert staging.is_dir() and (outside / "model.bin").is_file()
    assert (configured.target_dir / "linked").is_symlink()


def test_rejected_checkpoints_do_not_accumulate(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    fs.invalid = True
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()), max_local_checkpoints=1
    )
    cfg = config(tmp_path, [configured], 1)
    for version in ["v1", "v2", "v3"]:
        fs.latest = version
        with pytest.raises(UpdateError, match="checker rejected"):
            asyncio.run(Sidecar(cfg, control=control).run_once("weights"))
        assert versions(configured.target_dir) == {version}
    assert not control.events


def test_cleanup_error_does_not_undo_a_successful_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="vllm_gr.sidecar.controller")
    from vllm_gr.sidecar import controller

    fs, control = Files(), Control()
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()), max_local_checkpoints=1
    )
    old = cached(configured.target_dir, "v0", 1)
    original = controller.shutil.rmtree

    def remove(path: Any, *args: Any, **kwargs: Any) -> None:
        if path == old:
            raise PermissionError("read-only")
        original(path, *args, **kwargs)

    monkeypatch.setattr(controller.shutil, "rmtree", remove)
    assert asyncio.run(
        Sidecar(config(tmp_path, [configured], 1), control=control).run_once("weights")
    )
    assert versions(configured.target_dir) == {"v0", "v1"}
    assert "failed to delete" in caplog.text
    assert control.state("0")["checkpoint_version"] == "v1"


def test_checkpoint_retention_does_not_apply_to_trie_tasks(tmp_path: Path) -> None:
    fs, control = Files(), Control()
    configured = replace(
        task(tmp_path, fs, PausedUpdateProcess(control), RollingUpdate()), resource_type="trie"
    )
    for i in range(3):
        cached(configured.target_dir, f"v{i}", i)
    runtime = Sidecar(config(tmp_path, [configured], 1), control=control)
    asyncio.run(runtime._cleanup_checkpoints(configured, "v2"))
    assert versions(configured.target_dir) == {"v0", "v1", "v2"}
    assert not control.checks


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
