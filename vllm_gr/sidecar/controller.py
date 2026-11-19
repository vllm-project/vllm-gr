# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Stateless Task polling, resource staging, and bounded instance updates."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from .client import ControlClient
from .config import SidecarConfig, UpdateTask
from .contracts import Control, Instance, UpdateContext, UpdateError, check_name
from .validation import trie_file

logger = logging.getLogger(__name__)


class Sidecar:
    def __init__(self, config: SidecarConfig, *, control: Control | None = None) -> None:
        self.config = config
        self.instances = {instance.id: instance for instance in config.instances}
        self.tasks = {task.name: task for task in config.tasks}
        self.locks = {key: asyncio.Lock() for key in self.instances}
        self.task_locks = {key: asyncio.Lock() for key in self.tasks}
        self.control = control if control is not None else ControlClient()

    def _targets(self, task: UpdateTask) -> list[Instance]:
        keys = self.instances if task.target_instances == "all" else task.target_instances
        return [self.instances[key] for key in keys]

    async def _prepare(self, task: UpdateTask, version: str) -> Path:
        check_name(version, "version")
        root = task.target_dir
        root.mkdir(parents=True, exist_ok=True)
        published = root / version
        if published.is_symlink():
            raise UpdateError(f"Published resource must not be a symlink: {published}")
        if published.exists():
            path = trie_file(published) if task.resource_type == "trie" else published
            if not path.resolve().is_relative_to(published.resolve()):
                raise UpdateError("Resource path is outside the download directory")
            return path
        staging = root / ".partial" / uuid.uuid4().hex
        staging.parent.mkdir(exist_ok=True)
        try:
            await task.filesystem.download(task.address, version, staging)
            path = trie_file(staging) if task.resource_type == "trie" else staging
            try:
                relative = path.resolve().relative_to(staging.resolve())
            except ValueError as exc:
                raise UpdateError("Resource path is outside the download directory") from exc
            # Keep local publication order even when the downloader preserves mtimes.
            os.utime(staging, None)
            os.rename(staging, published)
            return published / relative
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    async def _apply(self, task: UpdateTask, instance: Instance, context: UpdateContext) -> bool:
        # Query status inside the shared instance lock, including across Tasks.
        async with self.locks[instance.id]:
            return await task.update_process.run(instance, context, checker=task.checker)

    async def _cleanup_checkpoints(self, task: UpdateTask, version: str | None) -> None:
        """Remove oldest unused copies while the task lock pins this update."""
        if task.resource_type != "checkpoint" or not task.target_dir.exists():
            return
        versions = sorted(
            (
                path
                for path in task.target_dir.iterdir()
                if not path.name.startswith(".") and not path.is_symlink() and path.is_dir()
            ),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        excess = len(versions) - task.max_local_checkpoints
        if excess <= 0:
            return
        # Share the update locks so a reported resource cannot change while we
        # remove files. Include non-target instances that may still use this cache.
        async with AsyncExitStack() as stack:
            for key in sorted(self.locks):
                await stack.enter_async_context(self.locks[key])
            try:
                statuses = await asyncio.gather(
                    *(self.control.status(instance) for instance in self.instances.values())
                )
            except Exception:
                logger.warning("Task %s cache cleanup skipped: status unavailable", task.name)
                return
            if any(
                not status.checkpoint_valid or status.checkpoint_path is None for status in statuses
            ):
                # A native load may still be running after a lost reply/restart;
                # its uncommitted target path is not exposed by service status.
                logger.warning("Task %s cache cleanup skipped: checkpoint state unknown", task.name)
                return
            protected = {version} if version is not None else set()
            root = task.target_dir.resolve()
            for status in statuses:
                if status.checkpoint_version is not None:
                    protected.add(status.checkpoint_version)
                if status.checkpoint_path is not None:
                    path = Path(status.checkpoint_path).resolve()
                    if path == root:
                        return
                    if path.is_relative_to(root):
                        protected.add(path.relative_to(root).parts[0])
            for path in versions:
                if excess <= 0:
                    break
                if path.name in protected:
                    continue
                try:
                    await asyncio.to_thread(shutil.rmtree, path)
                    excess -= 1
                except OSError:
                    logger.warning("Task %s failed to delete checkpoint %s", task.name, path)
            if excess:
                logger.warning(
                    "Task %s checkpoint cache exceeds retention by %d", task.name, excess
                )

    async def run_once(self, task_name: str) -> bool:
        """Recheck live instance versions every poll, including after a restart."""
        async with self.task_locks[task_name]:
            task = self.tasks[task_name]
            targets = self._targets(task)
            groups = [list(group) for group in task.update_mode.groups(targets)]
            seen: set[str] = set()
            for group in groups:
                keys = [instance.id for instance in group]
                if (
                    not keys
                    or len(set(keys)) != len(keys)
                    or seen.intersection(keys)
                    or any(instance not in targets for instance in group)
                ):
                    raise UpdateError("UpdateMode returned invalid or duplicate targets")
                seen.update(keys)
            if seen != {instance.id for instance in targets}:
                raise UpdateError("UpdateMode did not schedule every target instance")
            version = await task.filesystem.get_latest_version(task.address)
            await self._cleanup_checkpoints(task, version)
            if version is None:
                return False
            try:
                path = await self._prepare(task, version)
                context = UpdateContext(task.name, task.resource_type, version, path)
                changed = False
                for group in groups:
                    results = await asyncio.gather(
                        *(self._apply(task, instance, context) for instance in group),
                        return_exceptions=True,
                    )
                    failures = [result for result in results if isinstance(result, BaseException)]
                    if failures:
                        raise UpdateError(f"Task {task_name} stopped: {failures[0]}") from failures[
                            0
                        ]
                    changed |= any(results)
                return changed
            finally:
                await self._cleanup_checkpoints(task, version)

    async def _watch(self, task: UpdateTask) -> None:
        while True:
            try:
                await self.run_once(task.name)
            except Exception:
                logger.exception("Task %s update failed", task.name)
            await asyncio.sleep(task.poll_interval_seconds)

    async def run(self) -> None:
        async with asyncio.TaskGroup() as group:
            for task in self.config.tasks:
                group.create_task(self._watch(task), name=f"sidecar-{task.name}")
