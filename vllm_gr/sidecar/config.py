# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Python object composition; configuration does not start background work."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from vllm_gr.update_status import ResourceType

from .contracts import (
    Checker,
    FileSystem,
    Instance,
    UpdateMode,
    UpdateProcess,
    check_name,
)
from .validation import default_checker


@dataclass(frozen=True)
class UpdateTask:
    """A resource watcher; checker(old_path, new_path) returns True or raises.

    Omit checker to compare checkpoint file names/sizes or Trie tensor names.
    Both synchronous and asynchronous checkers are supported. old_path is None
    when the service cannot report a local baseline; defaults reject that case.
    max_local_checkpoints bounds checkpoint history after updates; referenced
    and in-progress versions may temporarily exceed it. Trie files are unchanged.
    """

    name: str
    filesystem: FileSystem
    address: str
    target_dir: Path
    update_mode: UpdateMode
    update_process: UpdateProcess
    resource_type: ResourceType
    poll_interval_seconds: float = 30
    target_instances: str | tuple[str, ...] | list[str] = "all"
    checker: Checker | None = None
    max_local_checkpoints: int = 2
    """Checkpoint copies to retain, excluding temporary in-use excess versions."""

    def __post_init__(self) -> None:
        check_name(self.name, "task name")
        object.__setattr__(self, "target_dir", Path(self.target_dir).resolve())
        if self.resource_type not in {"checkpoint", "trie"}:
            raise ValueError("resource_type must be checkpoint or trie")
        if type(self.max_local_checkpoints) is not int or self.max_local_checkpoints < 1:
            raise ValueError("max_local_checkpoints must be a positive integer")
        if self.checker is None:
            object.__setattr__(self, "checker", default_checker(self.resource_type))
        elif not callable(self.checker):
            raise ValueError("checker must be callable")
        if (
            not self.address
            or not math.isfinite(self.poll_interval_seconds)
            or self.poll_interval_seconds <= 0
        ):
            raise ValueError("Task address and positive finite polling interval are required")
        if isinstance(self.target_instances, str):
            if self.target_instances != "all":
                raise ValueError("target_instances must be 'all' or a sequence of IDs")
        else:
            object.__setattr__(self, "target_instances", tuple(self.target_instances))
            if not self.target_instances or len(set(self.target_instances)) != len(
                self.target_instances
            ):
                raise ValueError("target_instances must contain distinct IDs")


@dataclass(frozen=True)
class SidecarConfig:
    instances: tuple[Instance, ...] | list[Instance]
    tasks: tuple[UpdateTask, ...] | list[UpdateTask]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        ids = {instance.id for instance in self.instances}
        if not ids or len(ids) != len(self.instances):
            raise ValueError("Instance IDs must be nonempty and unique")
        if not self.tasks or len({task.name for task in self.tasks}) != len(self.tasks):
            raise ValueError("Task names must be nonempty and unique")
        for instance in self.instances:
            check_name(instance.id, "instance ID")
            url = urlsplit(instance.control_url)
            if url.scheme not in {"http", "https"} or not url.hostname:
                raise ValueError("Instance control_url must be an HTTP(S) URL")
        roots: list[Path] = []
        for task in self.tasks:
            if task.target_instances != "all" and not set(task.target_instances) <= ids:
                raise ValueError(f"Task {task.name} references an unknown instance")
            path = task.target_dir
            if any(path.is_relative_to(other) or other.is_relative_to(path) for other in roots):
                raise ValueError("Task caches must not overlap")
            roots.append(path)
