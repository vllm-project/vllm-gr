# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Sidecar values and extension points, independent of the inference runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vllm_gr.update_status import ResourceType, ServiceStatus

Checker = Callable[[Path | None, Path], bool | Awaitable[bool]]


class UpdateError(RuntimeError):
    """A resource update could not complete safely."""


class FileSystem(Protocol):
    async def get_latest_version(self, address: str) -> str | None: ...

    async def download(self, address: str, version: str, target_dir: Path) -> None: ...


@dataclass(frozen=True)
class Instance:
    id: str
    control_url: str


@dataclass(frozen=True)
class UpdateContext:
    task_name: str
    resource_type: ResourceType
    version: str
    local_path: Path


class Control(Protocol):
    async def pause_and_drain(self, instance: Instance, context: UpdateContext) -> None: ...

    async def reload(self, instance: Instance, context: UpdateContext) -> None: ...

    async def resume(self, instance: Instance, context: UpdateContext) -> None: ...

    async def status(self, instance: Instance) -> ServiceStatus: ...


class UpdateMode(Protocol):
    def groups(self, instances: Sequence[Instance]) -> Iterable[Sequence[Instance]]: ...


class UpdateProcess(Protocol):
    async def run(
        self, instance: Instance, context: UpdateContext, *, checker: Checker | None = None
    ) -> bool:
        """Inspect live status; return whether the instance was changed."""
        ...


def check_name(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
        raise ValueError(f"{label} must be a nonempty directory name")
    return value
