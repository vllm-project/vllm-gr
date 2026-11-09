# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure instance grouping; no transport or resource-loading behavior."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .contracts import Instance


@dataclass(frozen=True)
class RollingUpdate:
    max_concurrent_updates: int = 1

    def __post_init__(self) -> None:
        if type(self.max_concurrent_updates) is not int or self.max_concurrent_updates < 1:
            raise ValueError("max_concurrent_updates must be a positive integer")

    def groups(self, instances: Sequence[Instance]) -> Iterable[Sequence[Instance]]:
        for start in range(0, len(instances), self.max_concurrent_updates):
            yield instances[start : start + self.max_concurrent_updates]


@dataclass(frozen=True)
class SyncUpdate:
    def groups(self, instances: Sequence[Instance]) -> Iterable[Sequence[Instance]]:
        if instances:
            yield instances
