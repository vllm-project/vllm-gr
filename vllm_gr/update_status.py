# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared service-state and committed-resource status contract."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

ResourceType = Literal["checkpoint", "trie"]


class ServiceState(Enum):
    RUNNING = "running"
    PAUSED = "paused"


@dataclass(frozen=True)
class ServiceStatus:
    service_state: ServiceState
    checkpoint_version: str | None
    trie_version: str | None
    checkpoint_path: str | None
    trie_path: str | None
    checkpoint_valid: bool

    def version_for(self, resource: ResourceType) -> str | None:
        return self.checkpoint_version if resource == "checkpoint" else self.trie_version

    def path_for(self, resource: ResourceType) -> str | None:
        return self.checkpoint_path if resource == "checkpoint" else self.trie_path
