# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reloadable constraint tables; old versions expire lazily after a fixed interval."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

INITIAL_CONSTRAINT_VERSION = "__initial__"


class VersionedConstraintTable:
    """Default accesses use the latest table. API calls release expired old tables.

    Retention starts when reload replaces a version. There is no timer thread,
    session tracking, preparation phase, or separate publication operation.
    """

    def __init__(
        self,
        table: Any,
        *,
        loader: Callable[[str], Any],
        retention_seconds: float = 60,
        before_release: Callable[[], None] | None = None,
    ) -> None:
        if not math.isfinite(retention_seconds) or retention_seconds < 0:
            raise ValueError("Trie retention_seconds must be finite and non-negative")
        self.current = INITIAL_CONSTRAINT_VERSION
        # Each backend owns its CPU table and, when applicable, device buffers.
        self.tables = {self.current: table}
        self.retention_seconds = retention_seconds
        self._loader = loader
        self._before_release = before_release
        self._retired_at: dict[str, float] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.get(), name)

    def get(self, version: str | None = None) -> Any:
        self._collect_expired()
        key = self.current if version is None else version
        return self.tables[key]

    def reload(self, path: str, version: str) -> Any:
        self._collect_expired()
        if not version.strip() or version == INITIAL_CONSTRAINT_VERSION:
            raise ValueError("Invalid or reserved Trie version")
        candidate = self._loader(path)
        try:
            current = self.get()
            if candidate.info.table_metadata.trie_count != current.info.table_metadata.trie_count:
                raise ValueError("Online Trie reload cannot change the loaded Trie count")
            if getattr(candidate, "vocab_slice", None) != getattr(current, "vocab_slice", None):
                raise ValueError("Online Trie reload cannot change the model vocabulary projection")
            if version in self.tables:
                retained = self.tables[version]
                if (
                    retained.info.table_metadata.artifact_digest
                    != candidate.info.table_metadata.artifact_digest
                ):
                    raise ValueError("A Trie version cannot be reused for different contents")
        except Exception:
            candidate.close()
            raise
        if version in self.tables:
            candidate.close()
            candidate = self.tables[version]
        if version != self.current:
            self._retired_at[self.current] = time.monotonic()
            self.tables[version] = candidate
            self.current = version
            self._retired_at.pop(version, None)
        self._collect_expired()
        return candidate

    def _collect_expired(self) -> None:
        now = time.monotonic()
        expired = [
            version
            for version, retired in self._retired_at.items()
            if version != self.current and now - retired >= self.retention_seconds
        ]
        if expired and self._before_release is not None:
            self._before_release()
        for version in expired:
            self.tables.pop(version).close()
            del self._retired_at[version]

    def close(self) -> None:
        values = list(self.tables.values())
        if values and self._before_release is not None:
            self._before_release()
        for table in values:
            table.close()
        self.tables.clear()
        self._retired_at.clear()
