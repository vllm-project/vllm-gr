# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NOVI-compatible ready-version discovery and pinned HDFS CLI downloads.

Reference: GR-NOVI/inference/deploy/sidecar_model_updater.py. Supports its
_SUCCESS marker, timestamp naming, client selection, CLI dialects and fast-get
arguments. P2P transport and NOVI's model-management logic are not dependencies.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from collections.abc import Callable
from pathlib import Path

import regex as re

from ..contracts import UpdateError, check_name

_FAST_CLIENTS = ("/opt/tiger/hdfs_client/bin/hdfs", "/opt/tiger/arnold/hdfs_client/hdfs")


def timestamp_version(name: str) -> int | None:
    """NOVI checkpoint suffix _<10+ digits> or Trie suffix _YYYYMMDD_HHMMSS."""
    date = re.search(r"_(\d{8})_(\d{6})$", name)
    if date:
        return int(date[1] + date[2])
    stamp = re.search(r"(?:^|_)(\d{10,})$", name)
    return int(stamp[1]) if stamp else None


class HDFSDownloader:
    def __init__(
        self,
        binary: str | None = None,
        *,
        uses_dfs_subcommand: bool | None = None,
        success_marker: str = "_SUCCESS",
        version_key: Callable[[str], int | None] = timestamp_version,
        metadata_timeout: float = 120,
        download_timeout: float = 3600,
    ) -> None:
        explicit = binary or os.environ.get("HDFS_BIN")
        resolved = explicit or next(
            (p for p in _FAST_CLIENTS if os.path.isfile(p) and os.access(p, os.X_OK)),
            shutil.which("hdfs") or "hdfs",
        )
        assert resolved is not None
        self.binary: str = resolved
        override = os.environ.get("HDFS_USES_DFS_SUBCOMMAND")
        if uses_dfs_subcommand is None:
            uses_dfs_subcommand = (
                override.lower() in {"1", "true", "yes"}
                if override is not None
                else "/opt/tiger/hdfs_client/" not in self.binary
            )
        self.uses_dfs = uses_dfs_subcommand
        self.success_marker = check_name(success_marker, "success_marker")
        self.version_key = version_key
        if metadata_timeout <= 0 or download_timeout <= 0:
            raise ValueError("HDFS timeouts must be positive")
        self.metadata_timeout = metadata_timeout
        self.download_timeout = download_timeout

    def _argv(self, command: str, *args: str) -> list[str]:
        prefix = [self.binary, "dfs", f"-{command}"] if self.uses_dfs else [self.binary, command]
        if command == "get" and os.path.normpath(self.binary) in _FAST_CLIENTS:
            prefix.extend(["-s", "-c", "128", "--ct", "32", "-t", "8"])
        return [*prefix, *args]

    async def _run(self, command: str, *args: str, missing_ok: bool = False) -> str | None:
        argv = self._argv(command, *args)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise UpdateError(f"Cannot start HDFS client {self.binary}: {exc}") from exc
        try:
            timeout = self.download_timeout if command == "get" else self.metadata_timeout
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise UpdateError(f"HDFS {command} timed out") from exc
        error = stderr.decode(errors="replace").strip()
        if process.returncode:
            # The deployed Hadoop wrapper emits this notice even for a missing
            # marker. It is not an HDFS error; retain all other diagnostics.
            diagnostic = "\n".join(
                line
                for line in error.splitlines()
                if line != "WARNING: skip config refresh: must run as tiger user"
            )
            absent = not diagnostic or any(
                text in diagnostic.lower()
                for text in ("no such file", "does not exist", "not found")
            )
            if missing_ok and process.returncode == 1 and absent:
                return None
            raise UpdateError(f"HDFS {command} failed ({process.returncode}): {error[-2000:]}")
        return stdout.decode(errors="replace")

    async def get_latest_version(self, address: str) -> str | None:
        output = await self._run("ls", address)
        candidates: list[tuple[int, str]] = []
        for line in (output or "").splitlines():
            parts = line.split()
            if len(parts) < 8 or not parts[0].startswith("d"):
                continue
            name = parts[-1].rstrip("/").rsplit("/", 1)[-1]
            stamp = self.version_key(name)
            if stamp is not None:
                candidates.append((stamp, check_name(name, "version")))
        for _, name in sorted(candidates, reverse=True):
            marker = f"{address.rstrip('/')}/{name}/{self.success_marker}"
            if await self._run("test", "-e", marker, missing_ok=True) is not None:
                return name
        return None

    async def download(self, address: str, version: str, target_dir: Path) -> None:
        check_name(version, "version")
        target_dir = Path(target_dir)
        if target_dir.exists():
            raise UpdateError(f"Download target already exists: {target_dir}")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            source = f"{address.rstrip('/')}/{version}"
            marker = f"{source}/{self.success_marker}"
            if await self._run("test", "-e", marker, missing_ok=True) is None:
                raise UpdateError(f"Version {version} is not published")
            await self._run("get", source, str(target_dir))
            if not (target_dir / self.success_marker).is_file():
                raise UpdateError(f"Downloaded version {version} lacks {self.success_marker}")
        except BaseException:
            shutil.rmtree(target_dir, ignore_errors=True)
            shutil.rmtree(Path(str(target_dir) + "._COPYING_"), ignore_errors=True)
            raise
