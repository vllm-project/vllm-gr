# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual CLI subprocesses, marker discovery and pinned downloads."""

import asyncio
import json
from pathlib import Path

import pytest

from vllm_gr.sidecar import HDFSDownloader, UpdateError
from vllm_gr.sidecar.filesystems.hdfs import timestamp_version


def ready(root: Path, name: str) -> None:
    path = root / "models" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "_SUCCESS").touch()
    (path / "value").write_text(name)


@pytest.mark.parametrize("dfs", [True, False])  # type: ignore[misc]
def test_pinned_download_and_ready_discovery(
    hdfs: tuple[HDFSDownloader, Path, Path], tmp_path: Path, dfs: bool
) -> None:
    fs, remote, log = hdfs
    fs.uses_dfs = dfs
    ready(remote, "iter1_label_1700000000000")
    (remote / "models/iter2_label_1800000000000").mkdir()
    (remote / "models/config.json").touch()

    async def run() -> None:
        selected = await fs.get_latest_version("hdfs://cluster/models")
        assert selected == "iter1_label_1700000000000"
        ready(remote, "iter2_label_1800000000000")
        target = tmp_path / "download"
        await fs.download("hdfs://cluster/models", selected, target)
        assert (target / "value").read_text() == selected
        assert await fs.get_latest_version("hdfs://cluster/models") == "iter2_label_1800000000000"
        with pytest.raises(UpdateError, match="already exists"):
            await fs.download("hdfs://cluster/models", selected, target)

    asyncio.run(run())
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert all((call[0] == "dfs") is dfs for call in calls)
    get = next(call for call in calls if "get" in call or "-get" in call)
    assert get[-2] == "hdfs://cluster/models/iter1_label_1700000000000"


def test_metadata_errors_are_not_no_update(
    hdfs: tuple[HDFSDownloader, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, remote, _ = hdfs
    (remote / "models").mkdir()
    assert asyncio.run(fs.get_latest_version("hdfs://cluster/models")) is None
    monkeypatch.setenv("HDFS_TEST_DENIED", "1")
    with pytest.raises(UpdateError, match="Permission denied"):
        asyncio.run(fs.get_latest_version("hdfs://cluster/models"))


def test_timeout_terminates_cli(
    hdfs: tuple[HDFSDownloader, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, _, _ = hdfs
    fs.metadata_timeout = 0.05
    monkeypatch.setenv("HDFS_TEST_SLEEP", "1")
    with pytest.raises(UpdateError, match="timed out"):
        asyncio.run(fs.get_latest_version("hdfs://cluster/models"))


def test_missing_selected_version_never_falls_back(
    hdfs: tuple[HDFSDownloader, Path, Path], tmp_path: Path
) -> None:
    fs, remote, _ = hdfs
    ready(remote, "iter2_label_1800000000000")
    target = tmp_path / "absent"
    with pytest.raises(UpdateError, match="not published"):
        asyncio.run(fs.download("hdfs://cluster/models", "iter1_label_1700000000000", target))
    assert not target.exists()


@pytest.mark.parametrize(  # type: ignore[misc]
    "binary,dfs,fast",
    [
        ("/opt/tiger/hdfs_client/bin/hdfs", False, True),
        ("/opt/tiger/arnold/hdfs_client/hdfs", True, True),
        ("/usr/bin/hdfs", True, False),
    ],
)
def test_novi_client_dialects(
    binary: str, dfs: bool, fast: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HDFS_USES_DFS_SUBCOMMAND", raising=False)
    fs = HDFSDownloader(binary)
    argv = fs._argv("get", "source", "target")
    assert (argv[1] == "dfs") is dfs
    assert ("--ct" in argv) is fast


@pytest.mark.parametrize(  # type: ignore[misc]
    "name,key",
    [
        ("iter1_x_1700000000000", 1700000000000),
        ("trie_x_20261103_012345", 20261103012345),
        ("1700000000000", 1700000000000),
        ("_SUCCESS", None),
    ],
)
def test_novi_version_names(name: str, key: int | None) -> None:
    assert timestamp_version(name) == key


def test_wrapper_notice_does_not_turn_missing_marker_into_failure(
    hdfs: tuple[HDFSDownloader, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, remote, _ = hdfs
    ready(remote, "ckpt_1700000000")
    (remote / "models/ckpt_1800000000").mkdir()
    monkeypatch.setenv("HDFS_TEST_NOTICE", "1")
    assert asyncio.run(fs.get_latest_version("hdfs://cluster/models")) == "ckpt_1700000000"
    monkeypatch.setenv("HDFS_TEST_DENIED", "1")
    with pytest.raises(UpdateError, match="Permission denied"):
        asyncio.run(fs.get_latest_version("hdfs://cluster/models"))


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
