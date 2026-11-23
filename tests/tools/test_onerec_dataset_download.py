# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import sys
from pathlib import Path
from typing import cast

import pytest

from tools.AB_Tests import download_onerec_dataset as downloader

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def run_download(monkeypatch: pytest.MonkeyPatch, data: Path, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["download_onerec_dataset.py", "--data-dir", str(data), *args])
    return cast(int, downloader.main())


def test_verify_does_not_register_arbitrary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = downloader.destination_for(tmp_path, "video")
    video.parent.mkdir()
    video.write_bytes(b"unverified")

    assert run_download(monkeypatch, tmp_path, "--verify-only") != 0
    assert not (tmp_path / ".onerec-manifest.json").exists()


def test_verify_detects_changes_and_preserves_other_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache.parquet"
    cache.write_bytes(b"download")
    monkeypatch.setattr(downloader, "hf_hub_download", lambda **kwargs: str(cache))
    data = tmp_path / "data"
    assert run_download(monkeypatch, data, "--tasks", "video,ad") == 0
    assert run_download(monkeypatch, data) == 0
    manifest = data / ".onerec-manifest.json"
    original = manifest.read_bytes()
    assert {entry["task"] for entry in json.loads(original)["files"]} == {"video", "ad"}

    assert run_download(monkeypatch, data, "--verify-only") == 0
    assert manifest.read_bytes() == original
    wrong_revision = "a" * 40
    assert run_download(monkeypatch, data, "--verify-only", "--revision", wrong_revision) != 0
    error = capsys.readouterr().err
    assert wrong_revision in error and downloader.DEFAULT_REVISION in error
    video = downloader.destination_for(data, "video")
    video.write_bytes(b"modified")
    assert cache.read_bytes() == b"download"
    assert run_download(monkeypatch, data, "--verify-only") != 0
    error = capsys.readouterr().err
    assert str(video) in error and "sha256" in error
    assert manifest.read_bytes() == original


def test_download_endpoint_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache.parquet"
    cache.write_bytes(b"download")
    endpoints = []

    def download(*, endpoint: str, **kwargs: str) -> str:
        assert kwargs["revision"] == downloader.DEFAULT_REVISION
        endpoints.append(endpoint)
        return str(cache)

    monkeypatch.setattr(downloader, "hf_hub_download", download)
    data = tmp_path / "data"
    monkeypatch.setenv("HF_ENDPOINT", "")
    assert run_download(monkeypatch, data) == 0
    monkeypatch.setenv("HF_ENDPOINT", "https://env.example")
    assert run_download(monkeypatch, data) == 0
    assert run_download(monkeypatch, data, "--endpoint", "https://cli.example") == 0
    assert endpoints == ["https://huggingface.co", "https://env.example", "https://cli.example"]
