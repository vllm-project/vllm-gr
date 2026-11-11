# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for the opt-in stage profiler."""

import json
import os
from pathlib import Path

import pytest

from vllm_gr import stage_profile

pytestmark = pytest.mark.cpu_test


def test_profiler_is_inert_until_enabled(tmp_path: Path) -> None:
    assert not stage_profile.enabled()
    stage_profile.record("unit")
    stage_profile.count("counter")
    stage_profile.durations("span", start_ns=stage_profile.now())
    assert list(tmp_path.iterdir()) == []


def test_profiler_writes_spans_and_flushed_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stage_profile, "_ENABLED", True)
    monkeypatch.setattr(stage_profile, "_PREFIX", str(tmp_path / "prof"))
    monkeypatch.setattr(stage_profile, "_STREAM", None)
    monkeypatch.setattr(stage_profile, "_COUNTS", {})

    stage_profile.durations("span", start_ns=stage_profile.now(), tag="x")
    stage_profile.count("hits", 3)
    stage_profile._flush_counts()

    path = tmp_path / f"prof.{os.getpid()}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "span"
    assert rows[0]["tag"] == "x"
    assert rows[0]["us"] >= 0.0
    counts = [row for row in rows if row["kind"] == "counts"][-1]
    assert counts["hits"] == 3
