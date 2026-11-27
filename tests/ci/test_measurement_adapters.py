# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Preserve measurement policy, provenance and output requirements at the adapter boundary."""

import json
import os
from pathlib import Path

import pytest

from tests.helpers import measurement_gate as gate
from tests.helpers.artifacts import snapshot_spec

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize("suite", ["recall", "performance", "performance-ab"])
def test_missing_measurement_report_is_not_success(tmp_path: Path, suite: str) -> None:
    with pytest.raises(FileNotFoundError):
        gate.validate_outputs(suite, tmp_path)


@pytest.mark.parametrize("complete", [False, "true", None])
def test_performance_requires_complete_runs(tmp_path: Path, complete: object) -> None:
    for name in ("setup", "offline", "online"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"complete": complete}))
    with pytest.raises(ValueError, match="Incomplete"):
        gate.validate_outputs("performance", tmp_path)


def test_ab_preserves_context_repetition_and_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in list(os.environ):
        if key.startswith(("V1_", "VLLM_GR_TEST")) or key in {
            "TRIPLES_PATH",
            "CATALOG_PATH",
        }:
            monkeypatch.delenv(key)
    triples = tmp_path / "video.json"
    triples.write_text("[]")
    values = {
        "TRIPLES_PATH": str(triples),
        "V1_PERF_BASE_SHA": "a" * 40,
        "V1_PERF_HEAD_SHA": "b" * 40,
        "V1_PERF_CANDIDATE_SHA": "c" * 40,
        "V1_PERF_PR_NUMBER": "17",
        "V1_PERF_MEASURED_RUNS": "40",
        "V1_PERF_WARMUP_RUNS": "2",
        "P99_TTFT_THRESHOLD": "100",
        "ACCURACY_FAILURE_MODE": "warn",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("V1_RECALL_MODEL_REVISION", "unrelated")
    env = gate.prepare_environment(
        "performance-ab", tmp_path, lambda name: pytest.fail(f"Unexpected fixture: {name}")
    )
    assert all(env[key] == value for key, value in values.items())
    assert env["V1_PERF_MODEL_REVISION"] == snapshot_spec("onerec1b7").revision
    assert env["V1_PERF_OUTPUT_DIR"] == str(tmp_path)


def test_recall_keeps_explicit_inputs_and_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in ("VLLM_GR_TEST_MODEL", "CATALOG_PATH", "TRIPLES_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("V1_RECALL_DATASET_PATH", str(tmp_path))
    monkeypatch.setenv("V1_RECALL_REPEATS", "3")
    for key in ("V1_RECALL_TRIPLES_PATH", "V1_RECALL_TRIPLES_IID_PATH", "V1_RECALL_BASELINE_FILE"):
        path = tmp_path / f"{key}.json"
        path.write_text('{"identity": "unchanged"}')
        monkeypatch.setenv(key, str(path))
    env = gate.prepare_environment(
        "recall", tmp_path, lambda name: pytest.fail(f"Unexpected fixture: {name}")
    )
    assert env["V1_RECALL_REPEATS"] == "3"
    assert env["V1_RECALL_BASELINE_FILE"].endswith("V1_RECALL_BASELINE_FILE.json")
    assert env["V1_RECALL_DATASET_PATH"] == str(tmp_path)
    assert not (tmp_path / "inputs.json").exists()


def test_performance_prepares_missing_video_triples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TRIPLES_PATH", raising=False)
    monkeypatch.delenv("VLLM_GR_TEST_DATASET", raising=False)

    def video(dataset: Path) -> Path:
        assert dataset == tmp_path
        (dataset / "triples.json").write_text("[]")
        return dataset

    monkeypatch.setattr(gate, "video_inputs", video)

    def fixture(name: str) -> Path:
        assert name == "onerec_dataset"
        return tmp_path

    env = gate.prepare_environment("performance", tmp_path, fixture)
    assert Path(env["TRIPLES_PATH"]).is_file()
