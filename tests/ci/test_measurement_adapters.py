# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Preserve measurement policy, provenance and output requirements at the adapter boundary."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.helpers import measurement_gate as gate
from tests.helpers.artifacts import snapshot_spec

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_pangu_provenance_ignores_unrelated_onerec_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VLLM_GR_TEST_MODEL", "unrelated-onerec-model")
    monkeypatch.setenv("V1_RECALL_BASELINE_FILE", str(tmp_path / "unrelated-missing.json"))
    data = tmp_path / "data.json"
    data.write_text('[{"question":"q","answer":"a"}]')
    gate.prepare_environment(
        "pangu-offline", tmp_path, lambda name: tmp_path if name == "pangu_model" else data
    )
    report = json.loads((tmp_path / "inputs.json").read_text())
    assert report["model"] == str(tmp_path)
    assert set(report["inputs"]) == {"PANGU1B_DATASET_PATH"}


@pytest.fixture(autouse=True)
def reset_vocab_slice_config() -> None:
    pass


@pytest.mark.parametrize(
    "suite", ["pangu-offline", "pangu-online", "recall", "performance", "performance-ab"]
)
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
        if key.startswith(("V1_", "PANGU", "VLLM_GR_TEST")) or key in {
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
    assert (
        json.loads((tmp_path / "inputs.json").read_text())["revision"]
        == env["V1_PERF_MODEL_REVISION"]
    )
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
    provenance = json.loads((tmp_path / "inputs.json").read_text())
    assert provenance["dataset_source"] == "local"
    assert provenance["dataset_revision"] is None
    assert provenance["inputs"]["V1_RECALL_BASELINE_FILE"]["sha256"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX Bash gate")
@pytest.mark.parametrize("mode", ["online", "offline"])
def test_success_without_new_pangu_output_fails(tmp_path: Path, mode: str) -> None:
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nif [ "$1" = "-m" ]; then exit 0; fi\nexec python3 "$@"\n')
    python.chmod(0o755)
    result = tmp_path / "previous.json"
    result.write_text('{"accuracy": 1.0}')
    prefix = f"PANGU1B_{mode.upper()}"
    env = {
        **os.environ,
        "PATH": str(tmp_path) + ":" + os.environ["PATH"],
        f"{prefix}_RESULT_JSON": str(result),
        f"{prefix}_ACCURACY_THRESHOLD": "0.9",
        f"{prefix}_ACCURACY_FAILURE_MODE": "blocking",
    }
    script = gate.REPO_ROOT / "tests/system" / gate.SCRIPTS[f"pangu-{mode}"]
    run = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert run.returncode == 1, run.stderr
    assert result.read_text() == '{"accuracy": 1.0}'


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
