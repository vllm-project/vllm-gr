# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CI contracts for Beam Recall calibration and reporting artifacts."""

import json
import sys
from pathlib import Path

import pytest

from benchmarks.beam_search.compare_recall_baseline import TASKS, main, summarize
from tests.helpers.beam_recall import make_repeated_results, set_sid_recall
from tests.helpers.paths import REPO_ROOT


def test_committed_baseline_uses_measured_upgrade_calibration() -> None:
    path = REPO_ROOT / "benchmarks/beam_search/recall_baseline_batch4.json"
    baseline = json.loads(path.read_text(encoding="utf-8"))
    count = baseline["source_ci"]["calibration_repeats"]
    assert count >= 3
    assert baseline["runtime_provenance"]["vllm"] == "0.30.0"
    assert baseline["runtime_provenance"]["accelerator"] == "NVIDIA A100-PCIE-40GB"
    assert baseline["source_ci"]["workflow_run_id"] == 41385992758
    for task in TASKS:
        values = baseline["calibration_sid_recall32"][task]
        assert len(values) == count
        assert values[0] == baseline["tasks"][task]["recall@32"]
        assert len(baseline["tasks"][task]["sample_ids"]) == 1000


def test_cli_writes_repeat_artifacts_and_step_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    step_summary = tmp_path / "github-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    monkeypatch.delenv("V1_RECALL_BASELINE_FILE", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["compare_recall_baseline", "--result-dir", str(tmp_path), "--repeats", "10"]
    )
    main()
    assert (tmp_path / "b4_baseline_candidate.json").exists()
    repeated = json.loads((tmp_path / "b4_repeat_summary.json").read_text(encoding="utf-8"))
    assert repeated["repeat_count"] == 10
    assert len(repeated["runs"]) == 10
    assert "All 10 repeats wall time: 1050 s." in step_summary.read_text(encoding="utf-8")


def test_cli_confirmed_failure_preserves_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    candidate, result, _, _ = summarize(tmp_path, 10)
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    monkeypatch.setenv("V1_RECALL_BASELINE_FILE", str(baseline))
    monkeypatch.setattr(
        sys, "argv", ["compare_recall_baseline", "--result-dir", str(tmp_path), "--repeats", "3"]
    )
    for run in (1, 2, 3):
        set_sid_recall(tmp_path, run, "video", 0.038)
    with pytest.raises(SystemExit, match="Recall gate failed"):
        main()
    assert (tmp_path / "b4_baseline_candidate.json").exists()
    assert (tmp_path / "b4_repeat_summary.json").exists()
    assert "FAILED" in (tmp_path / "b4_repeat_summary.md").read_text(encoding="utf-8")


pytestmark = pytest.mark.cpu_test
