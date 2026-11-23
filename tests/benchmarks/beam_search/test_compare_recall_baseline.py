# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU checks for repeated OneRec accuracy-gate aggregation."""

import json
from pathlib import Path

import pytest

from benchmarks.beam_search.compare_recall_baseline import TASKS, summarize
from tests.helpers.beam_recall import make_repeated_results, set_sid_recall


def test_ten_repeats_report_variation_and_confirm_task_specific_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    candidate, result, summary, failures = summarize(tmp_path, 10)
    assert not failures
    assert len(result["runs"]) == 10
    assert result["per_run_wall_seconds"]["min"] == 101
    assert result["per_run_wall_seconds"]["max"] == 110
    assert result["task_variability"]["video"]["sid_recall32_pp"]["range"] == pytest.approx(0.103)
    assert "| video | 7 | 3.9070% |" in summary
    assert candidate["tasks"]["video"]["recall@32"] == pytest.approx(0.04001)

    baseline_path = tmp_path / "baseline.json"
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline_path.write_text(json.dumps(candidate), encoding="utf-8")
    _, result, summary, failures = summarize(tmp_path, 10, baseline_path)
    assert not failures
    assert result["baseline_sid_recall32_floors"]["video"] == pytest.approx(0.03907)
    assert result["confirmation_required"] is False
    assert "| Task | SID baseline | Current SID | Delta SID (pp) | " in summary
    assert "| video | 3.9070% | 4.0045% | +0.0975 | 3.0010% | 3.0055% | +0.0045 |" in summary
    assert "Historical median" not in summary
    assert "Baseline check: passed." in summary

    set_sid_recall(tmp_path, 1, "video", 0.038)
    _, result, summary, failures = summarize(tmp_path, 1, baseline_path)
    assert not failures
    assert result["suspected_tasks"] == ["video"]
    assert result["confirmation_required"] is True
    assert "| video | 3.9070% | 3.8000% | -0.1070 | 3.0010% | 3.0010% | +0.0000 |" in summary
    assert "| Task | Inference s | Task work s |" in summary
    assert "| Task | Run | SID Recall@32 |" not in summary
    assert "| Run | Five-task wall s |" not in summary
    assert "| Task | SID min | SID median |" not in summary
    assert "Total wall time:" in summary
    assert "Baseline check: pending confirmation." in summary

    set_sid_recall(tmp_path, 2, "video", 0.038)
    _, result, summary, failures = summarize(tmp_path, 3, baseline_path)
    assert not failures
    assert result["confirmation_required"] is False

    set_sid_recall(tmp_path, 3, "video", 0.038)
    _, result, summary, failures = summarize(tmp_path, 3, baseline_path)
    assert failures == [
        "video: SID Recall@32 below committed baseline 3.9070% in all three measurements"
    ]
    assert result["baseline_failures"] == failures
    assert "Baseline check: FAILED." in summary


def test_repeated_gate_rejects_changed_sample_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    file = tmp_path / "runs" / "run-02" / "ad" / "model_ad_bw128_bs4_1000.json"
    metrics = json.loads(file.read_text(encoding="utf-8"))
    metrics["sample_ids"][0] = -1
    file.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="run 2/ad: input differs from run 1: sample_ids"):
        summarize(tmp_path, 10)


@pytest.mark.parametrize("calibration_repeats", [3, 10])  # type: ignore[misc]
def test_declared_calibration_count_keeps_screening_and_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calibration_repeats: int
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    candidate, result, _, _ = summarize(tmp_path, calibration_repeats)
    candidate["source_ci"] = {"calibration_repeats": calibration_repeats}
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    floor = min(candidate["calibration_sid_recall32"]["video"])
    set_sid_recall(tmp_path, 1, "video", floor)
    _, result, summary, failures = summarize(tmp_path, 1, baseline)
    assert not failures and not result["confirmation_required"]
    assert result["baseline_calibration_repeats"] == calibration_repeats
    assert f"from {calibration_repeats} calibration measurements" in summary
    for run in (1, 2, 3):
        set_sid_recall(tmp_path, run, "video", floor - 0.001)
    _, result, _, failures = summarize(tmp_path, 1, baseline)
    assert not failures and result["confirmation_required"]
    _, _, _, failures = summarize(tmp_path, 3, baseline)
    assert len(failures) == 1 and failures[0].startswith("video:")


@pytest.mark.parametrize("calibration_repeats", [0, 1, 2, True, 3.0, "3"])  # type: ignore[misc]
def test_invalid_calibration_count_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calibration_repeats: object
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    candidate, result, _, _ = summarize(tmp_path, 3)
    candidate["source_ci"] = {"calibration_repeats": calibration_repeats}
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="at least three calibration measurements"):
        summarize(tmp_path, 1, baseline)


def test_calibration_samples_must_match_declared_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    candidate, result, _, _ = summarize(tmp_path, 3)
    candidate["source_ci"] = {"calibration_repeats": 3}
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    candidate["calibration_sid_recall32"]["ad"].pop()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="ad: invalid historical SID calibration"):
        summarize(tmp_path, 1, baseline)


def test_graph_profile_comes_from_metrics_and_requires_b4_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_GR_PREFILL_GRAPH_KV_BOUND", "changed")
    candidate, _, _, _ = summarize(tmp_path, 1)
    assert candidate["graph_profile"]["prefill_kv_bound"] == "8192"
    path = tmp_path / "runs" / "run-01" / "video" / "model_video_bw128_bs4_1000.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["graph_runtime"]["dispatches"] = {"fallback:4:no_graph": 250}
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="B=4 graph execution"):
        summarize(tmp_path, 1)


def test_different_tasks_dipping_in_different_rounds_do_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repeated_results(tmp_path, monkeypatch)
    candidate, result, _, _ = summarize(tmp_path, 10)
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    for run, task in ((1, "video"), (2, "ad"), (3, "product")):
        set_sid_recall(tmp_path, run, task, 0.038)
    _, result, _, failures = summarize(tmp_path, 3, baseline)
    assert result["suspected_tasks"] == ["video"]
    assert failures == []


pytestmark = pytest.mark.cpu_test
