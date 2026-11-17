# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU checks for repeated OneRec accuracy-gate aggregation."""

import json
import sys
from pathlib import Path

import pytest

from benchmarks.beam_search.v1_recall_summary import TASKS, main, summarize
from tests.helpers.paths import REPO_ROOT


def make_repeated_results(root: Path) -> None:
    (root / "commit.txt").write_text("test-commit\n", encoding="utf-8")
    (root / "total_wall_seconds.txt").write_text("1050\n", encoding="utf-8")
    for number in range(1, 11):
        run_dir = root / "runs" / f"run-{number:02d}"
        run_dir.mkdir(parents=True)
        (run_dir / "total_wall_seconds.txt").write_text(f"{100 + number}\n", encoding="utf-8")
        for group, seconds in (("sid2pid", 60), ("sid2iid", 40)):
            group_dir = run_dir / "groups" / group
            group_dir.mkdir(parents=True)
            (group_dir / "wall_seconds.txt").write_text(f"{seconds}\n", encoding="utf-8")
        for task in TASKS:
            task_dir = run_dir / task
            task_dir.mkdir()
            (task_dir / "wall_seconds.txt").write_text("20\n", encoding="utf-8")
            recall = 0.04 + number * 0.00001
            if task == "video" and number == 7:
                recall -= 0.001
            metrics = {
                "task": task,
                "total_samples": 1000,
                "batch_size": 4,
                "beam_width": 128,
                "temperature": 0.6,
                "sample_ids": list(range(1000)),
                "recall@32": recall,
                "pid_recall@32": 0.03 + number * 0.00001,
                "elapsed_time": 10 + number,
                "source_sha256": f"source-{task}",
                "artifact_digest": "artifact",
                "tokenizer_digest": "tokenizer",
                "interactive_tailcut_count": 0,
                "model_revision": "test-revision",
                "engine_config": {
                    "max_model_len": 8192,
                    "max_num_batched_tokens": 10000,
                    "async_scheduling": True,
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": False,
                    "max_num_seqs": 128,
                    "max_logprobs": 256,
                    "enforce_eager": False,
                    "dtype": "auto",
                    "seed": 0,
                },
                "graph_profile": {
                    "prefill_bucket_grid": "16,24,32,48",
                    "prefill_capture_num_reqs": "4",
                    "prefill_kv_bound": "8192",
                    "v1_startup_capture_batch_size": "4",
                },
                "graph_runtime": {
                    "captures": [[[16, 4]]],
                    "effective_profiles": [
                        {"runner": "MockRunner", "buckets": [16], "kv_bounds": {"16:4": 8192}}
                    ],
                    "dispatches": {"replay:4": 250},
                    "expected_b4_replays": 250,
                },
                "runtime_provenance": {
                    "accelerator": "test GPU",
                    "cuda": "test",
                    "torch": "test",
                    "vllm": "test",
                    "python": "test",
                },
                "prefill_graph_buckets": "16,24,32,48",
                "prefill_graph_capture_num_reqs": "4",
                "v1_startup_capture_batch_size": "4",
            }
            (task_dir / f"model_{task}_bw128_bs4_1000.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )


def _set_sid_recall(root: Path, run: int, task: str, recall: float) -> None:
    file = root / "runs" / f"run-{run:02d}" / task / f"model_{task}_bw128_bs4_1000.json"
    metrics = json.loads(file.read_text(encoding="utf-8"))
    metrics["recall@32"] = recall
    file.write_text(json.dumps(metrics), encoding="utf-8")


def test_ten_repeats_report_variation_and_confirm_task_specific_regression(tmp_path: Path) -> None:
    make_repeated_results(tmp_path)
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

    _set_sid_recall(tmp_path, 1, "video", 0.038)
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

    _set_sid_recall(tmp_path, 2, "video", 0.038)
    _, result, summary, failures = summarize(tmp_path, 3, baseline_path)
    assert not failures
    assert result["confirmation_required"] is False

    _set_sid_recall(tmp_path, 3, "video", 0.038)
    _, result, summary, failures = summarize(tmp_path, 3, baseline_path)
    assert failures == [
        "video: SID Recall@32 below committed baseline 3.9070% in all three measurements"
    ]
    assert result["baseline_failures"] == failures
    assert "Baseline check: FAILED." in summary


def test_repeated_gate_rejects_changed_sample_identity(tmp_path: Path) -> None:
    make_repeated_results(tmp_path)
    file = tmp_path / "runs" / "run-02" / "ad" / "model_ad_bw128_bs4_1000.json"
    metrics = json.loads(file.read_text(encoding="utf-8"))
    metrics["sample_ids"][0] = -1
    file.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="run 2/ad: input differs from run 1: sample_ids"):
        summarize(tmp_path, 10)


@pytest.mark.parametrize("calibration_repeats", [3, 10])  # type: ignore[misc]
def test_declared_calibration_count_keeps_screening_and_confirmation(
    tmp_path: Path, calibration_repeats: int
) -> None:
    make_repeated_results(tmp_path)
    candidate, result, _, _ = summarize(tmp_path, calibration_repeats)
    candidate["source_ci"] = {"calibration_repeats": calibration_repeats}
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    floor = min(candidate["calibration_sid_recall32"]["video"])
    _set_sid_recall(tmp_path, 1, "video", floor)
    _, result, summary, failures = summarize(tmp_path, 1, baseline)
    assert not failures and not result["confirmation_required"]
    assert result["baseline_calibration_repeats"] == calibration_repeats
    assert f"from {calibration_repeats} calibration measurements" in summary
    for run in (1, 2, 3):
        _set_sid_recall(tmp_path, run, "video", floor - 0.001)
    _, result, _, failures = summarize(tmp_path, 1, baseline)
    assert not failures and result["confirmation_required"]
    _, _, _, failures = summarize(tmp_path, 3, baseline)
    assert len(failures) == 1 and failures[0].startswith("video:")


@pytest.mark.parametrize("calibration_repeats", [0, 1, 2, True, 3.0, "3"])  # type: ignore[misc]
def test_invalid_calibration_count_is_rejected(tmp_path: Path, calibration_repeats: object) -> None:
    make_repeated_results(tmp_path)
    candidate, result, _, _ = summarize(tmp_path, 3)
    candidate["source_ci"] = {"calibration_repeats": calibration_repeats}
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="at least three calibration measurements"):
        summarize(tmp_path, 1, baseline)


def test_calibration_samples_must_match_declared_count(tmp_path: Path) -> None:
    make_repeated_results(tmp_path)
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


def test_committed_baseline_uses_measured_upgrade_calibration() -> None:
    path = REPO_ROOT / "benchmarks/beam_search/v1_recall_b4_baseline.json"
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
    make_repeated_results(tmp_path)
    step_summary = tmp_path / "github-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    monkeypatch.delenv("V1_RECALL_BASELINE_FILE", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["v1_recall_summary", "--result-dir", str(tmp_path), "--repeats", "10"]
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
    make_repeated_results(tmp_path)
    candidate, result, _, _ = summarize(tmp_path, 10)
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    monkeypatch.setenv("V1_RECALL_BASELINE_FILE", str(baseline))
    monkeypatch.setattr(
        sys, "argv", ["v1_recall_summary", "--result-dir", str(tmp_path), "--repeats", "3"]
    )
    for run in (1, 2, 3):
        _set_sid_recall(tmp_path, run, "video", 0.038)
    with pytest.raises(SystemExit, match="Recall gate failed"):
        main()
    assert (tmp_path / "b4_baseline_candidate.json").exists()
    assert (tmp_path / "b4_repeat_summary.json").exists()
    assert "FAILED" in (tmp_path / "b4_repeat_summary.md").read_text(encoding="utf-8")


def test_graph_profile_comes_from_metrics_and_requires_b4_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repeated_results(tmp_path)
    monkeypatch.setenv("VLLM_GR_PREFILL_GRAPH_KV_BOUND", "changed")
    candidate, _, _, _ = summarize(tmp_path, 1)
    assert candidate["graph_profile"]["prefill_kv_bound"] == "8192"
    path = tmp_path / "runs" / "run-01" / "video" / "model_video_bw128_bs4_1000.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["graph_runtime"]["dispatches"] = {"fallback:4:no_graph": 250}
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="B=4 graph execution"):
        summarize(tmp_path, 1)


def test_different_tasks_dipping_in_different_rounds_do_not_fail(tmp_path: Path) -> None:
    make_repeated_results(tmp_path)
    candidate, result, _, _ = summarize(tmp_path, 10)
    candidate["calibration_sid_recall32"] = {
        task: [run["tasks"][task]["recall@32"] for run in result["runs"]] for task in TASKS
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(candidate), encoding="utf-8")
    for run, task in ((1, "video"), (2, "ad"), (3, "product")):
        _set_sid_recall(tmp_path, run, task, 0.038)
    _, result, _, failures = summarize(tmp_path, 3, baseline)
    assert result["suspected_tasks"] == ["video"]
    assert failures == []


pytestmark = pytest.mark.cpu_test
