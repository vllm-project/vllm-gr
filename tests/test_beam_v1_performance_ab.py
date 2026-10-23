# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import statistics
from pathlib import Path

import pytest

from benchmarks.beam_search.v1_performance_ab import compare

BASE_SHA = "a" * 40
PR_SHA = "b" * 40


def write_pass(root: Path, sha: str, offline_ms: list[float | list[float]]) -> None:
    root.mkdir()
    (root / "commit.txt").write_text(sha + "\n", encoding="utf-8")
    (root / "setup.json").write_text(
        json.dumps(
            {
                "benchmark_driver_sha256": "d" * 64,
                "model": "OpenOneRec/OneRec-1.7B",
                "revision": "cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b",
                "input_tokens_including_sid_begin": 1024,
                "beam_width": 128,
                "generated_steps": 3,
                "batch_sizes": [4, 2, 1],
                "warmup_runs": 10,
                "b1_warmup_runs": 20,
                "measured_runs": 20,
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.7,
                "artifact_digest": "same-artifact",
                "tokenizer_digest": "same-tokenizer",
                "execution_mode": "v1",
                "graph_enabled": True,
                "prefix_caching": False,
                "chunked_prefill": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "prompt_token_ids.json").write_text('"same"', encoding="utf-8")
    cases = []
    values_by_count = dict(zip((1, 2, 4), offline_ms, strict=True))
    for count in (4, 2, 1):
        values = values_by_count[count]
        samples = [values] * 20 if isinstance(values, (int, float)) else values
        cases.append(
            {
                "mode": "offline",
                "batch_size": count,
                "group_samples_ms": samples,
                "group_latency_ms": {"mean_ms": statistics.fmean(samples)},
                "prompt_tokens_including_sid_begin": [1024] * count,
            }
        )
    (root / "offline.json").write_text(
        json.dumps({"mode": "offline", "complete": True, "cases": cases}),
        encoding="utf-8",
    )


def write_run(
    root: Path,
    sha: str,
    offline_ms: list[float | list[float]],
    second_pass_ms: list[float | list[float]] | None = None,
) -> None:
    root.mkdir()
    write_pass(root / "1", sha, offline_ms)
    write_pass(root / "2", sha, second_pass_ms if second_pass_ms is not None else offline_ms)


def test_ab_report_fails_offline_regression(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [33.78, 56.32, 93.44])
    write_run(pr, PR_SHA, [34.32, 56.32, 93.44])

    result, report = compare(base, pr, BASE_SHA, PR_SHA)

    assert result["complete"] is True
    assert result["rows"][0]["status"] == "FAIL"  # +1.60%
    assert len(result["rows"]) == 3
    assert len(result["failures"]) == 3
    assert result["max_regression_percent"] == 1.0
    assert result["gated_modes"] == ["offline"]
    assert result["gated_metrics"] == ["mean", "p50", "p90"]
    assert "Change %" in report
    assert "| 1 | p50 | 33.780 | 34.320 | +1.60% | FAIL |" in report
    assert "| 1 | p90 | 33.780 | 34.320 | +1.60% | FAIL |" in report
    assert "Base CV" in report
    assert "batches per pass: B=4→2→1" in report
    assert "warmup per pass: B=1 20, B=2/4 10" in report
    assert "Overall: **FAIL**" in report


def test_ab_rejects_different_prompt_tokens(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [33.78, 56.32, 93.44])
    write_run(pr, PR_SHA, [33.78, 56.32, 93.44])
    (base / "1" / "prompt_token_ids.json").write_text('"base"', encoding="utf-8")
    (pr / "1" / "prompt_token_ids.json").write_text('"pr"', encoding="utf-8")

    with pytest.raises(ValueError, match="different prompt_token_ids.json"):
        compare(base, pr, BASE_SHA, PR_SHA)


def test_ab_allows_offline_changes_within_one_percent(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [33.78, 56.32, 93.44])
    write_run(pr, PR_SHA, [34.05, 56.75, 94.15])

    result, report = compare(base, pr, BASE_SHA, PR_SHA)

    assert result["failures"] == []
    assert len(result["rows"]) == 3
    assert all(row["status"] == "PASS" for row in result["rows"])
    assert "Overall: **PASS**" in report


def test_ab_rejects_different_benchmark_drivers(tmp_path: Path) -> None:
    base, pr = tmp_path / "base", tmp_path / "pr"
    write_run(base, BASE_SHA, [100.0, 100.0, 100.0])
    write_run(pr, PR_SHA, [100.0, 100.0, 100.0])
    path = pr / "2/setup.json"
    setup = json.loads(path.read_text())
    setup["benchmark_driver_sha256"] = "e" * 64
    path.write_text(json.dumps(setup))
    with pytest.raises(ValueError, match="benchmark settings differ"):
        compare(base, pr, BASE_SHA, PR_SHA)


def test_ab_accepts_exact_one_percent_boundary(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [100.0, 100.0, 100.0])
    write_run(pr, PR_SHA, [101.0, 101.0, 101.0])

    result, report = compare(base, pr, BASE_SHA, PR_SHA)

    assert result["failures"] == []
    assert all(set(row["metric_statuses"].values()) == {"PASS"} for row in result["rows"])
    assert "Overall: **PASS**" in report


def test_ab_reports_tail_and_variability_from_forty_samples(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [[10.0] * 20, 20.0, 30.0])
    write_run(pr, PR_SHA, [[10.0] * 18 + [20.0] * 2, 20.0, 30.0])

    result, report = compare(base, pr, BASE_SHA, PR_SHA)

    row = result["rows"][0]
    assert row["base_mean_ms"] == 10.0
    assert row["pr_mean_ms"] == 11.0
    assert row["change_percent"] == pytest.approx(10.0)
    assert row["p50_change_percent"] == 0.0
    assert row["p90_change_percent"] == pytest.approx(10.0)
    assert row["base_cv_percent"] == 0.0
    assert row["pr_cv_percent"] == pytest.approx(27.27, abs=0.01)
    assert row["metric_statuses"] == {"mean": "FAIL", "p50": "PASS", "p90": "FAIL"}
    assert "| 1 | p50 | 10.000 | 10.000 | +0.00% | PASS |" in report
    assert "| 1 | p90 | 10.000 | 11.000 | +10.00% | FAIL |" in report


def test_ab_gates_each_latency_statistic_independently(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(
        base,
        BASE_SHA,
        [[100.0] * 20, [100.0] * 10 + [120.0] * 10, [100.0] * 20],
    )
    write_run(
        pr,
        PR_SHA,
        [
            [100.0] * 19 + [130.0],
            [90.0] * 9 + [103.0] + [120.0] * 10,
            [80.0] * 3 + [100.0] * 14 + [120.0] * 3,
        ],
    )

    result, report = compare(base, pr, BASE_SHA, PR_SHA)

    assert [row["metric_statuses"] for row in result["rows"]] == [
        {"mean": "FAIL", "p50": "PASS", "p90": "PASS"},
        {"mean": "PASS", "p50": "FAIL", "p90": "PASS"},
        {"mean": "PASS", "p50": "PASS", "p90": "FAIL"},
    ]
    assert len(result["failures"]) == 3
    assert "offline B=1 mean" in result["failures"][0]
    assert "offline B=2 p50" in result["failures"][1]
    assert "offline B=4 p90" in result["failures"][2]
    assert "| 4 | p90 | 100.000 | 120.000 | +20.00% | FAIL |" in report


def test_ab_reports_cv_without_gating_it(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [100.0, 100.0, 100.0])
    write_run(pr, PR_SHA, [[99.1] * 10 + [100.9] * 10, 100.0, 100.0])

    result, report = compare(base, pr, BASE_SHA, PR_SHA)

    assert result["failures"] == []
    assert result["rows"][0]["pr_cv_percent"] == pytest.approx(0.9)
    assert "| 1 | 0.00% | 0.90% |" in report


def test_ab_rejects_fewer_than_twenty_samples(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [10.0, 20.0, 30.0])
    write_run(pr, PR_SHA, [[10.0] * 19, 20.0, 30.0])

    with pytest.raises(ValueError, match="invalid offline B/C=1 timing samples"):
        compare(base, pr, BASE_SHA, PR_SHA)


def test_ab_uses_both_passes_for_each_side(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [100.0, 100.0, 100.0])
    write_run(pr, PR_SHA, [100.0, 100.0, 100.0], [104.0, 100.0, 100.0])

    result, report = compare(base, pr, BASE_SHA, PR_SHA)

    assert result["rows"][0]["base_mean_ms"] == 100.0
    assert result["rows"][0]["pr_mean_ms"] == 102.0
    assert result["rows"][0]["metric_statuses"] == {"mean": "FAIL", "p50": "FAIL", "p90": "FAIL"}
    assert "order: A→B→B→A" in report
    assert "(40 per side)" in report


def test_ab_rejects_missing_final_pass(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pr = tmp_path / "pr"
    write_run(base, BASE_SHA, [100.0, 100.0, 100.0])
    write_run(pr, PR_SHA, [100.0, 100.0, 100.0])
    (base / "2" / "commit.txt").unlink()

    with pytest.raises(ValueError, match="missing A/B pass"):
        compare(base, pr, BASE_SHA, PR_SHA)
