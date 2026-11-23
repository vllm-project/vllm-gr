# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
ROOT = Path(__file__).resolve().parents[2]
Summary = dict[str, Any]


def compare_summaries(
    directory: Path,
    base: Sequence[tuple[int, float | str] | Summary],
    head: Sequence[tuple[int, float | str] | Summary],
    *,
    git_shas: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    markdown = directory / "report.md"
    report_json = directory / "report.json"
    command = [
        sys.executable,
        str(ROOT / "tools/AB_Tests/compare_ab.py"),
        "--output",
        str(markdown),
        "--json-output",
        str(report_json),
        "--label",
        "Synthetic comparison",
    ]
    for side, runs in (("base", base), ("head", head)):
        for index, run in enumerate(runs):
            summary = directory / f"{side}-{index}.json"
            git_sha = (git_shas or {}).get(summary.name, side)
            if isinstance(run, dict):
                payload = run
            else:
                completed, latency = run
                payload = {
                    "source": {"git_sha": git_sha},
                    "results": {
                        "requests": {"completed": completed},
                        "latency_ms": {
                            "e2el": dict.fromkeys(("mean", "p50", "p90", "p99"), latency)
                        },
                    },
                }
            summary.write_text(json.dumps(payload), encoding="utf-8")
            command.extend([f"--{side}", str(summary)])
    subprocess.run(command, capture_output=True, check=True, timeout=10)
    return markdown.read_text(encoding="utf-8"), json.loads(report_json.read_text(encoding="utf-8"))


def test_single_run_per_side_has_no_measurement_order_claim(tmp_path: Path) -> None:
    markdown, report = compare_summaries(tmp_path, [(100, 100.0)], [(100, 110.0)])
    assert "| Samples | 100 baseline, 100 head |" in markdown
    assert "| Repeats | 1 baseline, 1 head |" in markdown
    assert "A-B-B-A" not in markdown
    assert "averaged" not in markdown
    assert (report["base_repeats"], report["head_repeats"]) == (1, 1)
    assert report["qualification"] == "historical"
    assert report["baseline_eligible"] is False
    assert report["rows"][0]["delta_pct"] == pytest.approx(10.0)
    assert report["regressions"] == report["improvements"] == {}
    assert "No regressions" not in markdown


def test_uneven_run_counts_keep_the_median(tmp_path: Path) -> None:
    markdown, report = compare_summaries(
        tmp_path, [(98, 100.0), (99, 110.0), (100, 900.0)], [(100, 101.0)]
    )
    assert "| Samples | 297 baseline, 100 head |" in markdown
    assert "| Repeats | 3 baseline, 1 head |" in markdown
    assert "A-B-B-A" not in markdown
    assert (report["base_repeats"], report["head_repeats"]) == (3, 1)
    assert (report["base_completed"], report["head_completed"]) == (98, 100)
    assert (report["base_total_completed"], report["head_total_completed"]) == (297, 100)
    assert (report["base_sha"], report["head_sha"]) == ("base", "head")
    assert {row["base_ms"] for row in report["rows"]} == {110.0}


@pytest.mark.parametrize(("side", "side_label"), [("base", "Baseline"), ("head", "Head")])  # type: ignore[misc]
def test_mixed_revisions_are_rejected_before_merging(
    tmp_path: Path, side: str, side_label: str
) -> None:
    base: list[tuple[int, float | str]] = [(100, "unusable"), (100, 110.0)]
    head = [(100, 100.0), (100, 110.0)]
    with pytest.raises(subprocess.CalledProcessError) as exc:
        compare_summaries(tmp_path, base, head, git_shas={f"{side}-1.json": "other-sha"})

    assert exc.value.returncode == 2
    error = exc.value.stderr.decode("utf-8")
    assert f"{side_label} summaries use different Git revisions" in error
    assert f"{tmp_path / f'{side}-1.json'} has source.git_sha 'other-sha'" in error
    assert f"{tmp_path / f'{side}-0.json'} has {side!r}" in error
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "report.json").exists()


def test_blank_revision_is_rejected(tmp_path: Path) -> None:
    runs = [(100, 100.0)]
    with pytest.raises(subprocess.CalledProcessError) as exc:
        compare_summaries(tmp_path, runs, runs, git_shas={"head-0.json": " "})

    assert exc.value.returncode == 2
    error = exc.value.stderr.decode("utf-8")
    assert (
        f"Head summary {tmp_path / 'head-0.json'} must provide a nonempty source.git_sha" in error
    )


def offline_summary(side: str, count: int = 100, warm_ms: float = 100.0) -> Summary:
    return {
        "run": {"status": "success"},
        "source": {"git_sha": side, "raw_schema_version": "vllm-gr.offline.raw.v3"},
        "model": {"id": f"/{side}/model", "revision": "1" * 40},
        "dataset": {
            "task": "video",
            "path": f"/{side}/video_test.parquet",
            "sha256": "2" * 64,
            "measured_sha256": "2" * 64,
            "selection": {
                "sample_ids_sha256": "3" * 64,
                "workload_sha256": "4" * 64,
                "warmup_workload_sha256": "5" * 64,
            },
        },
        "environment": {
            "gpu": {"name": "NVIDIA H20", "memory_mib": 97871, "driver_version": "570"},
            "runtime": {"vllm_version": "0.30.0", "torch_version": "2.11.0"},
        },
        "scenario": {
            "execution_mode": "offline",
            "n": 128,
            "input_tokens_target": 1024,
            "num_prompts": count,
            "warmup_requests": 1,
            "max_concurrency": 1,
            "batch_size": 1,
            "server_args": {
                "catalog_path": f"/{side}/catalog.json",
                "catalog_sha256": "6" * 64,
                "constraint_artifact_sha256": "7" * 64,
                "tokenizer_sha256": "8" * 64,
                "enable_prefix_caching": True,
            },
            "benchmark_args": {
                "max_tokens": 5,
                "measurement_mode": "beam-search-v1-e2e",
                "cache_protocol": "paired-reset-then-repeat",
                "phase_definition": {"version": "beam-v1-e2e"},
                "instrumentation": {"canonical_contaminated_by_diagnostic": False},
            },
        },
        "results": {
            "requests": {"completed": count, "failed": 0},
            "samples": {
                "unit": "request",
                "e2el_ms": [100.0] * count,
                "e2el_hit_ms": [warm_ms] * count,
                "cold_cached_tokens": [0] * count,
                "warm_cached_tokens": [960] * count,
                "input_tokens": [1023] * count,
            },
            "latency_ms": {
                metric: {**dict.fromkeys(("mean", "p50", "p90", "p99"), latency), "unit": "ms"}
                for metric, latency in (("e2el", 100.0), ("e2el_hit", warm_ms))
            },
        },
    }


@pytest.mark.parametrize(("warm_ms", "improved"), [(98.0, False), (97.9, True)])  # type: ignore[misc]
def test_verified_measurements_use_content_identity_and_noise_threshold(
    tmp_path: Path, warm_ms: float, improved: bool
) -> None:
    base = offline_summary("base")
    del base["dataset"]["measured_sha256"]
    markdown, report = compare_summaries(
        tmp_path, [base], [offline_summary("head", warm_ms=warm_ms)]
    )
    assert report["qualification"] == "performance"
    assert report["baseline_eligible"] is True
    assert report["qualification_reasons"] == []
    assert ("p50" in report["improvements"].get("e2el_hit", {})) is improved
    assert "No regressions" in markdown


@pytest.mark.parametrize(  # type: ignore[misc]
    ("warm_counts", "reason"),
    [
        ([960] * 99 + [0], "a warm request reused zero prefix cache tokens"),
        (None, "warm prefix cache observations unavailable"),
    ],
)
def test_missing_or_zero_cache_observations_leave_descriptive_deltas(
    tmp_path: Path, warm_counts: list[int] | None, reason: str
) -> None:
    head = offline_summary("head", warm_ms=50.0)
    head["results"]["samples"]["warm_cached_tokens"] = warm_counts
    markdown, report = compare_summaries(tmp_path, [offline_summary("base")], [head])
    assert report["qualification"] == "smoke"
    assert report["regressions"] == report["improvements"] == {}
    assert any(row["delta_pct"] == -50.0 for row in report["rows"])
    assert report["qualification_reasons"] == [reason]
    assert reason in markdown
    assert "No regressions" not in markdown


@pytest.mark.parametrize(  # type: ignore[misc]
    ("base_reuse", "head_reuses"),
    [
        pytest.param([16] * 100, [[1008] * 100], id="unequal-reuse"),
        pytest.param([16, 1008] * 50, [[1008, 16] * 50], id="different-order"),
        pytest.param([16] * 100, [[16] * 100, [1008] * 100], id="later-repeat"),
    ],
)
def test_mismatched_positive_cache_reuse_leaves_descriptive_deltas(
    tmp_path: Path, base_reuse: list[int], head_reuses: list[list[int]]
) -> None:
    base = offline_summary("base")
    base["results"]["samples"]["warm_cached_tokens"] = base_reuse
    heads = []
    for reuse in head_reuses:
        head = offline_summary("head", warm_ms=80.0)
        head["results"]["samples"]["warm_cached_tokens"] = reuse
        heads.append(head)

    markdown, report = compare_summaries(tmp_path, [base], heads)
    reason = "warm prefix cache reuse differs across measurements"
    assert report["qualification"] == "smoke"
    assert report["baseline_eligible"] is False
    assert report["qualification_reasons"] == [reason]
    assert report["head_repeats"] == len(head_reuses)
    assert report["regressions"] == report["improvements"] == {}
    assert any(row["metric"] == "e2el_hit" and row["delta_pct"] == -20.0 for row in report["rows"])
    assert reason in markdown
    assert "No regressions" not in markdown


@pytest.mark.parametrize("pre_fix_summary", [False, True])  # type: ignore[misc]
def test_missing_measured_dataset_digest_leaves_descriptive_deltas(
    tmp_path: Path, pre_fix_summary: bool
) -> None:
    head = offline_summary("head", warm_ms=50.0)
    head["dataset"]["sha256"] = "9" * 64
    if pre_fix_summary:
        del head["dataset"]["measured_sha256"]
        head["run"]["qualification_reasons"] = ["historical or incomplete measurement identity"]
    else:
        head["dataset"]["measured_sha256"] = None

    markdown, report = compare_summaries(tmp_path, [offline_summary("base")], [head])
    assert report["qualification"] == "smoke"
    assert report["baseline_eligible"] is False
    assert report["regressions"] == report["improvements"] == {}
    assert any(row["delta_pct"] == -50.0 for row in report["rows"])
    assert report["qualification_reasons"] == ["dataset.measured_sha256 is unavailable"]
    assert "Measurements are descriptive" in markdown
    assert "No regressions" not in markdown


@pytest.mark.parametrize("sample_unit", ["request", "logical-batch"])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("field", "values", "expected"),
    [
        ("cold_cached_tokens", [0] * 99 + [1025], "cold_cached_tokens[99]"),
        ("warm_cached_tokens", [960] * 99 + [1025], "warm_cached_tokens[99]"),
        ("input_tokens", [], "input_tokens must contain 100"),
    ],
)
def test_invalid_single_request_cache_evidence_prevents_report(
    tmp_path: Path, sample_unit: str, field: str, values: list[int], expected: str
) -> None:
    head = offline_summary("head")
    head["results"]["samples"]["unit"] = sample_unit
    head["results"]["samples"][field] = values
    with pytest.raises(subprocess.CalledProcessError) as exc:
        compare_summaries(tmp_path, [offline_summary("base")], [head])
    assert exc.value.returncode == 2
    assert f"{tmp_path / 'head-0.json'}: {expected}" in exc.value.stderr.decode("utf-8")
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "report.json").exists()


def test_mixed_summary_generations_keep_shared_metrics(tmp_path: Path) -> None:
    markdown, report = compare_summaries(tmp_path, [(100, 110.0)], [offline_summary("head")])
    assert report["qualification"] == "historical"
    assert report["baseline_eligible"] is False
    assert {row["metric"] for row in report["rows"]} == {"e2el"}
    assert "historical summary lacks verified offline measurement evidence" in markdown


def test_old_raw_version_with_null_cache_and_estimated_ids_stays_readable(tmp_path: Path) -> None:
    base = offline_summary("base")
    base["source"]["raw_schema_version"] = "vllm-gr.offline.raw.v2"
    base["dataset"]["selection"] = {"sample_ids_sha256": "0" * 64}
    base["results"]["samples"].update(cold_cached_tokens=None, warm_cached_tokens=None)
    base["results"]["latency_ms"]["e2el"]["p50"] = 90.0

    _, report = compare_summaries(tmp_path, [base], [offline_summary("head")])
    assert report["qualification"] == "historical"
    assert report["baseline_eligible"] is False
    assert {row["metric"] for row in report["rows"]} == {"e2el", "e2el_hit"}


def test_two_samples_qualify_for_smoke_only(tmp_path: Path) -> None:
    _, report = compare_summaries(
        tmp_path, [offline_summary("base", count=2)], [offline_summary("head", count=2)]
    )
    assert report["qualification"] == "smoke"
    assert report["qualification_reasons"] == ["smoke run uses fewer than 100 measured requests"]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("path", "value"),
    [
        ("scenario.server_args.catalog_sha256", "9" * 64),
        ("dataset.measured_sha256", "9" * 64),
        ("dataset.selection.workload_sha256", "9" * 64),
        ("results.samples.e2el_ms", []),
        ("results.samples.e2el_hit_ms", [float("nan")] * 100),
        ("results.latency_ms.e2el.p99", 120.0),
    ],
)
def test_rejects_mismatched_content_or_invalid_new_samples(
    tmp_path: Path, path: str, value: object
) -> None:
    head = offline_summary("head")
    fields = path.split(".")
    parent = head
    for field in fields[:-1]:
        parent = parent[field]
    parent[fields[-1]] = value

    with pytest.raises(subprocess.CalledProcessError) as exc:
        compare_summaries(tmp_path, [offline_summary("base")], [head])
    assert exc.value.returncode == 2
    error = exc.value.stderr.decode("utf-8")
    assert f"{tmp_path / 'head-0.json'}: {path}" in error
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "report.json").exists()
