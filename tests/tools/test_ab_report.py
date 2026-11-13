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


def compare_summaries(
    directory: Path,
    base: Sequence[tuple[int, float | str]],
    head: Sequence[tuple[int, float | str]],
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
        for index, (completed, latency) in enumerate(runs):
            summary = directory / f"{side}-{index}.json"
            git_sha = (git_shas or {}).get(summary.name, side)
            summary.write_text(
                json.dumps(
                    {
                        "source": {"git_sha": git_sha},
                        "results": {
                            "requests": {"completed": completed},
                            "latency_ms": {
                                "e2el": dict.fromkeys(("mean", "p50", "p90", "p99"), latency)
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
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


def test_uneven_run_counts_keep_the_median(tmp_path: Path) -> None:
    markdown, report = compare_summaries(
        tmp_path, [(98, 100.0), (99, 110.0), (100, 900.0)], [(100, 101.0)]
    )
    assert "| Samples | 297 baseline, 100 head |" in markdown
    assert "| Repeats | 3 baseline, 1 head |" in markdown
    assert "A-B-B-A" not in markdown
    assert (report["base_repeats"], report["head_repeats"]) == (3, 1)
    assert (report["base_completed"], report["head_completed"]) == (98, 100)
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
