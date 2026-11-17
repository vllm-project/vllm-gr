# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compare two complete Beam V1 timing runs from the same CI runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

COUNTS = (1, 2, 4)
MAX_OFFLINE_REGRESSION_PERCENT = 1.0
GATED_METRICS = ("mean", "p50", "p90")
THRESHOLD_ROUNDING_TOLERANCE = 1e-12
SETUP_KEYS = (
    "benchmark_driver_sha256",
    "model",
    "revision",
    "input_tokens_including_sid_begin",
    "beam_width",
    "generated_steps",
    "batch_sizes",
    "warmup_runs",
    "measured_runs",
    "max_model_len",
    "gpu_memory_utilization",
    "artifact_digest",
    "tokenizer_digest",
    "graph_enabled",
    "prefix_caching",
    "chunked_prefill",
)
RUNTIME_IDENTITY_KEYS = ("artifact_digest", "tokenizer_digest")
CROSS_RUNTIME_SETUP_KEYS = tuple(key for key in SETUP_KEYS if key not in RUNTIME_IDENTITY_KEYS)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def artifact_workload_digest(root: Path, setup: dict[str, Any]) -> str:
    """Hash the constraint-table workload while ignoring tokenizer identity.

    A vLLM upgrade can change the tokenizer's serialized state without changing
    the token IDs used by this fixed workload. The artifact embeds that runtime
    tokenizer digest in its safetensors metadata, so its whole-file digest also
    changes. Validate both runtime identities locally, then compare the stable
    metadata and tensor payload across the A/B environments.
    """
    from safetensors import SafetensorError, safe_open

    path = root / "constraint_table.safetensors"
    try:
        with path.open("rb") as stream:
            actual_digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError as exc:
        raise ValueError(f"missing constraint_table.safetensors in {root}") from exc
    expected_digest = setup.get("artifact_digest")
    if expected_digest != actual_digest:
        raise ValueError(f"artifact digest mismatch in {root}")
    workload = hashlib.sha256()
    try:
        # Reuse the format reader's dtype, shape and offset validation. Load
        # one tensor at a time; the whole-file digest above streams the bytes.
        with safe_open(path, framework="numpy") as artifact:
            metadata = artifact.metadata() or {}
            if metadata.get("tokenizer_digest") != setup.get("tokenizer_digest"):
                raise ValueError(f"tokenizer digest mismatch in {root}")
            stable_metadata = {
                key: value for key, value in metadata.items() if key != "tokenizer_digest"
            }
            workload.update(
                json.dumps(stable_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            for name in sorted(artifact.keys()):
                view = artifact.get_slice(name)
                tensor = artifact.get_tensor(name)
                descriptor = {
                    "name": name,
                    "dtype": view.get_dtype(),
                    "shape": view.get_shape(),
                    "size": tensor.nbytes,
                }
                workload.update(
                    json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
                )
                workload.update(tensor.tobytes())
    except SafetensorError as exc:
        raise ValueError(f"invalid safetensors artifact in {root}: {exc}") from exc
    return "sha256:" + workload.hexdigest()


def measured_cases(
    root: Path, mode: str, measured_runs: int, input_len: int
) -> dict[int, dict[str, Any]]:
    data = read_json(root / f"{mode}.json")
    if data.get("mode") != mode or data.get("complete") is not True:
        raise ValueError(f"incomplete {mode} result in {root}")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != len(COUNTS):
        raise ValueError(f"wrong {mode} case count in {root}")
    by_count = {}
    for case in cases:
        count = case.get("batch_size" if mode == "offline" else "concurrency")
        if case.get("mode") != mode or count not in COUNTS or count in by_count:
            raise ValueError(f"unexpected or duplicate {mode} B/C={count} in {root}")
        samples = case.get("group_samples_ms")
        if (
            not isinstance(samples, list)
            or len(samples) != measured_runs
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in samples
            )
        ):
            raise ValueError(f"invalid {mode} B/C={count} timing samples in {root}")
        mean = case["group_latency_ms"]["mean_ms"]
        if not math.isfinite(mean) or not math.isclose(
            mean, statistics.fmean(samples), rel_tol=1e-9
        ):
            raise ValueError(f"inconsistent {mode} B/C={count} mean in {root}")
        if case.get("prompt_tokens_including_sid_begin") != [input_len] * count:
            raise ValueError(f"wrong {mode} B/C={count} prompt length in {root}")
        by_count[count] = case
    return by_count


def compare(base_root: Path, pr_root: Path, base_sha: str, pr_sha: str) -> tuple[dict, str]:
    base_roots = (base_root / "1", base_root / "2")
    pr_roots = (pr_root / "1", pr_root / "2")
    runs = (
        (base_roots[0], base_sha),
        (pr_roots[0], pr_sha),
        (pr_roots[1], pr_sha),
        (base_roots[1], base_sha),
    )
    for root, expected_sha in runs:
        if not (root / "commit.txt").is_file():
            raise ValueError(f"missing A/B pass in {root}")
        if (root / "commit.txt").read_text(encoding="utf-8").strip() != expected_sha:
            raise ValueError(f"commit mismatch in {root}")
    setups = {root: read_json(root / "setup.json") for root, _ in runs}
    base_setup = setups[base_roots[0]]
    for root, _ in runs[1:]:
        setup = setups[root]
        if any(base_setup.get(key) != setup.get(key) for key in CROSS_RUNTIME_SETUP_KEYS):
            raise ValueError(f"benchmark settings differ in {root}")
    for side_roots in (base_roots, pr_roots):
        first_setup = setups[side_roots[0]]
        if any(
            first_setup.get(key) != setups[side_roots[1]].get(key) for key in RUNTIME_IDENTITY_KEYS
        ):
            raise ValueError(f"runtime identity changed between A/B passes in {side_roots[1]}")
    if (
        base_setup.get("model") != "OpenOneRec/OneRec-1.7B"
        or base_setup.get("revision") != "cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b"
        or base_setup.get("input_tokens_including_sid_begin") != 1024
        or base_setup.get("beam_width") != 128
        or base_setup.get("generated_steps") != 3
        or base_setup.get("batch_sizes") != list(COUNTS)
        or base_setup.get("warmup_runs") != 10
        or base_setup.get("measured_runs") != 20
    ):
        raise ValueError("A/B gate requires the fixed OneRec B=1,2,4 workload")
    for name in ("gpu.csv", "prompt_token_ids.json"):
        base_file = base_roots[0] / name
        if name == "prompt_token_ids.json" and not base_file.is_file():
            raise ValueError("missing prompt_token_ids.json")
        for root, _ in runs[1:]:
            file = root / name
            if base_file.exists() != file.exists():
                raise ValueError(f"inconsistent {name} across A/B runs")
            if base_file.exists() and base_file.read_bytes() != file.read_bytes():
                raise ValueError(f"A/B runs used different {name}")
    artifact_workloads = {artifact_workload_digest(root, setups[root]) for root, _ in runs}
    if len(artifact_workloads) != 1:
        raise ValueError("A/B runs used different constraint table contents")
    constraint_workload_digest = next(iter(artifact_workloads))

    rows = []
    failures = []
    base_cases = [
        measured_cases(
            root,
            "offline",
            base_setup["measured_runs"],
            base_setup["input_tokens_including_sid_begin"],
        )
        for root in base_roots
    ]
    pr_cases = [
        measured_cases(
            root,
            "offline",
            base_setup["measured_runs"],
            base_setup["input_tokens_including_sid_begin"],
        )
        for root in pr_roots
    ]
    for count in COUNTS:
        base_samples = [
            sample for cases in base_cases for sample in cases[count]["group_samples_ms"]
        ]
        pr_samples = [sample for cases in pr_cases for sample in cases[count]["group_samples_ms"]]
        base_ms = statistics.fmean(base_samples)
        pr_ms = statistics.fmean(pr_samples)
        change_percent = (pr_ms / base_ms - 1) * 100
        base_p50, pr_p50 = statistics.median(base_samples), statistics.median(pr_samples)
        base_p90 = statistics.quantiles(base_samples, n=10, method="inclusive")[8]
        pr_p90 = statistics.quantiles(pr_samples, n=10, method="inclusive")[8]
        p50_change_percent = (pr_p50 / base_p50 - 1) * 100
        p90_change_percent = (pr_p90 / base_p90 - 1) * 100
        metric_changes = {
            "mean": change_percent,
            "p50": p50_change_percent,
            "p90": p90_change_percent,
        }
        metric_statuses = {
            metric: (
                "FAIL"
                if metric_changes[metric] > MAX_OFFLINE_REGRESSION_PERCENT
                and not math.isclose(
                    metric_changes[metric],
                    MAX_OFFLINE_REGRESSION_PERCENT,
                    rel_tol=0,
                    abs_tol=THRESHOLD_ROUNDING_TOLERANCE,
                )
                else "PASS"
            )
            for metric in GATED_METRICS
        }
        status = "FAIL" if "FAIL" in metric_statuses.values() else "PASS"
        rows.append(
            {
                "mode": "offline",
                "count": count,
                "base_mean_ms": base_ms,
                "pr_mean_ms": pr_ms,
                "change_percent": change_percent,
                "base_p50_ms": base_p50,
                "pr_p50_ms": pr_p50,
                "p50_change_percent": p50_change_percent,
                "base_p90_ms": base_p90,
                "pr_p90_ms": pr_p90,
                "p90_change_percent": p90_change_percent,
                "base_cv_percent": statistics.pstdev(base_samples) / base_ms * 100,
                "pr_cv_percent": statistics.pstdev(pr_samples) / pr_ms * 100,
                "metric_statuses": metric_statuses,
                "status": status,
            }
        )
        for metric, metric_status in metric_statuses.items():
            if metric_status == "FAIL":
                failures.append(
                    f"offline B={count} {metric}: {metric_changes[metric]:+.2f}% > "
                    f"+{MAX_OFFLINE_REGRESSION_PERCENT:.2f}%"
                )

    lines = [
        "## Beam V1 performance A/B",
        "",
        f"Base: `{base_sha}`; PR: `{pr_sha}`",
        f"Model: `{base_setup['model']}`; revision: `{base_setup['revision']}`; "
        f"input tokens: {base_setup['input_tokens_including_sid_begin']}; "
        f"beam width: {base_setup['beam_width']}; generated steps: {base_setup['generated_steps']}; "
        f"order: A→B→B→A; warmup: {base_setup['warmup_runs']} per pass; "
        f"measured: {base_setup['measured_runs']} per pass "
        f"({2 * base_setup['measured_runs']} per side)",
        "",
        "Group latency is end-to-end caller time in milliseconds. Positive change means slower. "
        f"Offline mean, p50, or p90 latency regression above "
        f"{MAX_OFFLINE_REGRESSION_PERCENT:.2f}% fails the gate.",
        "CV is diagnostic only; CV = population standard deviation / mean.",
        f"Prompt token IDs and normalized constraint tensors matched across all four passes. "
        f"Constraint workload digest: `{constraint_workload_digest}`. Runtime tokenizer and "
        "artifact identities are verified within each side and may differ across vLLM versions.",
        "",
        "| B | Statistic | Base ms | PR ms | Change % | Gate |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        for metric, base_key, pr_key, change_key in (
            ("mean", "base_mean_ms", "pr_mean_ms", "change_percent"),
            ("p50", "base_p50_ms", "pr_p50_ms", "p50_change_percent"),
            ("p90", "base_p90_ms", "pr_p90_ms", "p90_change_percent"),
        ):
            gate = row["metric_statuses"][metric]
            lines.append(
                f"| {row['count']} | {metric} | {row[base_key]:.3f} | "
                f"{row[pr_key]:.3f} | {row[change_key]:+.2f}% | {gate} |"
            )
    lines.extend(("", "| B | Base CV | PR CV |", "| ---: | ---: | ---: |"))
    for row in rows:
        lines.append(
            f"| {row['count']} | {row['base_cv_percent']:.2f}% | {row['pr_cv_percent']:.2f}% |"
        )
    lines.extend(("", "Overall: **FAIL**" if failures else "Overall: **PASS**", ""))
    return (
        {
            "base_sha": base_sha,
            "pr_sha": pr_sha,
            "rows": rows,
            "max_regression_percent": MAX_OFFLINE_REGRESSION_PERCENT,
            "gated_modes": ["offline"],
            "gated_metrics": list(GATED_METRICS),
            "constraint_workload_digest": constraint_workload_digest,
            "runtime_identities": {
                "base": {key: base_setup.get(key) for key in RUNTIME_IDENTITY_KEYS},
                "pr": {key: setups[pr_roots[0]].get(key) for key in RUNTIME_IDENTITY_KEYS},
            },
            "failures": failures,
            "complete": True,
        },
        "\n".join(lines),
    )


def should_retry(result: dict) -> bool:
    """Confirm a regression confined to one batch size with another ABBA run."""
    if result.get("complete") is not True:
        raise ValueError("cannot decide retry from an incomplete A/B report")
    affected_batches = {row["count"] for row in result["rows"] if row["status"] == "FAIL"}
    return len(affected_batches) == 1


def combine_retry_results(
    initial: tuple[dict, str], confirmation: tuple[dict, str]
) -> tuple[dict, str]:
    initial_result, initial_report = initial
    confirmation_result, confirmation_report = confirmation
    if not should_retry(initial_result):
        raise ValueError("A/B confirmation is only allowed for one affected batch size")
    for key in ("base_sha", "pr_sha", "max_regression_percent", "gated_metrics"):
        if initial_result[key] != confirmation_result[key]:
            raise ValueError(f"A/B confirmation differs in {key}")

    passed = not confirmation_result["failures"]
    result = {
        **confirmation_result,
        "retry_performed": True,
        "initial_rows": initial_result["rows"],
        "initial_failures": initial_result["failures"],
    }
    first_section = initial_report.replace(
        "## Beam V1 performance A/B", "### Initial ABBA", 1
    ).replace("Overall: **FAIL**", "Initial: **FAIL**")
    second_section = confirmation_report.replace(
        "## Beam V1 performance A/B", "### Confirmation ABBA", 1
    ).replace(
        "Overall: **PASS**" if passed else "Overall: **FAIL**",
        "Confirmation: **PASS**" if passed else "Confirmation: **FAIL**",
    )
    final = "PASS" if passed else "FAIL"
    report = (
        "## Beam V1 performance A/B\n\n"
        "The initial run exceeded +1%. A second complete ABBA run determines the gate.\n\n"
        f"{first_section.strip()}\n\n{second_section.strip()}\n\n"
        f"Final gate: **{final}**"
        + (" (initial regression was not reproduced)" if passed else "")
        + "\n"
    )
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--pr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--pr-sha", required=True)
    args = parser.parse_args()
    result, report = compare(args.base, args.pr, args.base_sha, args.pr_sha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, end="", flush=True)
    if (path := os.environ.get("GITHUB_STEP_SUMMARY")) and os.environ.get(
        "V1_PERF_DEFER_SUMMARY"
    ) != "1":
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(report)
    if result["failures"]:
        raise SystemExit("Beam V1 A/B gate failed:\n" + "\n".join(result["failures"]))


if __name__ == "__main__":
    main()
