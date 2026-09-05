#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate OneRec summary results against a versioned constrained golden."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_CONTRACT_FIELDS = (
    "format_version",
    "beam_width",
    "temperature",
    "execution",
    "execution_variant",
    "seed",
    "sample_manifest_sha256",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _mismatch(
    errors: list[str],
    *,
    task: str,
    field: str,
    expected: Any,
    actual: Any,
) -> None:
    if actual != expected:
        errors.append(f"{task}: {field}: expected {expected!r}, got {actual!r}")


def _is_metric_value(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and 0.0 <= numeric <= 1.0


def _metrics_for_mode(metrics: list[str], evaluation_mode: str) -> list[str]:
    if evaluation_mode == "both":
        return metrics
    select_pid = evaluation_mode == "pid"
    return [metric for metric in metrics if metric.startswith("pid_") == select_pid]


def evaluate_constrained_golden(
    golden: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    tasks: Sequence[str] | None = None,
    evaluation_mode: str | None = None,
) -> dict[str, Any]:
    """Return a report whose passed field implements the CI gate policy."""
    if golden.get("version") != 1:
        raise ValueError("constrained golden must use version 1")

    policy = golden.get("policy")
    protocol = golden.get("protocol")
    golden_tasks = golden.get("tasks")
    if not isinstance(policy, Mapping):
        raise ValueError("constrained golden is missing policy")
    if not isinstance(protocol, Mapping):
        raise ValueError("constrained golden is missing protocol")
    if not isinstance(golden_tasks, Mapping):
        raise ValueError("constrained golden is missing tasks")
    if policy.get("comparison") != "candidate_greater_than_or_equal":
        raise ValueError("unsupported constrained golden comparison policy")

    blocking_metrics = policy.get("blocking_metrics")
    diagnostic_metrics = policy.get("diagnostic_metrics", [])
    blocking_absolute_tolerance = policy.get("blocking_absolute_tolerance", 0.0)
    blocking_metric_absolute_tolerances = policy.get("blocking_metric_absolute_tolerances", {})
    if not isinstance(blocking_metrics, list) or not blocking_metrics:
        raise ValueError("constrained golden must select blocking_metrics")
    if not isinstance(diagnostic_metrics, list):
        raise ValueError("diagnostic_metrics must be a list")
    if not _is_metric_value(blocking_absolute_tolerance):
        raise ValueError("blocking_absolute_tolerance must be a finite number in [0, 1]")
    blocking_absolute_tolerance = float(blocking_absolute_tolerance)
    if not isinstance(blocking_metric_absolute_tolerances, Mapping):
        raise ValueError("blocking_metric_absolute_tolerances must be an object")
    if any(not isinstance(metric, str) for metric in blocking_metric_absolute_tolerances):
        raise ValueError("blocking_metric_absolute_tolerances keys must be metric names")
    unknown_tolerance_metrics = sorted(
        set(blocking_metric_absolute_tolerances).difference(blocking_metrics)
    )
    if unknown_tolerance_metrics:
        raise ValueError(
            "blocking_metric_absolute_tolerances contains non-blocking metrics: "
            f"{unknown_tolerance_metrics}"
        )
    resolved_metric_tolerances: dict[str, float] = {}
    for metric, tolerance in blocking_metric_absolute_tolerances.items():
        if not _is_metric_value(tolerance):
            raise ValueError(
                "blocking_metric_absolute_tolerances values must be finite numbers in [0, 1]"
            )
        resolved_metric_tolerances[metric] = float(tolerance)
    resolved_evaluation_mode = evaluation_mode or protocol.get("evaluation_mode")
    if resolved_evaluation_mode not in ("sid", "pid", "both"):
        raise ValueError("evaluation_mode must be 'sid', 'pid', or 'both'")
    blocking_metrics = _metrics_for_mode(blocking_metrics, resolved_evaluation_mode)
    diagnostic_metrics = _metrics_for_mode(diagnostic_metrics, resolved_evaluation_mode)
    if not blocking_metrics:
        raise ValueError(
            f"constrained golden has no blocking metrics for {resolved_evaluation_mode}"
        )
    metric_names = list(dict.fromkeys([*blocking_metrics, *diagnostic_metrics]))

    selected_tasks = list(tasks) if tasks is not None else list(golden_tasks)
    if not selected_tasks:
        raise ValueError("at least one constrained-golden task must be selected")
    unknown_tasks = sorted(set(selected_tasks) - set(golden_tasks))
    if unknown_tasks:
        raise ValueError(f"tasks are absent from constrained golden: {unknown_tasks}")

    errors: list[str] = []
    task_reports: dict[str, Any] = {}
    for task in selected_tasks:
        expected = golden_tasks[task]
        candidate = candidates.get(task)
        if not isinstance(expected, Mapping):
            raise ValueError(f"invalid constrained golden task entry: {task}")
        if not isinstance(candidate, Mapping):
            errors.append(f"{task}: candidate summary is missing")
            continue

        # Runtime identity (model location/revision and prepared-artifact
        # fingerprint) is informative provenance, not a correctness signal.
        # The L20 baseline is intentionally reusable across equivalent local
        # caches. Protocol settings that change test semantics remain gated.
        for field in PROTOCOL_CONTRACT_FIELDS:
            _mismatch(
                errors,
                task=task,
                field=field,
                expected=protocol.get(field),
                actual=candidate.get(field),
            )
        candidate_metrics = candidate.get("metrics")
        if not isinstance(candidate_metrics, Mapping):
            errors.append(f"{task}: candidate metrics are missing")
            continue
        _mismatch(
            errors,
            task=task,
            field="evaluation_mode",
            expected=resolved_evaluation_mode,
            actual=candidate_metrics.get("evaluation_mode"),
        )
        _mismatch(
            errors,
            task=task,
            field="selected_samples",
            expected=protocol.get("selected_samples"),
            actual=candidate.get("selected_samples"),
        )
        _mismatch(
            errors,
            task=task,
            field="metrics.total_samples",
            expected=protocol.get("selected_samples"),
            actual=candidate_metrics.get("total_samples"),
        )
        _mismatch(
            errors,
            task=task,
            field="source_sha256",
            expected=expected.get("source_sha256"),
            actual=candidate.get("source_sha256"),
        )
        _mismatch(
            errors,
            task=task,
            field="constraint",
            expected=expected.get("constraint"),
            actual=candidate.get("constraint"),
        )
        invalid_samples = candidate.get("invalid_samples")
        if isinstance(invalid_samples, bool) or invalid_samples != 0:
            errors.append(f"{task}: invalid_samples: expected integer 0, got {invalid_samples!r}")

        expected_metrics = expected.get("metrics")
        if not isinstance(expected_metrics, Mapping):
            raise ValueError(f"constrained golden metrics are missing for {task}")
        metric_report: dict[str, Any] = {}
        for metric in metric_names:
            baseline_value = expected_metrics.get(metric)
            candidate_value = candidate_metrics.get(metric)
            if not _is_metric_value(baseline_value):
                raise ValueError(f"invalid constrained golden metric: {task}/{metric}")
            if not _is_metric_value(candidate_value):
                errors.append(
                    f"{task}: candidate metric must be a finite number in [0, 1]: "
                    f"{metric}={candidate_value!r}"
                )
                continue
            baseline_float = float(baseline_value)
            candidate_float = float(candidate_value)
            delta = candidate_float - baseline_float
            is_blocking = metric in blocking_metrics
            metric_tolerance = (
                resolved_metric_tolerances.get(metric, blocking_absolute_tolerance)
                if is_blocking
                else 0.0
            )
            regressed = candidate_float < baseline_float - metric_tolerance - 1e-12
            metric_report[metric] = {
                "baseline": baseline_float,
                "candidate": candidate_float,
                "delta": delta,
                "blocking": is_blocking,
                "absolute_tolerance": metric_tolerance,
                "regressed": regressed,
            }
            if is_blocking and regressed:
                errors.append(
                    f"{task}: blocking metric {metric} regressed beyond "
                    f"absolute tolerance {metric_tolerance}: "
                    f"{baseline_float} -> {candidate_float}"
                )
        task_reports[task] = {"metrics": metric_report}

    aggregate_metrics: dict[str, Any] = {}
    for metric in metric_names:
        task_values = [
            task_reports[task]["metrics"].get(metric)
            for task in selected_tasks
            if task in task_reports
        ]
        if len(task_values) != len(selected_tasks) or any(values is None for values in task_values):
            continue
        baseline = math.fsum(values["baseline"] for values in task_values) / len(task_values)
        candidate = math.fsum(values["candidate"] for values in task_values) / len(task_values)
        aggregate_metrics[metric] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": candidate - baseline,
            "blocking": metric in blocking_metrics,
        }

    return {
        "passed": not errors,
        "golden_description": golden.get("description"),
        "selected_tasks": selected_tasks,
        "evaluation_mode": resolved_evaluation_mode,
        "blocking_metrics": blocking_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "blocking_absolute_tolerance": blocking_absolute_tolerance,
        "blocking_metric_absolute_tolerances": {
            metric: tolerance
            for metric, tolerance in resolved_metric_tolerances.items()
            if metric in blocking_metrics
        },
        "errors": errors,
        "tasks": task_reports,
        "aggregate_metrics": aggregate_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--evaluation-mode", choices=["sid", "pid", "both"])
    args = parser.parse_args()

    golden = _read_json(args.golden)
    task_names = args.tasks or list(golden.get("tasks", {}))
    candidate_dir = Path(args.candidate_dir)
    candidates = {
        task: _read_json(candidate_dir / f"{task}.summary.json")
        for task in task_names
        if (candidate_dir / f"{task}.summary.json").is_file()
    }
    report = evaluate_constrained_golden(
        golden,
        candidates,
        tasks=task_names,
        evaluation_mode=args.evaluation_mode,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if report["passed"]:
        return 0
    for error in report["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
