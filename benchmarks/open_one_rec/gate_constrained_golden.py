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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configuration contract: the minimal set of fields that define WHAT a run
# measures. A golden is intentionally reusable across a wide range of runtime
# parameters (max_model_len, batch size, gpu memory, VLLM_* env, ...) and the
# code under test; those are recorded as provenance but never fail the gate.
PROTOCOL_CONTRACT_FIELDS = (
    "sample_manifest_sha256",
    "beam_width",
    "temperature",
)

# Canonical five-task gate policy (kept identical across captured goldens so
# that offline and online baselines compare under the same rules).
CAPTURE_POLICY: dict[str, Any] = {
    "comparison": "candidate_greater_than_or_equal",
    "blocking_absolute_tolerance": 0.02,
    "blocking_metric_absolute_tolerances": {
        "pass@1": 0.03,
        "pid_pass@1": 0.03,
    },
    "blocking_metrics": [
        "pass@1",
        "pass@32",
        "recall@32",
        "pid_pass@1",
        "pid_pass@32",
        "pid_recall@32",
    ],
    "diagnostic_metrics": [
        "position1_pass@1",
        "position1_pass@32",
        "recall@1",
        "pid_position1_pass@1",
        "pid_position1_pass@32",
        "pid_recall@1",
    ],
    # The per-sample paired tests are the primary metric comparison; each metric
    # family is judged against this per-family p-value threshold. The aggregate
    # tolerance band above is only a fallback for goldens captured without
    # per-sample data.
    "per_sample": {"fdr_q": 0.05},
}

CAPTURE_PROTOCOL_FIELDS = (
    "beam_width",
    "temperature",
    "seed",
    "execution",
    "execution_variant",
    "sample_manifest_sha256",
    "prepared_manifest_sha256",
    "implementation_revision",
    "max_model_len",
    "config_env",
)


def capture_constrained_golden(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    description: str,
    model_label: str | None = None,
    revision_label: str | None = None,
    samples: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a golden manifest from one clean run's task summaries.

    Every summary must agree on the capture protocol fields; a mixed run
    (two engines, two manifests, two sample sets) is rejected instead of
    being silently averaged into one baseline.

    When per-sample metrics are provided (``samples``: task -> sample id ->
    metrics), they are frozen into the golden so the gate can run per-sample
    paired statistics against the candidate.
    """
    if not summaries:
        raise ValueError("capture needs at least one task summary")
    first = next(iter(summaries.values()))
    for field in CAPTURE_PROTOCOL_FIELDS:
        # config_env is a dict, so compare a canonical hashable form instead of
        # the raw value.
        distinct = {
            json.dumps(summary.get(field), sort_keys=True): summary.get(field)
            for summary in summaries.values()
        }
        if len(distinct) != 1:
            raise ValueError(
                f"capture protocol field {field!r} differs across summaries: "
                f"{sorted(map(repr, distinct.values()))}"
            )
    metrics_blocks: dict[str, Any] = {}
    modes: set[str] = set()
    for task, summary in summaries.items():
        metrics = summary.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"summary for task {task!r} is missing metrics")
        mode = metrics.get("evaluation_mode")
        if mode not in ("sid", "pid", "both"):
            raise ValueError(f"summary for task {task!r} has an invalid evaluation_mode")
        modes.add(mode)
        if summary.get("invalid_samples") != 0:
            raise ValueError(
                f"capture requires zero invalid samples, task {task!r} has "
                f"{summary.get('invalid_samples')!r}"
            )
        capture_metrics = _metrics_for_mode(
            [*CAPTURE_POLICY["blocking_metrics"], *CAPTURE_POLICY["diagnostic_metrics"]],
            mode,
        )
        metrics_blocks[task] = {metric: metrics.get(metric) for metric in capture_metrics}
        for metric, value in metrics_blocks[task].items():
            if not _is_metric_value(value):
                raise ValueError(f"task {task!r} metric {metric} is not a finite value in [0, 1]")
    if len(modes) != 1:
        raise ValueError(f"capture summaries disagree on evaluation_mode: {sorted(modes)}")
    metrics_mode = next(iter(modes))
    protocol: dict[str, Any] = {
        "format_version": 1,
        "model": model_label or first.get("model"),
        "revision": revision_label if revision_label is not None else first.get("revision"),
        "beam_width": first.get("beam_width"),
        "temperature": first.get("temperature"),
        "evaluation_mode": metrics_mode,
        "execution": first.get("execution"),
        "execution_variant": first.get("execution_variant"),
        "seed": first.get("seed"),
        "sample_manifest_sha256": first.get("sample_manifest_sha256"),
        "prepared_manifest_sha256": first.get("prepared_manifest_sha256"),
        "implementation_revision": first.get("implementation_revision"),
        "max_model_len": first.get("max_model_len"),
        "config_env": first.get("config_env"),
    }
    task_blocks: dict[str, Any] = {}
    for task, summary in summaries.items():
        block: dict[str, Any] = {
            "source_sha256": summary.get("source_sha256"),
            "constraint": summary.get("constraint"),
            "selected_samples": summary.get("selected_samples"),
            "metrics": metrics_blocks[task],
        }
        if samples is not None:
            task_samples = samples.get(task)
            if not isinstance(task_samples, Mapping):
                raise ValueError(f"capture per-sample data is missing for task {task!r}")
            selected = summary.get("selected_samples")
            if selected is not None and len(task_samples) != selected:
                raise ValueError(
                    f"capture per-sample data for task {task!r} has {len(task_samples)} "
                    f"samples but the summary selected {selected}"
                )
            block["samples"] = {str(key): value for key, value in task_samples.items()}
        task_blocks[task] = block
    return {
        "version": 1,
        "description": description,
        "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "policy": deepcopy(CAPTURE_POLICY),
        "protocol": protocol,
        "tasks": task_blocks,
    }


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


def _project_sample_metrics(metrics: Mapping[str, Any], evaluation_mode: str) -> dict[str, Any]:
    """Keep only the metric families the selected evaluation mode produces."""
    if evaluation_mode == "both":
        return dict(metrics)
    keep_pid = evaluation_mode == "pid"
    return {name: value for name, value in metrics.items() if name.startswith("pid_") == keep_pid}


def _load_sample_metrics(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load per-sample business metrics from a decode run's jsonl."""
    result: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            metrics = record.get("metrics")
            if sample_id is None or not isinstance(metrics, Mapping):
                raise ValueError(f"decode jsonl record is missing sample_id/metrics: {path}")
            key = str(sample_id)
            if key in result:
                raise ValueError(f"decode jsonl contains duplicate sample_id {key}: {path}")
            result[key] = dict(metrics)
    return result


# --- Per-sample paired statistics (the primary metric comparison) ------------
#
# The gate pairs the candidate and the golden on the SAME sample id and tests
# each metric FAMILY with a one-sided paired sign test on a per-sample family
# scalar (see _family_scalar): binary families (pass@k / position1_pass@k and
# their pid_ variants) collapse to the first-hit depth; continuous families
# (recall@k / pid_recall@k) collapse to their sum. Family-level testing keeps a
# regression that only moves the middle of the k range visible (the {1,32}
# endpoints miss a depth shift such as 10 -> 20) and keeps the hypothesis count
# at one per family instead of one per nested metric@k.
# Benign configuration noise flips samples in both directions, so the paired
# balance stays non-significant; a real regression is one-sided and stands out.
# Each family is judged against the same per-family p-value threshold. No
# multiple-testing correction is applied because the families are strongly
# dependent (same samples; pid_ variants re-score the SID ones).

DEFAULT_PER_SAMPLE_FDR = 0.05
_METRIC_EPSILON = 1e-9


def _metric_family(metric: str) -> str:
    return metric.split("@", 1)[0]


def _binomial_upper_tail(observed: int, trials: int) -> float:
    """P(X >= observed) for X ~ Binomial(trials, 0.5)."""
    if trials <= 0:
        return 1.0
    return math.fsum(math.comb(trials, rank) for rank in range(observed, trials + 1)) / (2**trials)


def _family_scalar(metrics: Mapping[str, Any], family: str) -> float | None:
    """Collapse one sample's metrics within ``family`` across k into a scalar.

    Binary families (``pass``/``position1_pass`` and their pid_ variants) are
    monotone in k, so the count of satisfied metrics is the first-hit depth;
    continuous families (``recall``/``pid_recall``) are summed across k. Testing
    the family scalar (instead of every ``metric@k``) keeps a regression that
    only moves the middle of the k range visible — the ``{1,32}`` endpoints miss
    a depth shift such as 10 -> 20 — and shrinks the hypothesis
    count from ~192 nested k-tests to one per family.
    """
    values = [value for name, value in metrics.items() if _metric_family(name) == family]
    if not values:
        return None
    total = 0.0
    for value in values:
        if isinstance(value, bool):
            total += 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            total += float(value)
    return total


def _paired_scalar_test(
    golden_values: Sequence[float],
    candidate_values: Sequence[float],
) -> tuple[float, int, int]:
    """One-sided sign test on per-sample family scalars.

    Returns ``(p, n_worse, n_better)``; ``n_worse`` counts samples whose scalar
    dropped and ``n_better`` those that rose. The p-value is the probability of
    at least ``n_worse`` drops out of the discordant pairs under the no-change
    null, so benign bidirectional noise stays non-significant.
    """
    n_worse = 0
    n_better = 0
    for golden_value, candidate_value in zip(golden_values, candidate_values):
        delta = float(candidate_value) - float(golden_value)
        if delta < -_METRIC_EPSILON:
            n_worse += 1
        elif delta > _METRIC_EPSILON:
            n_better += 1
    return _binomial_upper_tail(n_worse, n_worse + n_better), n_worse, n_better


def _per_sample_stats(
    golden_samples: Mapping[str, Any],
    candidate_samples: Mapping[str, Any],
    *,
    fdr_q: float,
) -> tuple[dict[str, Any], list[str]]:
    """Paired per-sample tests over metric FAMILIES.

    Each family (``pass``/``position1_pass``/``recall`` and their pid_ variants)
    is one hypothesis: the candidate and golden are compared on a per-sample
    family scalar (see ``_family_scalar``). Every family is judged against the
    same per-family threshold ``fdr_q``.

    No multiple-testing correction is applied: the ~6 families are computed from
    the SAME samples and the pid_ variants are a re-scoring of the SID ones, so
    they are strongly dependent (the effective number of independent tests is
    far below 6). A Benjamini-Hochberg step would divide the first-ranked
    threshold by the family count and make a real regression harder to flag
    without buying meaningful false-positive control.
    """
    common = sorted(set(golden_samples) & set(candidate_samples))
    families: set[str] = set()
    for sample_id in common:
        golden_metrics = golden_samples[sample_id]
        candidate_metrics = candidate_samples[sample_id]
        if isinstance(golden_metrics, Mapping) and isinstance(candidate_metrics, Mapping):
            families.update(
                _metric_family(metric) for metric in set(golden_metrics) & set(candidate_metrics)
            )
    tests: list[dict[str, Any]] = []
    per_case: dict[str, dict[str, str]] = {}
    for sample_id in common:
        per_case[sample_id] = {}
    for family in sorted(families):
        golden_values: list[float] = []
        candidate_values: list[float] = []
        for sample_id in common:
            golden_scalar = _family_scalar(golden_samples[sample_id], family)
            candidate_scalar = _family_scalar(candidate_samples[sample_id], family)
            if golden_scalar is None or candidate_scalar is None:
                continue
            delta = float(candidate_scalar) - float(golden_scalar)
            if delta < -_METRIC_EPSILON:
                label = "worse"
            elif delta > _METRIC_EPSILON:
                label = "better"
            else:
                label = "same"
            per_case[sample_id][family] = label
            golden_values.append(golden_scalar)
            candidate_values.append(candidate_scalar)
        p_value, n_worse, n_better = _paired_scalar_test(golden_values, candidate_values)
        tests.append(
            {
                "family": family,
                "p_value": p_value,
                "n_worse": n_worse,
                "n_better": n_better,
                "n_same": len(golden_values) - n_worse - n_better,
            }
        )
    report: dict[str, Any] = {
        "common_samples": len(common),
        "families_tested": len(tests),
        "fdr_q": fdr_q,
        "significant_regressions": [],
        "tests": tests,
        "per_case": per_case,
    }
    errors: list[str] = []
    for test in tests:
        if test["p_value"] > fdr_q or test["n_worse"] <= test["n_better"]:
            continue
        report["significant_regressions"].append(test["family"])
        errors.append(
            f"per-sample metric family {test['family']} regressed: "
            f"{test['n_worse']} worse vs {test['n_better']} better "
            f"(p={test['p_value']:.3g})"
        )
    return report, errors


def _validate_paired_samples(
    golden_samples: Mapping[str, Any],
    candidate_samples: Mapping[str, Any],
    *,
    task: str,
    expected_count: Any,
) -> None:
    """Reject incomplete or mismatched per-sample data before testing.

    The paired sign test is only meaningful when both sides cover the SAME
    complete sample set with the SAME finite metric schema. Silently
    intersecting the two sides (or accepting an empty mapping) would turn
    truncated, missing, or mismatched runs into a vacuous pass.
    """
    if not golden_samples or not candidate_samples:
        raise ValueError(f"{task}: per-sample metrics are empty")
    golden_ids = set(golden_samples)
    candidate_ids = set(candidate_samples)
    if golden_ids != candidate_ids:
        missing = sorted(golden_ids - candidate_ids)
        unexpected = sorted(candidate_ids - golden_ids)
        raise ValueError(
            f"{task}: per-sample sample ids differ between golden and candidate "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if isinstance(expected_count, int) and len(golden_ids) != expected_count:
        raise ValueError(
            f"{task}: per-sample sample count {len(golden_ids)} does not match "
            f"the declared {expected_count}"
        )
    for sample_id in sorted(golden_ids):
        golden_metrics = golden_samples[sample_id]
        candidate_metrics = candidate_samples[sample_id]
        if not isinstance(golden_metrics, Mapping) or not isinstance(candidate_metrics, Mapping):
            raise ValueError(f"{task}: sample {sample_id} is missing per-sample metrics")
        if set(golden_metrics) != set(candidate_metrics):
            raise ValueError(
                f"{task}: sample {sample_id} metric schema differs between golden and candidate"
            )
        for metric, value in candidate_metrics.items():
            if isinstance(value, bool):
                continue
            if not _is_metric_value(value):
                raise ValueError(
                    f"{task}: sample {sample_id} metric {metric} is not a finite value in [0, 1]"
                )


def evaluate_constrained_golden(
    golden: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    tasks: Sequence[str] | None = None,
    evaluation_mode: str | None = None,
    candidate_samples: Mapping[str, Mapping[str, Any]] | None = None,
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
    per_sample_policy = policy.get("per_sample")
    per_sample_fdr = DEFAULT_PER_SAMPLE_FDR
    if isinstance(per_sample_policy, Mapping) and "fdr_q" in per_sample_policy:
        fdr_value = per_sample_policy.get("fdr_q")
        if not _is_metric_value(fdr_value):
            raise ValueError("per_sample.fdr_q must be a finite number in [0, 1]")
        per_sample_fdr = float(fdr_value)
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

        # Only the minimal configuration-contract fields define what the run measures.
        # Everything else (model location, code revision, runtime parameters,
        # ...) is recorded but not gated. The per-task source digest and
        # constraint domain are gated below.
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
        # Per-task sample count (goldens captured with per-task counts store it
        # in the task block; older goldens only have the protocol-level value).
        golden_count = expected.get("selected_samples", protocol.get("selected_samples"))
        _mismatch(
            errors,
            task=task,
            field="selected_samples",
            expected=golden_count,
            actual=candidate.get("selected_samples"),
        )
        _mismatch(
            errors,
            task=task,
            field="metrics.total_samples",
            expected=golden_count,
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
        golden_task_samples = expected.get("samples")
        candidate_task_samples = (
            candidate_samples.get(task) if candidate_samples is not None else None
        )
        # A golden that froze per-sample metrics makes the paired tests the
        # metric verdict, so the candidate MUST ship the same per-sample data;
        # a missing/truncated candidate jsonl is an error, never a silent fall
        # back to the aggregate band (which would hide a lost task). The
        # aggregate band is only a fallback for goldens captured before
        # per-sample metrics existed.
        golden_has_samples = isinstance(golden_task_samples, Mapping) and bool(golden_task_samples)
        if golden_has_samples and not isinstance(candidate_task_samples, Mapping):
            errors.append(f"{task}: candidate per-sample metrics are missing")
            use_per_sample = False
        else:
            use_per_sample = golden_has_samples
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
            if is_blocking and regressed and not use_per_sample:
                errors.append(
                    f"{task}: blocking metric {metric} regressed beyond "
                    f"absolute tolerance {metric_tolerance}: "
                    f"{baseline_float} -> {candidate_float}"
                )
        task_reports[task] = {"metrics": metric_report}
        if use_per_sample:
            # A golden may have been captured with evaluation_mode="both"; a
            # SID-only (or PID-only) candidate is then still comparable. Project
            # both sides to the selected mode BEFORE the schema check and the
            # statistics so the extra family does not look like a schema
            # mismatch.
            projected_golden = {
                sample_id: _project_sample_metrics(metrics, resolved_evaluation_mode)
                for sample_id, metrics in golden_task_samples.items()
            }
            projected_candidate = {
                sample_id: _project_sample_metrics(metrics, resolved_evaluation_mode)
                for sample_id, metrics in candidate_task_samples.items()
            }
            _validate_paired_samples(
                projected_golden,
                projected_candidate,
                task=task,
                expected_count=golden_count,
            )
            stats_report, stats_errors = _per_sample_stats(
                projected_golden,
                projected_candidate,
                fdr_q=per_sample_fdr,
            )
            task_reports[task]["per_sample"] = stats_report
            errors.extend(stats_errors)

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
    parser.add_argument("--golden")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--evaluation-mode", choices=["sid", "pid", "both"])
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Build a golden manifest from the candidate summaries instead of gating",
    )
    parser.add_argument(
        "--capture-description",
        default="Captured five-task Constraint Tables golden.",
    )
    parser.add_argument(
        "--capture-model",
        help="Stable model label recorded in the manifest protocol "
        "(defaults to the summary's model path)",
    )
    parser.add_argument(
        "--capture-revision",
        help="Model revision label recorded in the manifest protocol "
        "(defaults to the summary's revision)",
    )
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    if args.capture:
        task_names = args.tasks or sorted(
            path.name[: -len(".summary.json")] for path in candidate_dir.glob("*.summary.json")
        )
        if not task_names:
            print("No task summaries found in the candidate directory.", file=sys.stderr)
            return 1
        summaries = {}
        samples = {}
        for task in task_names:
            summary_path = candidate_dir / f"{task}.summary.json"
            if not summary_path.is_file():
                print(f"Capture summary is missing for task: {task}", file=sys.stderr)
                return 1
            summaries[task] = _read_json(summary_path)
            jsonl_path = candidate_dir / f"{task}.jsonl"
            if jsonl_path.is_file():
                samples[task] = _load_sample_metrics(jsonl_path)
        manifest = capture_constrained_golden(
            summaries,
            description=args.capture_description,
            model_label=args.capture_model,
            revision_label=args.capture_revision,
            samples=samples or None,
        )
        rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Captured golden written to {output}")
        else:
            print(rendered)
        return 0

    if not args.golden:
        parser.error("--golden is required unless --capture is used")
    golden = _read_json(args.golden)
    task_names = args.tasks or list(golden.get("tasks", {}))
    candidates = {
        task: _read_json(candidate_dir / f"{task}.summary.json")
        for task in task_names
        if (candidate_dir / f"{task}.summary.json").is_file()
    }
    candidate_samples = {
        task: _load_sample_metrics(candidate_dir / f"{task}.jsonl")
        for task in task_names
        if (candidate_dir / f"{task}.jsonl").is_file()
    }
    report = evaluate_constrained_golden(
        golden,
        candidates,
        tasks=task_names,
        evaluation_mode=args.evaluation_mode,
        candidate_samples=candidate_samples or None,
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
