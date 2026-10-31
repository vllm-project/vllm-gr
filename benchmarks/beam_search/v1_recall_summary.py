# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize repeated fixed-sample Beam V1 Recall gates without loading a model."""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

TASKS = ("video", "ad", "product", "label_cond", "interactive")
BASELINE_FAILURE_MODES = frozenset(("blocking", "warning"))
IDENTITY_KEYS = (
    "source_sha256",
    "sample_ids",
    "artifact_digest",
    "tokenizer_digest",
    "interactive_tailcut_count",
    "model_revision",
    "engine_config",
    "prefill_graph_buckets",
    "prefill_graph_capture_num_reqs",
    "v1_startup_capture_batch_size",
)


def _read_seconds(path):
    value = int(path.read_text(encoding="utf-8"))
    if value < 0:
        raise ValueError(f"negative elapsed time in {path}")
    return value


def _task_result(run_dir, task):
    files = list((run_dir / task).glob("*_bs4_1000.json"))
    if len(files) != 1:
        raise ValueError(f"{run_dir.name}/{task}: expected one B=4 1000-sample metrics file")
    metrics = json.loads(files[0].read_text(encoding="utf-8"))
    candidate_checksum = metrics.get("candidate_checksum")
    checksum_schema = metrics.get("candidate_checksum_schema")
    if (
        metrics.get("task") != task
        or metrics.get("total_samples") != 1000
        or metrics.get("batch_size") != 4
        or metrics.get("beam_width") != 128
        or metrics.get("temperature") != 0.6
        or len(metrics.get("sample_ids", [])) != 1000
        or not all(
            isinstance(metrics.get(key), (int, float))
            and math.isfinite(metrics[key])
            and 0 <= metrics[key] <= 1
            for key in ("recall@32", "pid_recall@32")
        )
        or not isinstance(metrics.get("elapsed_time"), (int, float))
        or not math.isfinite(metrics["elapsed_time"])
        or metrics["elapsed_time"] < 0
    ):
        raise ValueError(f"{run_dir.name}/{task}: incomplete or invalid accuracy result")
    if (candidate_checksum is None) != (checksum_schema is None) or (
        candidate_checksum is not None
        and (
            not isinstance(candidate_checksum, str)
            or checksum_schema != "sample-id-and-ordered-decoded-sids-v1"
            or not candidate_checksum.startswith("sha256:")
            or len(candidate_checksum) != len("sha256:") + 64
            or any(character not in "0123456789abcdef" for character in candidate_checksum[7:])
        )
    ):
        raise ValueError(f"{run_dir.name}/{task}: invalid candidate checksum")
    identity = {key: metrics[key] for key in IDENTITY_KEYS if key != "engine_config"}
    identity["engine_config"] = {
        "max_model_len": metrics["engine_config"]["max_model_len"],
        "max_num_batched_tokens": metrics["engine_config"]["max_num_batched_tokens"],
    }
    policy = {
        key: metrics["engine_config"][key]
        for key in (
            "async_scheduling",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "max_num_seqs",
            "max_logprobs",
            "enforce_eager",
            "dtype",
            "seed",
        )
    }
    graph_profile = metrics.get("graph_profile")
    graph_runtime = metrics.get("graph_runtime")
    if (
        not isinstance(graph_profile, dict)
        or set(graph_profile)
        != {
            "prefill_bucket_grid",
            "prefill_capture_num_reqs",
            "prefill_kv_bound",
            "v1_startup_capture_batch_size",
        }
        or not isinstance(graph_runtime, dict)
        or not graph_runtime.get("captures")
        or not graph_runtime.get("effective_profiles")
        or any(not any(width == 4 for _, width in worker) for worker in graph_runtime["captures"])
        or graph_runtime.get("dispatches", {}).get("replay:4", 0)
        != graph_runtime.get("expected_b4_replays", -1)
        or graph_runtime.get("expected_b4_replays", 0) <= 0
        or any(
            key.startswith("fallback:4:") and count
            for key, count in graph_runtime.get("dispatches", {}).items()
        )
    ):
        raise ValueError(f"{run_dir.name}/{task}: B=4 graph execution was not verified")
    return {
        "recall@32": metrics["recall@32"],
        "pid_recall@32": metrics["pid_recall@32"],
        "inference_seconds": metrics["elapsed_time"],
        "task_work_seconds": _read_seconds(run_dir / task / "wall_seconds.txt"),
        "identity": identity,
        "graph_profile": graph_profile,
        "graph_runtime": graph_runtime,
        "engine_policy": policy,
        "runtime_provenance": metrics.get("runtime_provenance"),
        "candidate_checksum": candidate_checksum,
        "candidate_checksum_schema": checksum_schema,
    }


def _stats(values, scale=1):
    return {
        "min": min(values) * scale,
        "median": statistics.median(values) * scale,
        "max": max(values) * scale,
        "mean": statistics.mean(values) * scale,
        "range": (max(values) - min(values)) * scale,
        "sample_stddev": (statistics.stdev(values) if len(values) > 1 else 0.0) * scale,
        "max_abs_deviation_from_median": max(
            abs(value - statistics.median(values)) for value in values
        )
        * scale,
    }


def summarize(root, repeats, baseline_path=None, baseline_failure_mode="blocking"):
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if baseline_failure_mode not in BASELINE_FAILURE_MODES:
        raise ValueError(
            "baseline_failure_mode must be one of: " + ", ".join(sorted(BASELINE_FAILURE_MODES))
        )
    candidate = {
        "schema_version": 1,
        "commit": (root / "commit.txt").read_text(encoding="utf-8").strip(),
        "metric_scale": "fraction",
        "sample_selection": "pandas sample(n=1000, random_state=42) on source row IDs",
        "batch_size": 4,
        "beam_width": 128,
        "temperature": 0.6,
        "sample_count_per_task": 1000,
        "execution_groups": [["video", "ad", "label_cond", "interactive"], ["product"]],
        "graph_profile": None,
        "engine_policy": None,
        "runtime_provenance": None,
        "tasks": {},
    }
    runs = []
    for index in range(1, repeats + 1):
        run_dir = root / "runs" / f"run-{index:02d}"
        run = {
            "number": index,
            "wall_seconds": _read_seconds(run_dir / "total_wall_seconds.txt"),
            "sid_group_wall_seconds": _read_seconds(
                run_dir / "groups" / "sid2pid" / "wall_seconds.txt"
            ),
            "product_group_wall_seconds": _read_seconds(
                run_dir / "groups" / "sid2iid" / "wall_seconds.txt"
            ),
            "tasks": {task: _task_result(run_dir, task) for task in TASKS},
        }
        if index == 1:
            candidate["graph_profile"] = run["tasks"][TASKS[0]]["graph_profile"]
            candidate["engine_policy"] = run["tasks"][TASKS[0]]["engine_policy"]
            candidate["runtime_provenance"] = run["tasks"][TASKS[0]]["runtime_provenance"]
            candidate["tasks"] = {
                task: {
                    "recall@32": run["tasks"][task]["recall@32"],
                    "pid_recall@32": run["tasks"][task]["pid_recall@32"],
                    **run["tasks"][task]["identity"],
                }
                for task in TASKS
            }
        else:
            for task in TASKS:
                for key in IDENTITY_KEYS:
                    if run["tasks"][task]["identity"][key] != candidate["tasks"][task][key]:
                        raise ValueError(f"run {index}/{task}: input differs from run 1: {key}")
        for task in TASKS:
            if (
                run["tasks"][task]["graph_profile"] != candidate["graph_profile"]
                or run["tasks"][task]["engine_policy"] != candidate["engine_policy"]
                or run["tasks"][task]["runtime_provenance"] != candidate["runtime_provenance"]
            ):
                raise ValueError(f"run {index}/{task}: graph, engine, or runtime profile differs")
        runs.append(run)

    variability = {
        task: {
            "sid_recall32_pp": _stats([run["tasks"][task]["recall@32"] for run in runs], scale=100),
            "pid_recall32_pp": _stats(
                [run["tasks"][task]["pid_recall@32"] for run in runs], scale=100
            ),
        }
        for task in TASKS
    }
    candidate_stability = {}
    for task in TASKS:
        checksums = [run["tasks"][task]["candidate_checksum"] for run in runs]
        available = all(checksum is not None for checksum in checksums)
        candidate_stability[task] = {
            "available": available,
            "all_equal": available and len(set(checksums)) == 1,
            "unique_count": len(set(checksums)) if available else None,
        }
    total_times = _stats([run["wall_seconds"] for run in runs])
    failures = []
    baseline = None
    floors = None
    suspected_tasks = []
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (
            baseline.get("schema_version") != 1
            or baseline.get("batch_size") != 4
            or baseline.get("beam_width") != 128
            or baseline.get("temperature") != 0.6
            or baseline.get("sample_count_per_task") != 1000
            or baseline.get("execution_groups") != candidate["execution_groups"]
            or baseline.get("graph_profile") != candidate["graph_profile"]
            or baseline.get("engine_policy") != candidate["engine_policy"]
            or set(baseline.get("tasks", {})) != set(TASKS)
        ):
            raise ValueError("invalid B=4 1000-sample Recall baseline")
        calibration = baseline.get("calibration_sid_recall32")
        if not isinstance(calibration, dict) or set(calibration) != set(TASKS):
            raise ValueError("baseline must contain ten historical SID results per task")
        floors = {}
        for task in TASKS:
            expected = baseline["tasks"][task]
            for key in IDENTITY_KEYS:
                if candidate["tasks"][task][key] != expected.get(key):
                    raise ValueError(f"{task}: baseline input differs: {key}")
            values = calibration[task]
            if (
                not isinstance(values, list)
                or len(values) != 10
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    for value in values
                )
                or values[0] != expected["recall@32"]
            ):
                raise ValueError(f"{task}: invalid historical SID calibration")
            floors[task] = min(values)
            if runs[0]["tasks"][task]["recall@32"] < floors[task]:
                suspected_tasks.append(task)
        if repeats >= 3:
            for task in suspected_tasks:
                if all(run["tasks"][task]["recall@32"] < floors[task] for run in runs[:3]):
                    failures.append(
                        f"{task}: SID Recall@32 below committed baseline "
                        f"{floors[task]:.4%} in all three measurements"
                    )
    confirmation_required = bool(suspected_tasks) and repeats < 3
    baseline_warnings = []
    if baseline_failure_mode == "warning":
        if failures:
            baseline_warnings.extend(failures)
        elif confirmation_required:
            for task in suspected_tasks:
                baseline_warnings.append(
                    f"{task}: SID Recall@32 {runs[0]['tasks'][task]['recall@32']:.4%} "
                    f"is below committed baseline {floors[task]:.4%} in the screening "
                    "measurement; confirmation runs were skipped"
                )
    result = {
        "schema_version": 1,
        "repeat_count": repeats,
        "first_run_baseline_candidate": "b4_baseline_candidate.json",
        "runs": runs,
        "total_wall_seconds": _read_seconds(root / "total_wall_seconds.txt"),
        "per_run_wall_seconds": total_times,
        "task_variability": variability,
        "candidate_checksum_stability": candidate_stability,
        "baseline_file": str(baseline_path) if baseline_path is not None else None,
        "baseline_failure_mode": baseline_failure_mode,
        "baseline_sid_recall32_floors": floors,
        "suspected_tasks": suspected_tasks,
        "confirmation_required": confirmation_required,
        "confirmation_runs": 3 if baseline is not None else None,
        "confirmation_skipped": bool(baseline_warnings and confirmation_required),
        "baseline_failures": failures,
        "baseline_warnings": baseline_warnings,
    }
    lines = [
        f"## Beam V1 B=4 OneRec Recall@32: {repeats} "
        f"{'run' if repeats == 1 else 'runs'} × five tasks × 1000 samples",
        "",
        "Each run starts two fresh engines, uses the same fixed samples, and includes "
        "model initialization, constraint artifact construction and graph capture in "
        "its wall time. Model download and source-catalog generation happen outside "
        "these timings.",
        "",
        (
            "The candidate file records the initial measurement. No Recall floor is configured."
            if baseline_path is None
            else (
                "The committed SID baseline is each task's minimum Recall@32 "
                "from ten calibration measurements on fixed samples. In warning "
                "mode, a below-baseline screening measurement is reported without "
                "running confirmations and does not fail the gate."
                if baseline_failure_mode == "warning"
                else "The committed SID baseline is each task's minimum Recall@32 "
                "from ten calibration measurements on fixed samples. A result below "
                "it triggers two fresh confirmation runs. The gate fails only if "
                "all three measurements for the same task are below that baseline."
            )
        ),
        "",
    ]
    if baseline is not None:
        lines.extend(
            (
                "### Accuracy compared with the committed baseline",
                "",
                "Current values are medians across the available measurements "
                "(one measurement when there is one run). Delta = current - "
                "baseline, in percentage points. Only the SID baseline is gated; "
                "the PID baseline is a fixed reference for comparison.",
                "",
                "| Task | SID baseline | Current SID | Delta SID (pp) | "
                "PID baseline | Current PID | Delta PID (pp) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for task in TASKS:
            sid_baseline = floors[task]
            sid_current = statistics.median(run["tasks"][task]["recall@32"] for run in runs)
            pid_baseline = baseline["tasks"][task]["pid_recall@32"]
            pid_current = statistics.median(run["tasks"][task]["pid_recall@32"] for run in runs)
            lines.append(
                f"| {task} | {sid_baseline:.4%} | {sid_current:.4%} | "
                f"{(sid_current - sid_baseline) * 100:+.4f} | {pid_baseline:.4%} | "
                f"{pid_current:.4%} | {(pid_current - pid_baseline) * 100:+.4f} |"
            )
    lines.extend(
        (
            "",
            (
                "| Five-task wall s | Shared SID group s | Product group s |"
                if repeats == 1
                else "| Run | Five-task wall s | Shared SID group s | Product group s |"
            ),
            "| ---: | ---: | ---: |" if repeats == 1 else "| ---: | ---: | ---: | ---: |",
        )
    )
    for run in runs:
        if repeats == 1:
            lines.append(
                f"| {run['wall_seconds']} | {run['sid_group_wall_seconds']} | "
                f"{run['product_group_wall_seconds']} |"
            )
        else:
            lines.append(
                f"| {run['number']} | {run['wall_seconds']} | "
                f"{run['sid_group_wall_seconds']} | {run['product_group_wall_seconds']} |"
            )
    if repeats > 1:
        lines.extend(
            (
                "",
                "| Task | SID min | SID median | SID max | SID range (pp) | "
                "SID sample SD (pp) | PID range (pp) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for task in TASKS:
            sid = variability[task]["sid_recall32_pp"]
            pid = variability[task]["pid_recall32_pp"]
            lines.append(
                f"| {task} | {sid['min']:.4f}% | {sid['median']:.4f}% | "
                f"{sid['max']:.4f}% | {sid['range']:.4f} | "
                f"{sid['sample_stddev']:.4f} | {pid['range']:.4f} |"
            )
        lines.extend(
            (
                "",
                "Candidate checksums are diagnostic and do not affect the Recall gate result.",
                "",
                "| Task | Checksums recorded | All repeats equal | Unique checksums |",
                "| --- | --- | --- | ---: |",
            )
        )
        for task in TASKS:
            stability = candidate_stability[task]
            lines.append(
                f"| {task} | {'yes' if stability['available'] else 'no'} | "
                f"{'yes' if stability['all_equal'] else 'no'} | "
                f"{stability['unique_count'] if stability['unique_count'] is not None else 'n/a'} |"
            )
    single_baseline = baseline is not None and repeats == 1
    if single_baseline:
        lines.extend(
            (
                "",
                "### Task timing",
                "",
                "| Task | Inference s | Task work s |",
                "| --- | ---: | ---: |",
            )
        )
    else:
        lines.extend(
            ("", "| Task | Run | SID Recall@32 | PID Recall@32 | Inference s | Task work s |")
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for task in TASKS:
        for run in runs:
            item = run["tasks"][task]
            if single_baseline:
                lines.append(
                    f"| {task} | {item['inference_seconds']:.1f} | {item['task_work_seconds']} |"
                )
            else:
                lines.append(
                    f"| {task} | {run['number']} | {item['recall@32']:.4%} | "
                    f"{item['pid_recall@32']:.4%} | {item['inference_seconds']:.1f} | "
                    f"{item['task_work_seconds']} |"
                )
    lines.append("")
    if repeats > 1:
        lines.append(
            f"Per-repeat wall time: min {total_times['min']:.0f} s, "
            f"median {total_times['median']:.1f} s, max {total_times['max']:.0f} s, "
            f"sample SD {total_times['sample_stddev']:.1f} s."
        )
    lines.append(
        f"Total wall time: {result['total_wall_seconds']} s."
        if repeats == 1
        else f"All {repeats} repeats wall time: {result['total_wall_seconds']} s."
    )
    if repeats > 1:
        lines.append(
            "This variation is on one fixed 1000-sample set; it does not estimate "
            "uncertainty relative to the full test sets."
        )
    if baseline_path is not None:
        status = (
            "WARNING (non-blocking)"
            if baseline_warnings
            else "FAILED"
            if failures
            else "pending confirmation"
            if confirmation_required
            else "passed"
        )
        lines.append(f"Baseline check: {status}.")
        if confirmation_required and not baseline_warnings:
            lines.append("Confirmation needed for: " + ", ".join(suspected_tasks) + ".")
    if baseline_warnings:
        lines.extend(("", "> [!WARNING]", *[f"> {warning}" for warning in baseline_warnings]))
    elif failures:
        lines.extend(("", *[f"- {failure}" for failure in failures]))
    return candidate, result, "\n".join(lines) + "\n", failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    baseline_name = os.environ.get("V1_RECALL_BASELINE_FILE")
    baseline_path = Path(baseline_name) if baseline_name else None
    failure_mode = os.environ.get("V1_RECALL_BASELINE_FAILURE_MODE", "blocking").strip().lower()
    if failure_mode not in BASELINE_FAILURE_MODES:
        raise SystemExit(
            "V1_RECALL_BASELINE_FAILURE_MODE must be one of: "
            + ", ".join(sorted(BASELINE_FAILURE_MODES))
        )
    candidate, result, summary, failures = summarize(
        args.result_dir, args.repeats, baseline_path, failure_mode
    )
    (args.result_dir / "b4_baseline_candidate.json").write_text(
        json.dumps(candidate, indent=2) + "\n", encoding="utf-8"
    )
    (args.result_dir / "b4_repeat_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.result_dir / "b4_repeat_summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(summary)
    for warning in result["baseline_warnings"]:
        escaped = warning.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::warning title=Beam V1 Recall baseline::{escaped}", flush=True)
    if failures and failure_mode == "blocking":
        raise SystemExit("Recall gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
