#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compare baseline vs head offline AB summaries and emit a markdown delta report.

Reads one or more ``vllm-gr-summary.json`` files per side (one per measurement)
and reports, for a small set of primary latency metrics, the percent
change from baseline to head together with a noise-threshold verdict.

``delta% = (head - baseline) / baseline * 100`` — positive means slower (a
regression), negative means faster (an improvement). Each metric is a single
row; p50/p90/p99 are shown side by side as ``base→head (delta%)``.

Each reported latency statistic is merged by taking its median across the
measurements for that side. All measurements on a side must use the same Git
revision.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

if __package__:
    from . import generate_summary
else:
    import generate_summary

# Keep non-ASCII report glyphs (→, ⚠️, …) printable on every platform;
# Windows defaults stdout to a legacy code page that cannot encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Noise thresholds (same values as validate_lightweight_timing.py).
NOISE_LIMITS: dict[str, float] = {"mean": 2.0, "p50": 2.0, "p90": 3.0, "p99": 5.0}

# Primary metrics shown in the report. The finer-grained engine/CPU breakdown
# remains available in the per-scenario JSON summaries but is intentionally
# omitted from the main table to keep the signal readable.
KEY_METRICS = [
    "e2el",  # end-to-end, cold-cache (miss) request
    "e2el_hit",  # end-to-end, warm-cache (hit) request
    "prefill",  # prefill phase, combined miss+hit
    "prefill_hit",  # prefill phase, warm-cache hit
    "decode",  # decode phase, combined miss+hit
    "decode_hit",  # decode phase, warm-cache hit
    "total_beam",  # full beam-search aggregate
]

PERCENTILES = ("mean", "p50", "p90", "p99")
CANONICAL_METRICS = ("e2el", "e2el_hit")
CONTENT_FIELDS = (
    "dataset.measured_sha256",
    "dataset.selection.workload_sha256",
    "dataset.selection.warmup_workload_sha256",
    "scenario.server_args.catalog_sha256",
    "scenario.server_args.constraint_artifact_sha256",
    "scenario.server_args.tokenizer_sha256",
)
MEASUREMENT_FIELDS = (
    "scenario.n",
    "scenario.input_tokens_target",
    "scenario.num_prompts",
    "scenario.warmup_requests",
    "scenario.max_concurrency",
    "scenario.batch_size",
    "scenario.benchmark_args.max_tokens",
    "scenario.benchmark_args.measurement_mode",
    "scenario.benchmark_args.cache_protocol",
    "scenario.benchmark_args.phase_definition.version",
)
RUNTIME_FIELDS = ("environment.runtime.vllm_version", "environment.runtime.torch_version")
GPU_FIELDS = (
    "environment.gpu.name",
    "environment.gpu.memory_mib",
    "environment.gpu.driver_version",
)


def load(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("summary must be a JSON object")
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return result


def common_revision(side: str, paths: list[Path], summaries: list[dict]) -> str:
    expected_sha = ""
    for path, summary in zip(paths, summaries, strict=True):
        source = summary.get("source")
        git_sha = source.get("git_sha") if isinstance(source, dict) else None
        if not isinstance(git_sha, str) or not git_sha.strip():
            raise ValueError(f"{side} summary {path} must provide a nonempty source.git_sha")
        if expected_sha and git_sha != expected_sha:
            raise ValueError(
                f"{side} summaries use different Git revisions. "
                f"{path} has source.git_sha {git_sha!r} and {paths[0]} has {expected_sha!r}."
            )
        expected_sha = git_sha
    return expected_sha


def metric_sources(s: dict) -> tuple[dict, dict]:
    """Return canonical and post-canonical diagnostic latency maps."""
    results = s.get("results", {})
    canonical = results.get("latency_ms", {})
    diagnostic = (results.get("diagnostic") or {}).get("latency_ms", {})
    return canonical, diagnostic


def metric_source(s: dict, metric: str) -> dict:
    """Use canonical samples for E2E and diagnostics for internal stages."""
    canonical, diagnostic = metric_sources(s)
    if metric in CANONICAL_METRICS:
        return canonical
    return diagnostic or canonical


def merge_latency(summaries: list[dict]) -> dict[str, dict[str, float]]:
    """Median of each percentile of each metric across several summaries."""
    merged: dict[str, dict[str, float]] = {}
    for metric in KEY_METRICS:
        row: dict[str, float] = {}
        for pctl in PERCENTILES:
            values = []
            for summary in summaries:
                value = metric_source(summary, metric).get(metric, {}).get(pctl)
                if value is not None:
                    latency = float(value)
                    if math.isfinite(latency) and latency > 0:
                        values.append(latency)
            if values:
                row[pctl] = statistics.median(values)
        if row:
            merged[metric] = row
    return merged


def field(summary: dict, name: str):
    value = summary
    for part in name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def comparison_field(summary: dict, name: str):
    if name != "dataset.measured_sha256":
        return field(summary, name)

    dataset = field(summary, "dataset")
    if isinstance(dataset, dict) and "measured_sha256" in dataset:
        return dataset["measured_sha256"]

    # Earlier summaries could fill the SHA from the manifest on incomplete runs.
    reasons = field(summary, "run.qualification_reasons")
    if isinstance(reasons, list) and generate_summary.INCOMPLETE_MEASUREMENT_IDENTITY in reasons:
        return None
    return field(summary, "dataset.sha256")


def new_offline(summary: dict) -> bool:
    schema = field(summary, "source.raw_schema_version")
    samples = field(summary, "results.samples")
    return schema == generate_summary.OFFLINE_RAW_V3 or (
        field(summary, "scenario.execution_mode") == "offline"
        and (
            field(summary, "dataset.selection.workload_sha256") is not None
            or (
                isinstance(samples, dict)
                and any(
                    samples.get(f"{phase}_cached_tokens") is not None for phase in ("cold", "warm")
                )
            )
        )
    )


def valid_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_samples(summary: dict, count: int) -> None:
    for metric in CANONICAL_METRICS:
        sample_field = f"results.samples.{metric}_ms"
        samples = field(summary, sample_field)
        if (
            not isinstance(samples, list)
            or len(samples) != count
            or any(not valid_number(value) for value in samples)
        ):
            raise ValueError(f"{sample_field} must contain {count} finite positive values")
        metric_field = f"results.latency_ms.{metric}"
        if field(summary, f"{metric_field}.unit") != "ms":
            raise ValueError(f"{metric_field}.unit must be ms")
        quantiles = (
            statistics.quantiles(samples, n=100, method="inclusive") if count > 1 else samples * 99
        )
        expected = (statistics.fmean(samples), quantiles[49], quantiles[89], quantiles[98])
        for percentile, value in zip(PERCENTILES, expected, strict=True):
            reported = field(summary, f"{metric_field}.{percentile}")
            if not valid_number(reported) or not math.isclose(reported, value):
                raise ValueError(f"{metric_field}.{percentile} disagrees with recorded samples")


def is_digest(value: object, size: int) -> bool:
    return isinstance(value, str) and len(value) == size and set(value) <= set("0123456789abcdef")


def check_summary(summary: dict, completed: int) -> list[str]:
    if not new_offline(summary):
        return ["historical summary lacks verified offline measurement evidence"]

    if field(summary, "scenario.execution_mode") != "offline":
        raise ValueError("scenario.execution_mode must be offline")
    failed = field(summary, "results.requests.failed")
    if completed < 1 or type(failed) is not int or failed != 0:
        raise ValueError("results.requests requires positive completed and zero failed requests")
    if field(summary, "run.status") != "success":
        raise ValueError("run.status must be success")
    num_prompts = field(summary, "scenario.num_prompts")
    if type(num_prompts) is not int or num_prompts != completed:
        raise ValueError("scenario.num_prompts disagrees with results.requests.completed")
    actual_sha = field(summary, "source.runtime_source.git_sha")
    if actual_sha and actual_sha != field(summary, "source.git_sha"):
        raise ValueError("source.runtime_source.git_sha disagrees with source.git_sha")
    batch_size = field(summary, "scenario.batch_size")
    sample_unit = field(summary, "results.samples.unit")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("scenario.batch_size must be a positive integer")
    if sample_unit not in ("request", "logical-batch"):
        raise ValueError("results.samples.unit must be request or logical-batch")
    sample_count = (
        (completed + batch_size - 1) // batch_size if sample_unit == "logical-batch" else completed
    )
    validate_samples(summary, sample_count)

    reasons = []
    for name in (*CONTENT_FIELDS, "dataset.selection.sample_ids_sha256"):
        digest = comparison_field(summary, name)
        if digest is None:
            reasons.append(f"{name} is unavailable")
        elif not is_digest(digest, 64):
            raise ValueError(f"{name} must be a SHA-256 digest")
    model_revision = field(summary, "model.revision")
    if model_revision is None:
        reasons.append("model.revision is unavailable")
    elif not (
        is_digest(model_revision, 40)
        or (
            isinstance(model_revision, str)
            and model_revision.startswith("sha256:")
            and is_digest(model_revision[7:], 64)
        )
    ):
        raise ValueError("model.revision must identify the model snapshot or local files")
    if not all(field(summary, name) for name in RUNTIME_FIELDS):
        reasons.append("measured vLLM or Torch version is unavailable")
    if any(field(summary, name) is None for name in MEASUREMENT_FIELDS):
        reasons.append("offline measurement parameters are incomplete")
    if batch_size != 1 or field(summary, "scenario.max_concurrency") != 1:
        reasons.append("performance comparisons require single-request measurements")
    if field(summary, "scenario.server_args.enable_prefix_caching") is not True:
        reasons.append("prefix caching was disabled or unrecorded")
    cache_issue = generate_summary.cache_qualification_reason(
        field(summary, "results.samples"),
        sample_count,
        check_request_lengths=(
            field(summary, "source.raw_schema_version") == generate_summary.OFFLINE_RAW_V3
            and batch_size == 1
        ),
    )
    if cache_issue:
        reasons.append(cache_issue)
    contaminated = field(
        summary, "scenario.benchmark_args.instrumentation.canonical_contaminated_by_diagnostic"
    )
    if contaminated is True:
        reasons.append("worker lightweight timing was enabled during canonical sampling")
    elif contaminated is not False:
        reasons.append("canonical measurement instrumentation is unknown")
    if completed < 100:
        reasons.append("smoke run uses fewer than 100 measured requests")
    return reasons


def check_matching_runs(runs: list[tuple[Path, dict]]) -> None:
    seen = {}
    for path, summary in runs:
        fields = (*CONTENT_FIELDS, "model.revision", "dataset.task")
        if new_offline(summary):
            fields += (
                *MEASUREMENT_FIELDS,
                *RUNTIME_FIELDS,
                *GPU_FIELDS,
                "environment.gpu.uuid",
                "dataset.selection.sample_ids_sha256",
            )
            server_args = field(summary, "scenario.server_args") or {}
            if not isinstance(server_args, dict):
                raise ValueError(f"{path}: scenario.server_args must be an object")
            fields += tuple(
                f"scenario.server_args.{key}"
                for key in server_args
                if key != "catalog_path" and f"scenario.server_args.{key}" not in CONTENT_FIELDS
            )
        for name in fields:
            value = comparison_field(summary, name)
            if value is None or value == "":
                continue
            previous_path, previous_value = seen.setdefault(name, (path, value))
            if value != previous_value:
                raise ValueError(f"{path}: {name} differs from {previous_path}")


def pct(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        action="append",
        required=True,
        help="baseline summary (repeatable, one per measurement)",
    )
    parser.add_argument(
        "--head",
        type=Path,
        action="append",
        required=True,
        help="head summary (repeatable, one per measurement)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scenario", default="")
    args = parser.parse_args()

    def validate_runs(paths: list[Path], summaries: list[dict]) -> list[int]:
        counts = []
        for path, summary in zip(paths, summaries, strict=True):
            try:
                completed = field(summary, "results.requests.completed")
                if not new_offline(summary):
                    completed = int(completed)
                if type(completed) is not int or completed < 0:
                    raise ValueError("results.requests.completed must be a nonnegative integer")
                reasons.update(check_summary(summary, completed))
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"{path}: {exc}") from exc
            counts.append(completed)
        return counts

    reasons: set[str] = set()
    try:
        base_summaries = [load(path) for path in args.base]
        head_summaries = [load(path) for path in args.head]
        base_sha = common_revision("Baseline", args.base, base_summaries)
        head_sha = common_revision("Head", args.head, head_summaries)
        base_completed = validate_runs(args.base, base_summaries)
        head_completed = validate_runs(args.head, head_summaries)
        summaries = base_summaries + head_summaries
        check_matching_runs(list(zip(args.base + args.head, summaries, strict=True)))
        historical = any(not new_offline(summary) for summary in summaries)
        gpu_identity = ("environment.gpu.uuid",)
        if not historical and not any(
            all(field(summary, name) for summary in summaries for name in fields)
            for fields in (gpu_identity, GPU_FIELDS)
        ):
            reasons.add("measurement devices lack a common GPU identity")
        if not reasons:
            warm_counts = field(summaries[0], "results.samples.warm_cached_tokens")
            if any(
                field(summary, "results.samples.warm_cached_tokens") != warm_counts
                for summary in summaries[1:]
            ):
                reasons.add("warm prefix cache reuse differs across measurements")
        base = merge_latency(base_summaries)
        head = merge_latency(head_summaries)
    except (ValueError, TypeError, OverflowError) as exc:
        parser.error(str(exc))
    qualification = "historical" if historical else "smoke" if reasons else "performance"
    base_repeats = len(base_summaries)
    head_repeats = len(head_summaries)

    # metric -> percentile -> row
    data: dict[str, dict[str, dict[str, object]]] = {}
    for metric in KEY_METRICS:
        b = base.get(metric)
        h = head.get(metric)
        if not (b and h):
            continue
        for pctl in PERCENTILES:
            if pctl not in b or pctl not in h:
                continue
            bv, hv = b[pctl], h[pctl]
            delta = 100.0 * ((hv - bv) / bv)
            limit = NOISE_LIMITS[pctl]
            data.setdefault(metric, {})[pctl] = {
                "base_ms": bv,
                "head_ms": hv,
                "delta_pct": delta,
                "significant": abs(delta) > limit
                and not math.isclose(abs(delta), limit, rel_tol=1e-12),
            }

    def cell(row: dict[str, object] | None) -> str:
        if row is None:
            return "—"
        return f"{row['base_ms']:.2f}→{row['head_ms']:.2f} ({pct(float(row['delta_pct']))})"

    lines: list[str] = []
    lines.append(f"# AB Benchmark Report - {args.label}")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Baseline | `{base_sha}` |")
    lines.append(f"| Head | `{head_sha}` |")
    lines.append(f"| Scenario | `{args.scenario}` |")
    lines.append(f"| Samples | {sum(base_completed)} baseline, {sum(head_completed)} head |")
    lines.append(f"| Repeats | {base_repeats} baseline, {head_repeats} head |")
    lines.append(f"| Qualification | {qualification} |")
    lines.append("")
    lines.append(
        "> Each cell = `baseline→head (delta%)`. delta% = (head-baseline)/baseline; "
        "**positive = slower**, negative = faster. "
        "⚠️ = beyond the noise threshold "
        f"(mean/p50 +/-{NOISE_LIMITS['mean']:.0f}% / p90 +/-{NOISE_LIMITS['p90']:.0f}% / p99 +/-{NOISE_LIMITS['p99']:.0f}%)."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Metric | mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) | Beyond threshold |")
    lines.append("|---|---|---|---|---|---|")
    for metric in KEY_METRICS:
        if metric not in data:
            continue
        rows = data[metric]
        sig = any(r["significant"] for r in rows.values())
        verdict = "⚠️" if sig else "—"
        lines.append(
            f"| {metric} | {cell(rows.get('mean'))} | {cell(rows.get('p50'))} | {cell(rows.get('p90'))} | "
            f"{cell(rows.get('p99'))} | {verdict} |"
        )
    lines.append("")

    regressions: dict[str, dict[str, dict[str, object]]] = {}
    improvements: dict[str, dict[str, dict[str, object]]] = {}
    for metric, by_pctl in data.items():
        if reasons or metric not in CANONICAL_METRICS:
            continue
        for pctl, r in by_pctl.items():
            if not r["significant"]:
                continue
            target = regressions if r["delta_pct"] > 0 else improvements
            target.setdefault(metric, {})[pctl] = r

    lines.append("## Conclusion")
    lines.append("")
    if reasons:
        lines.append("**Measurements are descriptive.** " + ". ".join(sorted(reasons)) + ".")
    elif not regressions:
        lines.append("**✅ No regressions**")
        lines.append("")
    if not reasons and not regressions and not improvements:
        lines.append(
            "All changes are within the noise threshold; no significant performance change observed."
        )
    else:

        def concl_table(title: str, src: dict[str, dict[str, dict[str, object]]]) -> None:
            if not src:
                return
            lines.append(title)
            lines.append("")
            lines.append("| Metric | mean delta% | p50 delta% | p90 delta% | p99 delta% |")
            lines.append("|---|---|---|---|---|")
            for metric in KEY_METRICS:
                if metric not in src:
                    continue
                rows = src[metric]

                def d(p: str, rows=rows) -> str:
                    r = rows.get(p)
                    return pct(float(r["delta_pct"])) if r else "—"

                lines.append(f"| {metric} | {d('mean')} | {d('p50')} | {d('p90')} | {d('p99')} |")
            lines.append("")

        concl_table(f"**Regressions ({len(regressions)})**", regressions)
        concl_table(f"**Improvements ({len(improvements)})**", improvements)
    lines.append("")

    text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")

    if args.json_output:
        rows_out = [
            {"metric": metric, "percentile": pctl, **r}
            for metric, by_pctl in data.items()
            for pctl, r in by_pctl.items()
        ]
        payload = {
            "label": args.label,
            "qualification": qualification,
            "baseline_eligible": not reasons,
            "qualification_reasons": sorted(reasons),
            "scenario": args.scenario,
            "base_sha": base_sha,
            "head_sha": head_sha,
            # The JSON fields carry the first summary's counts for compatibility.
            "base_completed": base_completed[0],
            "head_completed": head_completed[0],
            "base_total_completed": sum(base_completed),
            "head_total_completed": sum(head_completed),
            "base_repeats": base_repeats,
            "head_repeats": head_repeats,
            "noise_limits_pct": NOISE_LIMITS,
            "rows": rows_out,
            "regressions": {
                k: {p: v["delta_pct"] for p, v in rs.items()} for k, rs in regressions.items()
            },
            "improvements": {
                k: {p: v["delta_pct"] for p, v in rs.items()} for k, rs in improvements.items()
            },
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
