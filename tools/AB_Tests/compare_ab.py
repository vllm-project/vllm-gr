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
import statistics
import sys
from pathlib import Path

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


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if metric in ("e2el", "e2el_hit"):
        return canonical
    return diagnostic or canonical


def merge_latency(summaries: list[dict]) -> dict[str, dict[str, float]]:
    """Median of each percentile of each metric across several summaries."""
    merged: dict[str, dict[str, float]] = {}
    for metric in KEY_METRICS:
        row: dict[str, float] = {}
        for pctl in PERCENTILES:
            values = [
                float(metric_source(s, metric)[metric][pctl])
                for s in summaries
                if metric in metric_source(s, metric) and pctl in metric_source(s, metric)[metric]
            ]
            if values:
                row[pctl] = statistics.median(values)
        if row:
            merged[metric] = row
    return merged


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

    base_summaries = [load(path) for path in args.base]
    head_summaries = [load(path) for path in args.head]
    try:
        base_sha = common_revision("Baseline", args.base, base_summaries)
        head_sha = common_revision("Head", args.head, head_summaries)
    except ValueError as exc:
        parser.error(str(exc))
    base = merge_latency(base_summaries)
    head = merge_latency(head_summaries)
    base_completed = [
        int(summary["results"]["requests"]["completed"]) for summary in base_summaries
    ]
    head_completed = [
        int(summary["results"]["requests"]["completed"]) for summary in head_summaries
    ]
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
            bv = b[pctl]
            hv = h[pctl]
            if bv <= 0:
                continue
            delta = 100.0 * (hv / bv - 1.0)
            data.setdefault(metric, {})[pctl] = {
                "base_ms": bv,
                "head_ms": hv,
                "delta_pct": delta,
                "significant": abs(delta) > NOISE_LIMITS[pctl],
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
    lines.append("")
    lines.append(
        "> Each cell = `baseline→head (delta%)`. delta% = (head-baseline)/baseline; "
        "**positive = slower (regression)**, negative = faster (improvement). "
        "⚠️ = beyond the noise threshold "
        f"(mean/p50 +/-{NOISE_LIMITS['mean']:.0f}% / p90 +/-{NOISE_LIMITS['p90']:.0f}% / p99 +/-{NOISE_LIMITS['p99']:.0f}%)."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Metric | mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) | Verdict |")
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
        for pctl, r in by_pctl.items():
            if not r["significant"]:
                continue
            target = regressions if r["delta_pct"] > 0 else improvements
            target.setdefault(metric, {})[pctl] = r

    lines.append("## Conclusion")
    lines.append("")
    if not regressions:
        lines.append("**✅ No regressions**")
        lines.append("")
    if not regressions and not improvements:
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
            "scenario": args.scenario,
            "base_sha": base_sha,
            "head_sha": head_sha,
            # The JSON fields carry the first summary's counts for compatibility.
            "base_completed": base_completed[0],
            "head_completed": head_completed[0],
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
