#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Analyze prediction-vs-GT error for acc_test JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze acc_test prediction/GT JSON files and print overall "
            "statistics, worst cases, and worst candidate entries."
        ),
    )
    parser.add_argument(
        "--pred-json",
        type=str,
        required=True,
        help="Prediction JSON path produced by examples/acc_test.py.",
    )
    parser.add_argument(
        "--gt-json",
        type=str,
        required=True,
        help="Golden-truth JSON path used for comparison.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help=(
            "Optional benchmark input JSON. When provided, enriches worst-case "
            "output with user_history_len and input candidate count."
        ),
    )
    parser.add_argument(
        "--top-k-cases",
        type=int,
        default=20,
        help="Number of worst cases to print (default: 20).",
    )
    parser.add_argument(
        "--top-k-entries",
        type=int,
        default=20,
        help="Number of worst candidate entries to print (default: 20).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Denominator floor for relative error (default: 1e-8).",
    )
    parser.add_argument(
        "--gt-abs-threshold",
        type=float,
        default=0.0,
        help=(
            "Only include entries with abs(gt_score) >= threshold in the "
            "thresholded relative-error statistics (default: 0.0)."
        ),
    )
    return parser.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be dict: {path}")
    return payload


def _build_case_map(payload: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{label} JSON must contain list field: cases")
    out: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} JSON has invalid case_id: {case!r}")
        if case_id in out:
            raise ValueError(f"{label} JSON has duplicate case_id: {case_id}")
        out[case_id] = case
    return out


def _build_input_case_meta_map(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    case_map = _build_case_map(payload, label="input")
    out: dict[str, dict[str, int]] = {}
    for case_id, case in case_map.items():
        user_history = case.get("user_history")
        candidates = case.get("candidates")
        if not isinstance(user_history, list) or not isinstance(candidates, list):
            raise ValueError(
                "input JSON cases must contain list fields: user_history, candidates"
            )
        out[case_id] = {
            "user_history_len": len(user_history),
            "input_num_candidates": len(candidates),
        }
    return out


def _candidate_id_keys(row: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in row.keys()
        if key not in {"score", "candidate_index", "task_scores"}
    )


def _validate_row_alignment(
    *,
    case_id: str,
    pred_row: dict[str, Any],
    gt_row: dict[str, Any],
) -> None:
    pred_idx = pred_row.get("candidate_index")
    gt_idx = gt_row.get("candidate_index")
    if pred_idx != gt_idx:
        raise ValueError(
            f"case={case_id} candidate_index mismatch: pred={pred_idx} gt={gt_idx}"
        )

    pred_id_keys = _candidate_id_keys(pred_row)
    gt_id_keys = _candidate_id_keys(gt_row)
    if pred_id_keys != gt_id_keys:
        raise ValueError(
            f"case={case_id} candidate id key mismatch: "
            f"pred={pred_id_keys} gt={gt_id_keys}"
        )

    for key in pred_id_keys:
        if pred_row.get(key) != gt_row.get(key):
            raise ValueError(
                f"case={case_id} candidate field mismatch for {key!r}: "
                f"pred={pred_row.get(key)!r} gt={gt_row.get(key)!r}"
            )


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * q) - 1))
    return sorted_values[idx]


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.9f}"


def main() -> int:
    args = _parse_args()
    pred_payload = _load_json(args.pred_json)
    gt_payload = _load_json(args.gt_json)
    input_case_meta = (
        _build_input_case_meta_map(_load_json(args.input_json))
        if args.input_json
        else {}
    )

    pred_cases = _build_case_map(pred_payload, label="prediction")
    gt_cases = _build_case_map(gt_payload, label="gt")

    if set(pred_cases.keys()) != set(gt_cases.keys()):
        missing_in_gt = sorted(set(pred_cases.keys()) - set(gt_cases.keys()))
        missing_in_pred = sorted(set(gt_cases.keys()) - set(pred_cases.keys()))
        raise ValueError(
            "Case mismatch between prediction and GT. "
            f"missing_in_gt={missing_in_gt[:10]} "
            f"missing_in_pred={missing_in_pred[:10]}"
        )

    rel_errors: list[float] = []
    abs_errors: list[float] = []
    rel_errors_thresholded: list[float] = []
    case_summaries: list[dict[str, Any]] = []
    entry_summaries: list[dict[str, Any]] = []

    for case_id in sorted(pred_cases.keys()):
        pred_case = pred_cases[case_id]
        gt_case = gt_cases[case_id]
        pred_scores = pred_case.get("scores")
        gt_scores = gt_case.get("scores")
        if not isinstance(pred_scores, list) or not isinstance(gt_scores, list):
            raise ValueError(f"case={case_id} missing scores list")
        if len(pred_scores) != len(gt_scores):
            raise ValueError(
                f"case={case_id} score length mismatch: "
                f"pred={len(pred_scores)} gt={len(gt_scores)}"
            )

        case_rel_errors: list[float] = []
        case_abs_errors: list[float] = []

        for pred_row, gt_row in zip(pred_scores, gt_scores):
            if not isinstance(pred_row, dict) or not isinstance(gt_row, dict):
                raise ValueError(f"case={case_id} score rows must be dicts")
            _validate_row_alignment(
                case_id=case_id,
                pred_row=pred_row,
                gt_row=gt_row,
            )

            pred_score = float(pred_row.get("score"))
            gt_score = float(gt_row.get("score"))
            abs_err = abs(pred_score - gt_score)
            rel_err = abs_err / max(abs(gt_score), args.eps)
            if not math.isfinite(abs_err) or not math.isfinite(rel_err):
                raise ValueError(
                    f"case={case_id} has non-finite error values: "
                    f"pred={pred_score} gt={gt_score}"
                )

            rel_errors.append(rel_err)
            abs_errors.append(abs_err)
            case_rel_errors.append(rel_err)
            case_abs_errors.append(abs_err)
            if abs(gt_score) >= args.gt_abs_threshold:
                rel_errors_thresholded.append(rel_err)

            summary = {
                "case_id": case_id,
                "candidate_index": pred_row.get("candidate_index"),
                "pred_score": pred_score,
                "gt_score": gt_score,
                "abs_error": abs_err,
                "rel_error": rel_err,
            }
            for key in _candidate_id_keys(pred_row):
                summary[key] = pred_row.get(key)
            entry_summaries.append(summary)

        case_summary = {
            "case_id": case_id,
            "num_candidates": len(pred_scores),
            "mean_abs_error": sum(case_abs_errors) / len(case_abs_errors),
            "max_abs_error": max(case_abs_errors),
            "mean_rel_error": sum(case_rel_errors) / len(case_rel_errors),
            "max_rel_error": max(case_rel_errors),
        }
        case_summary.update(input_case_meta.get(case_id, {}))
        case_summaries.append(case_summary)

    if not rel_errors:
        raise ValueError("No comparable score entries found")

    rel_errors_sorted = sorted(rel_errors)
    abs_errors_sorted = sorted(abs_errors)
    thresholded_sorted = sorted(rel_errors_thresholded)
    case_summaries.sort(key=lambda x: (x["mean_rel_error"], x["max_rel_error"]), reverse=True)
    entry_summaries.sort(key=lambda x: (x["rel_error"], x["abs_error"]), reverse=True)

    top_entry_case_hits = Counter(
        row["case_id"] for row in entry_summaries[: args.top_k_entries]
    )
    for row in case_summaries:
        row["worst_entry_hits"] = top_entry_case_hits.get(row["case_id"], 0)

    print("== Overall ==")
    print(f"count = {len(rel_errors)}")
    print(f"mean_relative_error = {_format_float(sum(rel_errors) / len(rel_errors))}")
    print(f"p50_relative_error = {_format_float(_percentile(rel_errors_sorted, 0.50))}")
    print(f"p95_relative_error = {_format_float(_percentile(rel_errors_sorted, 0.95))}")
    print(f"max_relative_error = {_format_float(rel_errors_sorted[-1])}")
    print(f"mean_absolute_error = {_format_float(sum(abs_errors) / len(abs_errors))}")
    print(f"p95_absolute_error = {_format_float(_percentile(abs_errors_sorted, 0.95))}")
    print(f"max_absolute_error = {_format_float(abs_errors_sorted[-1])}")
    if args.gt_abs_threshold > 0.0:
        print(
            "mean_relative_error(|gt|>=threshold) = "
            f"{_format_float(sum(thresholded_sorted) / len(thresholded_sorted)) if thresholded_sorted else 'nan'}"
        )
        print(f"threshold_count = {len(thresholded_sorted)}")

    print("\n== Worst Cases ==")
    for row in case_summaries[: args.top_k_cases]:
        meta_suffix = ""
        if "user_history_len" in row:
            meta_suffix += f" user_history_len={row['user_history_len']}"
        if "input_num_candidates" in row:
            meta_suffix += f" input_num_candidates={row['input_num_candidates']}"
        print(
            "case_id={case_id} num_candidates={num_candidates} "
            "mean_rel={mean_rel_error:.9f} max_rel={max_rel_error:.9f} "
            "mean_abs={mean_abs_error:.9f} max_abs={max_abs_error:.9f} "
            "worst_entry_hits={worst_entry_hits}".format(**row)
            + meta_suffix
        )

    print("\n== Worst Candidate Entries ==")
    for row in entry_summaries[: args.top_k_entries]:
        id_fields = [
            f"{key}={row[key]}" for key in sorted(row.keys())
            if key not in {
                "case_id",
                "candidate_index",
                "pred_score",
                "gt_score",
                "abs_error",
                "rel_error",
            }
        ]
        suffix = (" " + " ".join(id_fields)) if id_fields else ""
        print(
            f"case_id={row['case_id']} candidate_index={row['candidate_index']} "
            f"pred={row['pred_score']:.9f} gt={row['gt_score']:.9f} "
            f"abs={row['abs_error']:.9f} rel={row['rel_error']:.9f}{suffix}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
