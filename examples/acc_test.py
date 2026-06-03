#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Run vllm-GR on benchmark input JSON and optionally compare with GT JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.logger import init_logger

DATASET_NAME = "movielens-1m"
logger = init_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vllm-GR benchmark inference and optional GT comparison.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to converted vllm-GR model directory.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Input benchmark JSON path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output prediction JSON path.",
    )
    parser.add_argument(
        "--gt-json",
        type=str,
        default=None,
        help="Optional GT JSON path for relative-error comparison.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="GRLLM max_seq_len (default: 128).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=256,
        help="GRLLM max_candidates (default: 256).",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Optional single case_id to run.",
    )
    return parser.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_cases(
    cases: list[dict[str, Any]],
    case_id: str | None,
) -> list[dict[str, Any]]:
    if case_id is None:
        return cases
    selected = [case for case in cases if case["case_id"] == case_id]
    if not selected:
        raise ValueError(f"case_id not found in input JSON: {case_id}")
    return selected


def _validate_input_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    meta = payload.get("meta", {})
    dataset = meta.get("dataset")
    if dataset is not None and dataset != DATASET_NAME:
        raise ValueError(f"Unsupported dataset: {dataset}, expected {DATASET_NAME}")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("input JSON must contain a list field: cases")

    seen_case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every case must contain non-empty string case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id found: {case_id}")
        seen_case_ids.add(case_id)

        if not isinstance(case.get("user_history"), list):
            raise ValueError(f"case={case_id} missing user_history list")
        if not isinstance(case.get("candidates"), list):
            raise ValueError(f"case={case_id} missing candidates list")
        if len(case["candidates"]) == 0:
            raise ValueError(f"case={case_id} has empty candidates")
        if not isinstance(case.get("context"), dict):
            raise ValueError(f"case={case_id} missing context dict")
    return cases


def _prompt_from_case(case: dict[str, Any]) -> dict[str, Any]:
    candidates = case["candidates"]
    return {
        "user_id": case["user_id"],
        "user_history": case["user_history"],
        "candidates": candidates,
        "context": case["context"],
        "num_recommendations": len(candidates),
    }


def _extract_scores_from_output(
    all_scores: torch.Tensor | None,
    expected_num_candidates: int,
) -> list[float]:
    if all_scores is None:
        raise ValueError("output.all_scores is None; cannot build aligned score list")

    scores = all_scores.detach().cpu().float()
    if scores.ndim == 1:
        out = scores.tolist()
    elif scores.ndim == 2:
        if scores.shape[1] == expected_num_candidates:
            out = scores[0].tolist()
        elif scores.shape[0] == expected_num_candidates:
            out = scores[:, 0].tolist()
        else:
            raise ValueError(
                "Unsupported all_scores shape for alignment: "
                f"{tuple(scores.shape)}, expected candidates={expected_num_candidates}"
            )
    elif scores.numel() == expected_num_candidates:
        out = scores.reshape(-1).tolist()
    else:
        raise ValueError(
            "Unsupported all_scores dimensions for alignment: "
            f"shape={tuple(scores.shape)} candidates={expected_num_candidates}"
        )

    if len(out) != expected_num_candidates:
        raise ValueError(f"Score count mismatch: got {len(out)} expected {expected_num_candidates}")
    return [float(x) for x in out]


def _build_output_payload(
    input_payload: dict[str, Any],
    result_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    in_meta = input_payload.get("meta", {})
    return {
        "meta": {
            "dataset": in_meta.get("dataset", DATASET_NAME),
            "num_cases": len(result_cases),
        },
        "cases": result_cases,
    }


def _build_case_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("comparison JSON must contain a list field: cases")
    out: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("comparison case has missing/invalid case_id")
        if case_id in out:
            raise ValueError(f"comparison JSON has duplicate case_id: {case_id}")
        out[case_id] = case
    return out


def _mean_relative_error(
    pred_payload: dict[str, Any],
    gt_payload: dict[str, Any],
    eps: float = 1e-8,
) -> tuple[float, int]:
    pred_cases = _build_case_map(pred_payload)
    gt_cases = _build_case_map(gt_payload)

    if set(pred_cases.keys()) != set(gt_cases.keys()):
        missing_in_gt = sorted(set(pred_cases.keys()) - set(gt_cases.keys()))
        missing_in_pred = sorted(set(gt_cases.keys()) - set(pred_cases.keys()))
        raise ValueError(
            "Case mismatch between prediction and GT. "
            f"missing_in_gt={missing_in_gt[:5]} "
            f"missing_in_pred={missing_in_pred[:5]}"
        )

    total = 0.0
    count = 0
    for case_id in sorted(pred_cases.keys()):
        pred_scores = pred_cases[case_id].get("scores")
        gt_scores = gt_cases[case_id].get("scores")
        if not isinstance(pred_scores, list) or not isinstance(gt_scores, list):
            raise ValueError(f"case={case_id} missing scores list")
        if len(pred_scores) != len(gt_scores):
            raise ValueError(
                f"case={case_id} score length mismatch: pred={len(pred_scores)} gt={len(gt_scores)}"
            )

        for pred_row, gt_row in zip(pred_scores, gt_scores):
            pred_idx = pred_row.get("candidate_index")
            gt_idx = gt_row.get("candidate_index")
            pred_item = pred_row.get("item_movie_id")
            gt_item = gt_row.get("item_movie_id")
            if pred_idx != gt_idx or pred_item != gt_item:
                raise ValueError(
                    f"case={case_id} alignment mismatch at candidate entry: "
                    f"pred(index={pred_idx}, item={pred_item}) vs "
                    f"gt(index={gt_idx}, item={gt_item})"
                )
            pred_score = float(pred_row.get("score"))
            gt_score = float(gt_row.get("score"))
            rel = abs(pred_score - gt_score) / max(abs(gt_score), eps)
            if not math.isfinite(rel):
                raise ValueError(f"case={case_id} encountered non-finite relative error: {rel}")
            total += rel
            count += 1

    if count == 0:
        raise ValueError("No comparable score entries found for relative error")
    return total / count, count


def main() -> int:
    args = _parse_args()
    input_payload = _load_json(args.input_json)
    input_cases = _select_cases(
        _validate_input_payload(input_payload),
        args.case_id,
    )
    from vllm_gr import GRLLM

    gr_config = GRConfig(
        max_seq_len=args.max_seq_len,
        max_candidates=args.max_candidates,
    )
    llm = GRLLM(
        model=args.model,
        gr_config=gr_config,
    )

    result_cases: list[dict[str, Any]] = []
    infer_start = time.perf_counter()
    try:
        total_cases = len(input_cases)
        for idx, case in enumerate(input_cases, start=1):
            prompt = _prompt_from_case(case)
            output = llm.recommend(prompts=[prompt], use_tqdm=False)[0]
            candidates = case["candidates"]
            scores = _extract_scores_from_output(output.all_scores, len(candidates))
            result_cases.append(
                {
                    "case_id": case["case_id"],
                    "scores": [
                        {
                            "candidate_index": cand_idx,
                            "item_movie_id": int(cand["item_movie_id"]),
                            "score": float(score),
                        }
                        for cand_idx, (cand, score) in enumerate(zip(candidates, scores))
                    ],
                }
            )
            if idx % 20 == 0 or idx == total_cases:
                logger.info("[acc_test] progress: %d/%d", idx, total_cases)
    except Exception as exc:
        logger.exception("[acc_test] inference failed: %s", exc)
    finally:
        infer_seconds = time.perf_counter() - infer_start
        logger.info(
            "[acc_test] inference_seconds=%.6f cases=%d avg_case_seconds=%.6f",
            infer_seconds,
            len(result_cases),
            infer_seconds / len(result_cases) if result_cases else 0.0,
        )
        llm.shutdown()

    pred_payload = _build_output_payload(input_payload, result_cases)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pred_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[acc_test] wrote predictions: %s", output_path)

    if args.gt_json:
        gt_payload = _load_json(args.gt_json)
        mre, count = _mean_relative_error(pred_payload, gt_payload, eps=1e-8)
        logger.info("[acc_test] mean_relative_error=%.10f count=%d", mre, count)

    return 0


if __name__ == "__main__":
    main()
