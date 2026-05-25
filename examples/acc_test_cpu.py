#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run converted vllm-GR HSTU model on CPU and emit acc_test-compatible JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATASET_NAME = "movielens-1m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run converted vllm-GR benchmark inference on CPU and optionally "
            "compare with GT JSON. Output format matches examples/acc_test.py."
        ),
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
        help="Compatibility arg with acc_test.py. Unused by CPU-native path.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=256,
        help="Compatibility arg with acc_test.py. Unused by CPU-native path.",
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
                f"case={case_id} score length mismatch: "
                f"pred={len(pred_scores)} gt={len(gt_scores)}"
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
                raise ValueError(
                    f"case={case_id} encountered non-finite relative error: {rel}"
                )
            total += rel
            count += 1

    if count == 0:
        raise ValueError("No comparable score entries found for relative error")
    return total / count, count


def _extract_scores_from_predictions(
    predictions: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    expected_num_candidates: int,
) -> list[float]:
    if predictions is None:
        raise ValueError("predictions is None; cannot build aligned score list")

    if isinstance(predictions, (list, tuple)):
        if len(predictions) != 1:
            raise ValueError(
                "CPU single-case inference expected exactly one prediction shard, "
                f"got {len(predictions)}"
            )
        predictions = predictions[0]

    scores = predictions.detach().cpu().float()
    if scores.ndim == 1:
        out = scores.tolist()
    elif scores.ndim == 2:
        if scores.shape[1] == expected_num_candidates:
            out = scores[0].tolist()
        elif scores.shape[0] == expected_num_candidates:
            out = scores[:, 0].tolist()
        else:
            raise ValueError(
                "Unsupported predictions shape for alignment: "
                f"{tuple(scores.shape)}, expected candidates={expected_num_candidates}"
            )
    elif scores.numel() == expected_num_candidates:
        out = scores.reshape(-1).tolist()
    else:
        raise ValueError(
            "Unsupported predictions dimensions for alignment: "
            f"shape={tuple(scores.shape)} candidates={expected_num_candidates}"
        )

    if len(out) != expected_num_candidates:
        raise ValueError(
            f"Score count mismatch: got {len(out)} expected {expected_num_candidates}"
        )
    return [float(x) for x in out]


def _load_converted_context(model_dir: str) -> dict[str, Any]:
    from compare_sparse_case import (
        _build_feature_caps_from_converted_config,
        _build_feature_to_table_map_from_converted_config,
        _load_converted_weight_dict,
    )
    from vllm_gr.config.hstu_config import HSTUConfig
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

    config = HSTUConfig.from_pretrained(model_dir)
    model = HSTUForRecommendation(config=config, device=torch.device("cpu"))
    weights = _load_converted_weight_dict(model_dir)
    model.load_weights(iter(weights.items()))
    model.eval()

    return {
        "config": config,
        "feature_caps": _build_feature_caps_from_converted_config(config),
        "feature_to_table": _build_feature_to_table_map_from_converted_config(config),
        "model": model,
        "hstu_model": model.hstu,
    }


def _run_case_on_cpu(
    *,
    model: Any,
    feature_caps: dict[str, int],
    config: Any,
    case: dict[str, Any],
) -> dict[str, Any]:
    from compare_sparse_case import _build_sparse_kjts

    case_id = str(case["case_id"])
    uih_keys = list(config.hstu_uih_feature_names)
    candidate_keys = list(config.hstu_candidate_feature_names)
    contextual_keys = set(config.contextual_feature_to_max_length.keys())

    uih_features_kjt, candidates_features_kjt = _build_sparse_kjts(
        case=case,
        case_id=case_id,
        uih_keys=uih_keys,
        contextual_keys=contextual_keys,
        candidate_keys=candidate_keys,
        feature_caps=feature_caps,
    )

    with torch.inference_mode():
        predictions = model(
            input_ids=None,
            positions=None,
            uih_features=uih_features_kjt,
            candidates_features=candidates_features_kjt,
        )

    candidates = case["candidates"]
    scores = _extract_scores_from_predictions(predictions, len(candidates))
    return {
        "case_id": case_id,
        "scores": [
            {
                "candidate_index": cand_idx,
                "item_movie_id": int(cand["item_movie_id"]),
                "score": float(score),
            }
            for cand_idx, (cand, score) in enumerate(zip(candidates, scores))
        ],
    }


def main() -> int:
    args = _parse_args()
    del args.max_seq_len, args.max_candidates

    input_payload = _load_json(args.input_json)
    input_cases = _select_cases(
        _validate_input_payload(input_payload),
        args.case_id,
    )

    converted_ctx = _load_converted_context(args.model)
    model = converted_ctx["model"]
    feature_caps = converted_ctx["feature_caps"]
    config = converted_ctx["config"]

    result_cases: list[dict[str, Any]] = []
    total_cases = len(input_cases)
    for idx, case in enumerate(input_cases, start=1):
        result_cases.append(
            _run_case_on_cpu(
                model=model,
                feature_caps=feature_caps,
                config=config,
                case=case,
            )
        )
        if idx % 20 == 0 or idx == total_cases:
            print(f"progress: {idx}/{total_cases}")

    pred_payload = _build_output_payload(input_payload, result_cases)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pred_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote predictions: {output_path}")

    if args.gt_json:
        gt_payload = _load_json(args.gt_json)
        mre, count = _mean_relative_error(pred_payload, gt_payload, eps=1e-8)
        print(f"mean_relative_error={mre:.10f} count={count}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[acc_test_cpu] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
