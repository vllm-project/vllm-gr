#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build a prefix-cache-friendly replay input JSON from an existing dataset.

The source benchmark data often contains one request per user, which makes it
hard to observe cross-request prefix reuse inside a single pass. This tool
expands each seed case into multiple replay variants that intentionally share:

- the same user_id
- the same context
- the same user_history

while varying candidates in controlled ways so that HSTU can exercise both:

- full request replay hits
- partial prefix hits on ``contextual prefix + history`` with changed targets
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


DATASET_NAME = "movielens-1m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a replay-style input JSON for prefix-cache benchmarks.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to the source benchmark input JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to the generated replay input JSON.",
    )
    parser.add_argument(
        "--max-seed-cases",
        type=int,
        default=8,
        help="Use at most this many source cases as replay seeds (default: 8).",
    )
    parser.add_argument(
        "--variants-per-case",
        type=int,
        default=4,
        help="How many replay variants to emit for each seed case (default: 4).",
    )
    parser.add_argument(
        "--keep-candidate-prefix",
        type=int,
        default=2,
        help=(
            "Keep this many leading candidates unchanged in partial variants "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--case-id",
        type=str,
        action="append",
        default=None,
        help="Optional source case_id filter. Can be passed multiple times.",
    )
    return parser.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_input_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    meta = payload.get("meta", {})
    dataset = meta.get("dataset")
    if dataset is not None and dataset != DATASET_NAME:
        raise ValueError(f"Unsupported dataset: {dataset}, expected {DATASET_NAME}")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("input JSON must contain a list field: cases")

    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every case must contain non-empty string case_id")
        if not isinstance(case.get("user_history"), list):
            raise ValueError(f"case={case_id} missing user_history list")
        if not isinstance(case.get("candidates"), list):
            raise ValueError(f"case={case_id} missing candidates list")
        if not isinstance(case.get("context"), dict):
            raise ValueError(f"case={case_id} missing context dict")
    return cases


def _select_seed_cases(
    cases: list[dict[str, Any]],
    case_ids: list[str] | None,
    max_seed_cases: int,
) -> list[dict[str, Any]]:
    selected = cases
    if case_ids:
        case_id_set = set(case_ids)
        selected = [case for case in cases if case["case_id"] in case_id_set]
        if not selected:
            raise ValueError(f"No cases matched case_ids={sorted(case_id_set)!r}")
    if max_seed_cases > 0:
        selected = selected[:max_seed_cases]
    if not selected:
        raise ValueError("No source cases selected")
    return selected


def _rotate_tail(
    items: list[dict[str, Any]],
    *,
    keep_prefix: int,
    shift: int,
) -> list[dict[str, Any]]:
    if not items:
        return []
    keep_prefix = max(0, min(keep_prefix, len(items)))
    prefix = [copy.deepcopy(item) for item in items[:keep_prefix]]
    tail = [copy.deepcopy(item) for item in items[keep_prefix:]]
    if not tail:
        return prefix

    shift = shift % len(tail)
    if shift == 0:
        rotated_tail = tail
    else:
        rotated_tail = tail[shift:] + tail[:shift]
    return prefix + rotated_tail


def _build_variants_for_case(
    case: dict[str, Any],
    *,
    variants_per_case: int,
    keep_candidate_prefix: int,
) -> list[dict[str, Any]]:
    if variants_per_case <= 0:
        return []

    base_case = copy.deepcopy(case)
    base_case_id = str(case["case_id"])
    candidates = list(case["candidates"])
    variants: list[dict[str, Any]] = []

    for variant_idx in range(variants_per_case):
        variant = copy.deepcopy(base_case)
        variant["case_id"] = f"{base_case_id}__replay_{variant_idx + 1:02d}"

        if variant_idx == 0:
            variant_kind = "base"
            variant["candidates"] = copy.deepcopy(candidates)
        elif variant_idx == 1:
            variant_kind = "exact_replay"
            variant["candidates"] = copy.deepcopy(candidates)
        else:
            variant_kind = "candidate_tail_rotated"
            variant["candidates"] = _rotate_tail(
                candidates,
                keep_prefix=keep_candidate_prefix,
                shift=variant_idx - 1,
            )

        variant["prefix_cache_variant"] = {
            "source_case_id": base_case_id,
            "variant_index": variant_idx,
            "variant_kind": variant_kind,
        }
        variants.append(variant)

    return variants


def build_replay_payload(
    source_payload: dict[str, Any],
    *,
    case_ids: list[str] | None = None,
    max_seed_cases: int = 8,
    variants_per_case: int = 4,
    keep_candidate_prefix: int = 2,
) -> dict[str, Any]:
    source_cases = _validate_input_payload(source_payload)
    seed_cases = _select_seed_cases(source_cases, case_ids, max_seed_cases)

    replay_cases: list[dict[str, Any]] = []
    for case in seed_cases:
        replay_cases.extend(
            _build_variants_for_case(
                case,
                variants_per_case=variants_per_case,
                keep_candidate_prefix=keep_candidate_prefix,
            )
        )

    return {
        "meta": {
            "dataset": source_payload.get("meta", {}).get("dataset", DATASET_NAME),
            "source_num_cases": len(source_cases),
            "seed_num_cases": len(seed_cases),
            "generated_num_cases": len(replay_cases),
            "variants_per_case": variants_per_case,
            "keep_candidate_prefix": keep_candidate_prefix,
            "description": (
                "Replay-style benchmark input for prefix cache validation. "
                "Each source case is expanded into multiple requests that share "
                "the same user/context/history."
            ),
        },
        "cases": replay_cases,
    }


def main() -> int:
    args = _parse_args()
    source_payload = _load_json(args.input_json)
    replay_payload = build_replay_payload(
        source_payload,
        case_ids=args.case_id,
        max_seed_cases=args.max_seed_cases,
        variants_per_case=args.variants_per_case,
        keep_candidate_prefix=args.keep_candidate_prefix,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(replay_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("== Prefix Cache Benchmark Input Built ==")
    print(f"input_json={args.input_json}")
    print(f"output_json={output_path}")
    print(f"seed_cases={replay_payload['meta']['seed_num_cases']}")
    print(f"generated_cases={replay_payload['meta']['generated_num_cases']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[build_prefix_cache_benchmark_input] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
