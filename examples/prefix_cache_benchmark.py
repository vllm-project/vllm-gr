#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end prefix-cache benchmark for HSTU on vLLM-GR.

This script replays the same GR requests multiple times against one live
``GRLLM`` instance and reports:

- end-to-end latency per pass
- vLLM prefix-cache queried tokens
- vLLM prefix-cache hit tokens
- prefix-cache token hit rate

The default workflow is:
1. reset prefix cache
2. run one cold pass to populate the cache
3. run one hot replay pass to measure reuse
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from vllm.pooling_params import PoolingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATASET_NAME = "movielens-1m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end HSTU prefix-cache hit rate.",
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
        "--max-uih-len",
        "--max-seq-len",
        dest="max_uih_len",
        type=int,
        default=128,
        help=(
            "GRLLM max_seq_len for UIH/history truncation "
            "(default: 128). `--max-seq-len` is kept as a backward-compatible alias."
        ),
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
        action="append",
        default=None,
        help="Optional case_id filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=20,
        help="Maximum number of cases to replay (default: 20).",
    )
    parser.add_argument(
        "--hot-passes",
        type=int,
        default=1,
        help="Number of hot replay passes after the cold pass (default: 1).",
    )
    parser.add_argument(
        "--show-case-limit",
        type=int,
        default=10,
        help="Show at most this many per-case rows per pass (default: 10).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON report output path.",
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


def _select_cases(
    cases: list[dict[str, Any]],
    case_ids: list[str] | None,
    max_cases: int,
) -> list[dict[str, Any]]:
    selected = cases
    if case_ids:
        case_id_set = set(case_ids)
        selected = [case for case in cases if case["case_id"] in case_id_set]
        if not selected:
            raise ValueError(f"No cases matched case_ids={sorted(case_id_set)!r}")
    if max_cases > 0:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("No cases selected for benchmark")
    return selected


def _prompt_from_case(case: dict[str, Any]) -> dict[str, Any]:
    candidates = case["candidates"]
    return {
        "user_id": case["user_id"],
        "user_history": case["user_history"],
        "candidates": candidates,
        "context": case["context"],
        "num_recommendations": len(candidates),
    }


def _snapshot_prefix_cache_counters(llm) -> dict[str, int]:
    counters = llm.get_prefix_cache_counters()
    return {
        "requests": int(counters.get("requests", 0)),
        "queries": int(counters.get("queries", 0)),
        "hits": int(counters.get("hits", 0)),
    }


def _diff_counters(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0) - before.get(key, 0)) for key in ("requests", "queries", "hits")
    }


def _hit_rate(hits: int, queries: int) -> float:
    if queries <= 0:
        return 0.0
    return float(hits) / float(queries)


def _get_runtime_cache_block_size(llm) -> int:
    engine = llm.get_engine()
    cache_config = getattr(engine.vllm_config, "cache_config", None)
    if cache_config is None:
        return 0
    return int(getattr(cache_config, "block_size", 0) or 0)


def _get_runtime_device_type(llm) -> str:
    engine = llm.get_engine()
    device_config = getattr(engine.vllm_config, "device_config", None)
    return str(getattr(device_config, "device_type", "unknown"))


def _estimate_prompt_token_ids_len(llm, case: dict[str, Any]) -> int:
    engine = llm.get_engine()
    request = engine.input_processor.process_inputs(
        request_id=f"benchmark-len-{case['case_id']}",
        prompt=_prompt_from_case(case),
        params=PoolingParams(task="embed"),
    )
    prompt_token_ids = getattr(request, "prompt_token_ids", None) or []
    return len(prompt_token_ids)


def _estimate_cacheable_prefix_tokens(prompt_len: int, block_size: int) -> int:
    if prompt_len <= 1 or block_size <= 0:
        return 0
    # vLLM prefix cache only reuses full blocks, and the last token is still
    # recomputed so the effective upper bound is (prompt_len - 1).
    return ((prompt_len - 1) // block_size) * block_size


def _build_case_runtime_diagnostics(
    llm,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    block_size = _get_runtime_cache_block_size(llm)
    diagnostics: list[dict[str, Any]] = []
    for case in cases:
        prompt_len = _estimate_prompt_token_ids_len(llm, case)
        diagnostics.append(
            {
                "case_id": case["case_id"],
                "prompt_len": prompt_len,
                "max_cacheable_prefix_tokens": _estimate_cacheable_prefix_tokens(
                    prompt_len,
                    block_size,
                ),
            }
        )
    return diagnostics


def _print_runtime_cache_diagnostics(
    llm,
    case_diagnostics: list[dict[str, Any]],
    *,
    show_case_limit: int,
) -> None:
    block_size = _get_runtime_cache_block_size(llm)
    device_type = _get_runtime_device_type(llm)
    print("== Runtime Cache Config ==")
    print(f"device_type={device_type} block_size={block_size}")
    print("sample_prompt_lengths:")
    for row in case_diagnostics[: max(show_case_limit, 0)]:
        print(
            "  case_id={case_id} prompt_len={prompt_len} "
            "max_cacheable_prefix_tokens={max_cacheable_prefix_tokens}".format(**row)
        )
    if len(case_diagnostics) > max(show_case_limit, 0):
        print(f"  ... ({len(case_diagnostics) - max(show_case_limit, 0)} more cases)")

    if case_diagnostics and all(
        row["max_cacheable_prefix_tokens"] == 0 for row in case_diagnostics
    ):
        print(
            "warning: all selected cases are shorter than one reusable KV block "
            f"for the current runtime (block_size={block_size}). "
            "Prefix cache queries will still be counted, but prefix hits are "
            "expected to stay at 0."
        )


def _run_pass(
    *,
    llm,
    pass_name: str,
    cases: list[dict[str, Any]],
    show_case_limit: int,
) -> dict[str, Any]:
    print(f"\n== {pass_name} ==")
    pass_start = time.perf_counter()
    pass_before = _snapshot_prefix_cache_counters(llm)
    per_case_rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        prompt = _prompt_from_case(case)
        before = _snapshot_prefix_cache_counters(llm)
        start = time.perf_counter()
        _ = llm.recommend(prompts=[prompt], use_tqdm=False)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        after = _snapshot_prefix_cache_counters(llm)
        delta = _diff_counters(after, before)

        row = {
            "index": index,
            "case_id": case["case_id"],
            "history_len": len(case["user_history"]),
            "num_candidates": len(case["candidates"]),
            "latency_ms": latency_ms,
            "prefix_queries": delta["queries"],
            "prefix_hits": delta["hits"],
            "prefix_hit_rate": _hit_rate(delta["hits"], delta["queries"]),
        }
        per_case_rows.append(row)

    pass_after = _snapshot_prefix_cache_counters(llm)
    pass_delta = _diff_counters(pass_after, pass_before)
    elapsed_s = time.perf_counter() - pass_start
    avg_latency_ms = (
        sum(row["latency_ms"] for row in per_case_rows) / len(per_case_rows)
        if per_case_rows
        else 0.0
    )

    summary = {
        "pass_name": pass_name,
        "num_cases": len(per_case_rows),
        "elapsed_s": elapsed_s,
        "avg_latency_ms": avg_latency_ms,
        "prefix_requests": pass_delta["requests"],
        "prefix_queries": pass_delta["queries"],
        "prefix_hits": pass_delta["hits"],
        "prefix_hit_rate": _hit_rate(pass_delta["hits"], pass_delta["queries"]),
        "cases": per_case_rows,
    }

    print(
        "cases={num_cases} elapsed_s={elapsed_s:.4f} avg_latency_ms={avg_latency_ms:.3f} "
        "prefix_queries={prefix_queries} prefix_hits={prefix_hits} "
        "prefix_hit_rate={prefix_hit_rate:.4%}".format(**summary)
    )
    print("sample_cases:")
    for row in per_case_rows[: max(show_case_limit, 0)]:
        print(
            "  [{index:02d}] case_id={case_id} history_len={history_len} "
            "candidates={num_candidates} latency_ms={latency_ms:.3f} "
            "queries={prefix_queries} hits={prefix_hits} hit_rate={prefix_hit_rate:.4%}".format(
                **row
            )
        )
    if len(per_case_rows) > max(show_case_limit, 0):
        print(f"  ... ({len(per_case_rows) - max(show_case_limit, 0)} more cases)")

    return summary


def _assert_nonzero_hit_rate(
    *,
    summary: dict[str, Any],
    case_diagnostics: list[dict[str, Any]],
) -> None:
    if summary["prefix_queries"] <= 0:
        raise RuntimeError(
            f"{summary['pass_name']} produced no prefix-cache queries; "
            "benchmark cannot validate prefix reuse."
        )

    has_reusable_prefix = any(
        int(row.get("max_cacheable_prefix_tokens", 0)) > 0 for row in case_diagnostics
    )
    if not has_reusable_prefix:
        raise RuntimeError(
            f"{summary['pass_name']} has no reusable prefix blocks in the selected "
            "cases; benchmark input is not suitable for prefix-cache validation."
        )

    if summary["prefix_hit_rate"] <= 0.0:
        raise RuntimeError(
            f"{summary['pass_name']} prefix_hit_rate is 0; treat this benchmark run as a failure."
        )


def main() -> int:
    args = _parse_args()
    input_payload = _load_json(args.input_json)
    cases = _select_cases(
        _validate_input_payload(input_payload),
        args.case_id,
        args.max_cases,
    )

    from vllm_gr import GRLLM

    llm = GRLLM(
        model=args.model,
        max_seq_len=args.max_uih_len,
        max_candidates=args.max_candidates,
        enable_prefix_caching=True,
    )

    report: dict[str, Any] = {
        "model": args.model,
        "input_json": args.input_json,
        "max_uih_len": args.max_uih_len,
        "max_candidates": args.max_candidates,
        "num_cases": len(cases),
        "hot_passes": args.hot_passes,
        "passes": [],
    }

    try:
        case_diagnostics = _build_case_runtime_diagnostics(llm, cases)
        _print_runtime_cache_diagnostics(
            llm,
            case_diagnostics,
            show_case_limit=args.show_case_limit,
        )
        report["runtime_cache"] = {
            "device_type": _get_runtime_device_type(llm),
            "block_size": _get_runtime_cache_block_size(llm),
            "cases": case_diagnostics,
        }

        reset_ok = llm.reset_prefix_cache(reset_running_requests=True)
        print("== Prefix Cache Reset ==")
        print(f"reset_ok={reset_ok}")

        cold_summary = _run_pass(
            llm=llm,
            pass_name="cold_pass",
            cases=cases,
            show_case_limit=args.show_case_limit,
        )
        report["passes"].append(cold_summary)

        for hot_index in range(args.hot_passes):
            hot_summary = _run_pass(
                llm=llm,
                pass_name=f"hot_pass_{hot_index + 1}",
                cases=cases,
                show_case_limit=args.show_case_limit,
            )
            report["passes"].append(hot_summary)
            _assert_nonzero_hit_rate(
                summary=hot_summary,
                case_diagnostics=case_diagnostics,
            )

        if args.hot_passes <= 0:
            _assert_nonzero_hit_rate(
                summary=cold_summary,
                case_diagnostics=case_diagnostics,
            )

        print("\n== Overall ==")
        for summary in report["passes"]:
            print(
                "{pass_name}: avg_latency_ms={avg_latency_ms:.3f} "
                "prefix_queries={prefix_queries} prefix_hits={prefix_hits} "
                "prefix_hit_rate={prefix_hit_rate:.4%}".format(**summary)
            )

        if len(report["passes"]) >= 2:
            cold = report["passes"][0]
            hot = report["passes"][1]
            latency_gain = cold["avg_latency_ms"] - hot["avg_latency_ms"]
            print(
                "first_hot_vs_cold: latency_delta_ms={:.3f} hit_rate_delta={:.4%}".format(
                    latency_gain,
                    hot["prefix_hit_rate"] - cold["prefix_hit_rate"],
                )
            )
    finally:
        llm.shutdown()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote report: {output_path}")

    return 0


if __name__ == "__main__":
    main()
