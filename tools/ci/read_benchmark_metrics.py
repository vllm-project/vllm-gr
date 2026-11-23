# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Read OneRec accuracy and vLLM serving benchmark JSON for the CI gate."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

ACCURACY_KEYS = ("pass@1", "pass@32", "recall@32", "elapsed_time")


def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate field {name}")
        result[name] = value
    return result


def numeric_value(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must contain a number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and non-negative, got {value!r}")
    return float(value)


def read_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        metrics = json.load(source, object_pairs_hook=unique_fields)
    if not isinstance(metrics, dict):
        raise ValueError("expected a metrics object")
    return metrics


def read_accuracy(metrics: dict[str, Any]) -> list[float]:
    values = []
    for key in ACCURACY_KEYS:
        value = numeric_value(metrics.get(key), key)
        if key != "elapsed_time" and value > 1:
            raise ValueError(f"{key} must be in [0, 1], got {value}")
        values.append(value)
    return values


def read_serving(metrics: dict[str, Any]) -> tuple[int, str, str]:
    failed = metrics.get("failed")
    if type(failed) is not int or failed < 0:
        raise ValueError(f"failed must be a non-negative integer, got {failed!r}")
    ttft = numeric_value(metrics.get("p99_ttft_ms"), "p99_ttft_ms")
    tpot = numeric_value(metrics.get("p99_tpot_ms"), "p99_tpot_ms")
    # Keep the threshold decisions made from the old two-decimal console output.
    return failed, format(ttft, ".2f"), format(tpot, ".2f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("accuracy").add_argument("path", type=Path)
    modes.add_parser("serving").add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        metrics = read_metrics(args.path)
        values = read_accuracy(metrics) if args.mode == "accuracy" else read_serving(metrics)
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(1, f"Invalid benchmark metrics in {args.path}: {exc}\n")
    print(*values)


if __name__ == "__main__":
    main()
