#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Normalize resolver output across benchmark policy runtime generations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ci.resolve_benchmark_policy import FAILURE_MODES, PolicyError  # noqa: E402

COMMON_REQUIRED_KEYS = (
    "BENCHMARK_POLICY_NAME",
    "BENCHMARK_DEVICE",
    "BENCHMARK_BOARD_ID",
    "ACCURACY_FAILURE_MODE",
    "PASS_AT_1_THRESHOLD",
    "PASS_AT_32_THRESHOLD",
    "RECALL_AT_32_THRESHOLD",
    "ELAPSED_TIME_ONLINE_THRESHOLD",
    "ELAPSED_TIME_OFFLINE_THRESHOLD",
    "P99_TTFT_THRESHOLD",
    "P99_TPOT_THRESHOLD",
)
PERFORMANCE_MODE_KEYS = (
    "P99_TTFT_FAILURE_MODE",
    "P99_TPOT_FAILURE_MODE",
    "ONLINE_ELAPSED_FAILURE_MODE",
    "OFFLINE_ELAPSED_FAILURE_MODE",
)
THRESHOLD_KEYS = (
    "PASS_AT_1_THRESHOLD",
    "PASS_AT_32_THRESHOLD",
    "RECALL_AT_32_THRESHOLD",
    "ELAPSED_TIME_ONLINE_THRESHOLD",
    "ELAPSED_TIME_OFFLINE_THRESHOLD",
    "P99_TTFT_THRESHOLD",
    "P99_TPOT_THRESHOLD",
)
PANGU_KEYS = (
    "PANGU1B_OFFLINE_ACCURACY_THRESHOLD",
    "PANGU1B_ONLINE_ACCURACY_THRESHOLD",
    "PANGU1B_OFFLINE_ACCURACY_FAILURE_MODE",
    "PANGU1B_ONLINE_ACCURACY_FAILURE_MODE",
)


def read_github_env(path: Path) -> dict[str, str]:
    """Read the simple KEY=VALUE format emitted by the trusted resolver."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PolicyError(f"could not read resolved policy environment {path}: {exc}") from exc
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise PolicyError(f"invalid resolved environment line {line_number}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise PolicyError(f"invalid or duplicate resolved environment key: {key!r}")
        values[key] = value
    return values


def _require(values: dict[str, str], key: str) -> str:
    value = values.get(key, "")
    if not value:
        raise PolicyError(f"resolved policy is missing {key}")
    return value


def _validate_mode(value: str, key: str) -> str:
    if value not in FAILURE_MODES:
        raise PolicyError(f"{key} must be blocking or warning, got {value!r}")
    return value


def _validate_number(value: str, key: str) -> str:
    try:
        number = float(value)
    except ValueError as exc:
        raise PolicyError(f"{key} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise PolicyError(f"{key} must be finite and non-negative")
    return value


def normalize_environment(values: dict[str, str]) -> dict[str, str]:
    """Add v2 fields to v1 output and fail closed for missing Pangu policy."""
    normalized = {key: _require(values, key) for key in COMMON_REQUIRED_KEYS}
    normalized["ACCURACY_FAILURE_MODE"] = _validate_mode(
        normalized["ACCURACY_FAILURE_MODE"], "ACCURACY_FAILURE_MODE"
    )
    for key in THRESHOLD_KEYS:
        normalized[key] = _validate_number(normalized[key], key)

    legacy_performance_mode = values.get("PERFORMANCE_FAILURE_MODE", "")
    for key in PERFORMANCE_MODE_KEYS:
        mode = values.get(key) or legacy_performance_mode
        if not mode:
            raise PolicyError(f"resolved policy is missing {key} and PERFORMANCE_FAILURE_MODE")
        normalized[key] = _validate_mode(mode, key)

    present_pangu_keys = [key for key in PANGU_KEYS if values.get(key)]
    if not present_pangu_keys:
        pangu_accuracy_threshold = "0.875" if normalized["BENCHMARK_DEVICE"] == "npu" else "1.0"
        normalized.update(
            {
                "PANGU1B_POLICY_SOURCE": "strict-bootstrap",
                "PANGU1B_OFFLINE_ACCURACY_THRESHOLD": pangu_accuracy_threshold,
                "PANGU1B_ONLINE_ACCURACY_THRESHOLD": pangu_accuracy_threshold,
                "PANGU1B_OFFLINE_ACCURACY_FAILURE_MODE": "blocking",
                "PANGU1B_ONLINE_ACCURACY_FAILURE_MODE": "blocking",
            }
        )
    elif len(present_pangu_keys) != len(PANGU_KEYS):
        missing = sorted(set(PANGU_KEYS) - set(present_pangu_keys))
        raise PolicyError("resolved Pangu policy is incomplete: " + ", ".join(missing))
    else:
        normalized["PANGU1B_POLICY_SOURCE"] = values.get("PANGU1B_POLICY_SOURCE", "policy")
        for key in PANGU_KEYS[:2]:
            normalized[key] = _validate_number(values[key], key)
        for key in PANGU_KEYS[2:]:
            normalized[key] = _validate_mode(values[key], key)

    # Execution, service, transport, timeout, and parsing failures are never advisory.
    normalized["ONLINE_NON_ACCURACY_FAILURE_MODE"] = "blocking"
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--github-env", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    normalized = normalize_environment(read_github_env(args.env_file))
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"policy-json={payload}\n")
    if args.github_env:
        with args.github_env.open("a", encoding="utf-8") as stream:
            for key, value in normalized.items():
                if "\n" in value or "\r" in value:
                    raise PolicyError(f"multiline environment value for {key}")
                stream.write(f"{key}={value}\n")
    if not args.github_output and not args.github_env:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
