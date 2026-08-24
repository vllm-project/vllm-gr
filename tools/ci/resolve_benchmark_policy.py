#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate and resolve a branch-owned benchmark gate policy."""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO, cast

import yaml

SCHEMA_VERSION = 1
FAILURE_MODES = {"blocking", "warning"}
ACCURACY_KEYS = (
    "pass_at_1_min",
    "pass_at_32_min",
    "recall_at_32_min",
)
PERFORMANCE_KEYS = (
    "online_elapsed_seconds_max",
    "offline_elapsed_seconds_max",
    "p99_ttft_max",
    "p99_tpot_max",
    "avg_decode_time_max",
)
FAILURE_MODE_KEYS = (
    "accuracy",
    "performance",
    "online_non_accuracy",
)

STRICT_BOOTSTRAP_POLICY: dict[str, Any] = {
    "version": SCHEMA_VERSION,
    "name": "stable-bootstrap",
    "thresholds": {
        "accuracy": {
            "pass_at_1_min": 0.09,
            "pass_at_32_min": 0.29,
            "recall_at_32_min": 0.04,
        },
        "performance": {
            "online_elapsed_seconds_max": 17,
            "offline_elapsed_seconds_max": 20,
            "p99_ttft_max": 300,
            "p99_tpot_max": 0.02,
            "avg_decode_time_max": 120,
        },
    },
    "failure_modes": {
        "accuracy": "blocking",
        "performance": "blocking",
        "online_non_accuracy": "blocking",
    },
    "device_overrides": {
        "npu": {
            "boards": {
                "Ascend910_9362": {
                    "thresholds": {
                        "performance": {
                            "p99_ttft_max": 280,
                            "online_elapsed_seconds_max": 20,
                            "offline_elapsed_seconds_max": 25,
                        }
                    }
                }
            }
        }
    },
}


class PolicyError(ValueError):
    """Raised when a benchmark policy violates its schema."""


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise PolicyError(f"{path} keys must be strings")
    return cast(dict[str, Any], value)


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise PolicyError(f"{path} is missing required keys: {', '.join(missing)}")
    if extra:
        raise PolicyError(f"{path} has unsupported keys: {', '.join(extra)}")


def _number(value: object, path: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{path} must be a number")
    if not math.isfinite(float(value)) or value < 0:
        raise PolicyError(f"{path} must be finite and non-negative")
    return value


def _validate_threshold_group(
    value: object,
    keys: tuple[str, ...],
    path: str,
    *,
    partial: bool = False,
) -> dict[str, float | int]:
    group = _mapping(value, path)
    expected = set(keys)
    extra = sorted(set(group) - expected)
    if extra:
        raise PolicyError(f"{path} has unsupported keys: {', '.join(extra)}")
    if not partial:
        missing = sorted(expected - set(group))
        if missing:
            raise PolicyError(f"{path} is missing required keys: {', '.join(missing)}")
    return {key: _number(raw, f"{path}.{key}") for key, raw in group.items()}


def _validate_thresholds(value: object, path: str, *, partial: bool = False) -> dict[str, Any]:
    thresholds = _mapping(value, path)
    expected = {"accuracy", "performance"}
    extra = sorted(set(thresholds) - expected)
    if extra:
        raise PolicyError(f"{path} has unsupported keys: {', '.join(extra)}")
    if not partial:
        missing = sorted(expected - set(thresholds))
        if missing:
            raise PolicyError(f"{path} is missing required keys: {', '.join(missing)}")

    validated: dict[str, Any] = {}
    if "accuracy" in thresholds:
        validated["accuracy"] = _validate_threshold_group(
            thresholds["accuracy"], ACCURACY_KEYS, f"{path}.accuracy", partial=partial
        )
    if "performance" in thresholds:
        validated["performance"] = _validate_threshold_group(
            thresholds["performance"],
            PERFORMANCE_KEYS,
            f"{path}.performance",
            partial=partial,
        )
    return validated


def _validate_policy(raw_policy: object) -> dict[str, Any]:
    policy = _mapping(raw_policy, "policy")
    _require_exact_keys(
        policy,
        {"version", "name", "thresholds", "failure_modes", "device_overrides"},
        "policy",
    )

    if isinstance(policy["version"], bool) or policy["version"] != SCHEMA_VERSION:
        raise PolicyError(f"policy.version must be {SCHEMA_VERSION}, got {policy['version']!r}")
    if not isinstance(policy["name"], str) or not policy["name"].strip():
        raise PolicyError("policy.name must be a non-empty string")

    thresholds = _validate_thresholds(policy["thresholds"], "policy.thresholds")
    failure_modes = _mapping(policy["failure_modes"], "policy.failure_modes")
    _require_exact_keys(failure_modes, set(FAILURE_MODE_KEYS), "policy.failure_modes")
    for key, mode in failure_modes.items():
        if mode not in FAILURE_MODES:
            raise PolicyError(
                f"policy.failure_modes.{key} must be blocking or warning, got {mode!r}"
            )

    overrides = _mapping(policy["device_overrides"], "policy.device_overrides")
    extra_devices = sorted(set(overrides) - {"gpu", "npu"})
    if extra_devices:
        raise PolicyError(
            "policy.device_overrides has unsupported devices: " + ", ".join(extra_devices)
        )
    validated_overrides: dict[str, Any] = {}
    for device, raw_device_override in overrides.items():
        device_override = _mapping(raw_device_override, f"policy.device_overrides.{device}")
        _require_exact_keys(device_override, {"boards"}, f"policy.device_overrides.{device}")
        boards = _mapping(device_override["boards"], f"policy.device_overrides.{device}.boards")
        validated_boards: dict[str, Any] = {}
        for board, raw_board_override in boards.items():
            if not board:
                raise PolicyError(
                    f"policy.device_overrides.{device}.boards contains an empty board ID"
                )
            board_override = _mapping(
                raw_board_override,
                f"policy.device_overrides.{device}.boards.{board}",
            )
            _require_exact_keys(
                board_override,
                {"thresholds"},
                f"policy.device_overrides.{device}.boards.{board}",
            )
            validated_boards[board] = {
                "thresholds": _validate_thresholds(
                    board_override["thresholds"],
                    f"policy.device_overrides.{device}.boards.{board}.thresholds",
                    partial=True,
                )
            }
        validated_overrides[device] = {"boards": validated_boards}

    return {
        "version": SCHEMA_VERSION,
        "name": policy["name"].strip(),
        "thresholds": thresholds,
        "failure_modes": dict(failure_modes),
        "device_overrides": validated_overrides,
    }


def load_policy(path: Path) -> dict[str, Any]:
    """Load and validate a policy file."""
    try:
        raw_policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PolicyError(f"could not read benchmark policy {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PolicyError(f"invalid YAML in benchmark policy {path}: {exc}") from exc
    return _validate_policy(raw_policy)


def resolve_policy(policy: dict[str, Any], device: str, board_id: str) -> dict[str, str]:
    """Apply a device/board override and return benchmark environment values."""
    if device not in {"gpu", "npu"}:
        raise PolicyError(f"device must be gpu or npu, got {device!r}")
    validated = _validate_policy(policy)
    thresholds = deepcopy(validated["thresholds"])
    board_override = validated["device_overrides"].get(device, {}).get("boards", {}).get(board_id)
    if board_override:
        for group, values in board_override["thresholds"].items():
            thresholds[group].update(values)

    accuracy = thresholds["accuracy"]
    performance = thresholds["performance"]
    failure_modes = validated["failure_modes"]
    return {
        "BENCHMARK_POLICY_NAME": validated["name"],
        "BENCHMARK_DEVICE": device,
        "BENCHMARK_BOARD_ID": board_id,
        "ACCURACY_FAILURE_MODE": failure_modes["accuracy"],
        "PERFORMANCE_FAILURE_MODE": failure_modes["performance"],
        "ONLINE_NON_ACCURACY_FAILURE_MODE": failure_modes["online_non_accuracy"],
        "PASS_AT_1_THRESHOLD": str(accuracy["pass_at_1_min"]),
        "PASS_AT_32_THRESHOLD": str(accuracy["pass_at_32_min"]),
        "RECALL_AT_32_THRESHOLD": str(accuracy["recall_at_32_min"]),
        "ELAPSED_TIME_ONLINE_THRESHOLD": str(performance["online_elapsed_seconds_max"]),
        "ELAPSED_TIME_OFFLINE_THRESHOLD": str(performance["offline_elapsed_seconds_max"]),
        "P99_TTFT_THRESHOLD": str(performance["p99_ttft_max"]),
        "P99_TPOT_THRESHOLD": str(performance["p99_tpot_max"]),
    }


def write_github_env(values: dict[str, str], stream: TextIO) -> None:
    """Write simple validated key/value pairs to a GitHub environment file."""
    invalid_keys = [key for key, value in values.items() if "\n" in value or "\r" in value]
    if invalid_keys:
        raise PolicyError("resolved values contain a newline: " + ", ".join(sorted(invalid_keys)))
    stream.writelines(f"{key}={value}\n" for key, value in values.items())


def write_summary(values: dict[str, str], source_ref: str, bootstrap: bool, stream: TextIO) -> None:
    """Append the effective policy to the GitHub step summary."""
    stream.write("### Benchmark gate policy\n\n")
    if bootstrap:
        stream.write(
            "> [!WARNING]\n"
            "> The target base predates branch-owned benchmark policies. "
            "Using the one-time strict bootstrap policy.\n\n"
        )
    stream.write(f"Source ref: `{source_ref}`\n\n")
    rows = (
        ("Policy", values["BENCHMARK_POLICY_NAME"]),
        ("Device", values["BENCHMARK_DEVICE"]),
        ("Board", values["BENCHMARK_BOARD_ID"]),
        ("Accuracy failures", values["ACCURACY_FAILURE_MODE"]),
        ("Performance failures", values["PERFORMANCE_FAILURE_MODE"]),
        ("Online non-accuracy failures", values["ONLINE_NON_ACCURACY_FAILURE_MODE"]),
        ("pass@1 minimum", values["PASS_AT_1_THRESHOLD"]),
        ("pass@32 minimum", values["PASS_AT_32_THRESHOLD"]),
        ("recall@32 minimum", values["RECALL_AT_32_THRESHOLD"]),
        ("Online elapsed maximum", values["ELAPSED_TIME_ONLINE_THRESHOLD"] + "s"),
        ("Offline elapsed maximum", values["ELAPSED_TIME_OFFLINE_THRESHOLD"] + "s"),
        ("P99 TTFT maximum", values["P99_TTFT_THRESHOLD"]),
        ("P99 TPOT maximum", values["P99_TPOT_THRESHOLD"]),
    )
    stream.write("| Setting | Effective value |\n| --- | --- |\n")
    stream.writelines(f"| {label} | `{value}` |\n" for label, value in rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy-file", type=Path)
    source.add_argument("--bootstrap-strict", action="store_true")
    parser.add_argument("--device", required=True, choices=("gpu", "npu"))
    parser.add_argument("--board-id", default="unknown")
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--github-env", required=True, type=Path)
    parser.add_argument("--github-step-summary", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    policy = (
        _validate_policy(deepcopy(STRICT_BOOTSTRAP_POLICY))
        if args.bootstrap_strict
        else load_policy(args.policy_file)
    )
    values = resolve_policy(policy, args.device, args.board_id)
    with args.github_env.open("a", encoding="utf-8") as env_stream:
        write_github_env(values, env_stream)
    with args.github_step_summary.open("a", encoding="utf-8") as summary_stream:
        write_summary(values, args.source_ref, args.bootstrap_strict, summary_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
