# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from tools.ci.resolve_benchmark_policy import (
    STRICT_BOOTSTRAP_POLICY,
    PolicyError,
    load_policy,
    resolve_policy,
    write_github_env,
    write_summary,
)


def test_main_policy_resolves_stable_gpu_values() -> None:
    policy = load_policy(Path(".github/benchmark-gate.yaml"))

    values = resolve_policy(policy, "gpu", "unknown")

    assert values == {
        "BENCHMARK_POLICY_NAME": "stable",
        "BENCHMARK_DEVICE": "gpu",
        "BENCHMARK_BOARD_ID": "unknown",
        "ACCURACY_FAILURE_MODE": "blocking",
        "PERFORMANCE_FAILURE_MODE": "blocking",
        "ONLINE_NON_ACCURACY_FAILURE_MODE": "blocking",
        "PASS_AT_1_THRESHOLD": "0.09",
        "PASS_AT_32_THRESHOLD": "0.29",
        "RECALL_AT_32_THRESHOLD": "0.04",
        "ELAPSED_TIME_ONLINE_THRESHOLD": "17",
        "ELAPSED_TIME_OFFLINE_THRESHOLD": "20",
        "P99_TTFT_THRESHOLD": "300",
        "P99_TPOT_THRESHOLD": "0.02",
    }


def test_npu_board_override_changes_only_declared_thresholds() -> None:
    values = resolve_policy(deepcopy(STRICT_BOOTSTRAP_POLICY), "npu", "Ascend910_9362")

    assert values["P99_TTFT_THRESHOLD"] == "280"
    assert values["ELAPSED_TIME_ONLINE_THRESHOLD"] == "20"
    assert values["ELAPSED_TIME_OFFLINE_THRESHOLD"] == "25"
    assert values["P99_TPOT_THRESHOLD"] == "0.02"
    assert values["ACCURACY_FAILURE_MODE"] == "blocking"


def test_development_failure_modes_are_configurable() -> None:
    policy = deepcopy(STRICT_BOOTSTRAP_POLICY)
    policy["name"] = "development"
    policy["failure_modes"] = {
        "accuracy": "warning",
        "performance": "warning",
        "online_non_accuracy": "warning",
    }

    values = resolve_policy(policy, "gpu", "unknown")

    assert values["ACCURACY_FAILURE_MODE"] == "warning"
    assert values["PERFORMANCE_FAILURE_MODE"] == "warning"
    assert values["ONLINE_NON_ACCURACY_FAILURE_MODE"] == "warning"


@pytest.mark.parametrize(  # type: ignore[misc]
    ("mutation", "message"),
    [
        (lambda policy: policy.update(version=2), "policy.version must be 1"),
        (lambda policy: policy.update(version=True), "policy.version must be 1"),
        (
            lambda policy: policy["failure_modes"].update(accuracy="ignore"),
            "must be blocking or warning",
        ),
        (
            lambda policy: policy["thresholds"]["accuracy"].pop("pass_at_1_min"),
            "missing required keys: pass_at_1_min",
        ),
        (
            lambda policy: policy["thresholds"]["performance"].update(p99_ttft_max=-1),
            "must be finite and non-negative",
        ),
        (
            lambda policy: policy["thresholds"]["performance"].update(p99_tpot_max=float("nan")),
            "must be finite and non-negative",
        ),
        (
            lambda policy: policy.update(unexpected=True),
            "unsupported keys: unexpected",
        ),
    ],
)
def test_invalid_policies_are_rejected(mutation: Any, message: str) -> None:
    policy = deepcopy(STRICT_BOOTSTRAP_POLICY)
    mutation(policy)

    with pytest.raises(PolicyError, match=message):
        resolve_policy(policy, "gpu", "unknown")


def test_github_outputs_include_effective_policy_and_bootstrap_warning() -> None:
    values = resolve_policy(deepcopy(STRICT_BOOTSTRAP_POLICY), "gpu", "unknown")
    env_stream = StringIO()
    summary_stream = StringIO()

    write_github_env(values, env_stream)
    write_summary(values, "abc123", True, summary_stream)

    assert "ACCURACY_FAILURE_MODE=blocking\n" in env_stream.getvalue()
    assert "one-time strict bootstrap policy" in summary_stream.getvalue()
    assert "Source ref: `abc123`" in summary_stream.getvalue()
    assert "| Policy | `stable-bootstrap` |" in summary_stream.getvalue()
