# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
from pathlib import Path

HELPERS = Path("tools/ci/benchmark_gate_helpers.sh").resolve()


def run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "$1"; {command}', "bash", str(HELPERS)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_accuracy_failure_mode_controls_result() -> None:
    warning = run_bash("validate_accuracy_metric pass@1 0.08 0.09 warning gpu 86")
    blocking = run_bash("validate_accuracy_metric pass@1 0.08 0.09 blocking gpu 86")

    assert warning.returncode == 0
    assert "::warning title=Benchmark accuracy regression::" in warning.stdout
    assert blocking.returncode == 86
    assert "FAILURE on gpu" in blocking.stdout


def test_performance_failure_mode_controls_result() -> None:
    warning = run_bash("validate_performance_metric TTFT 301 300 warning npu 87")
    blocking = run_bash("validate_performance_metric TTFT 301 300 blocking npu 87")

    assert warning.returncode == 0
    assert "::warning title=Benchmark performance regression::" in warning.stdout
    assert blocking.returncode == 87
    assert "FAILURE on npu" in blocking.stdout
