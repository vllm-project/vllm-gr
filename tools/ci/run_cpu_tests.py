# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run CPU-only contracts without importing the device-wide tests/conftest.py."""

import subprocess
import sys

CPU_TESTS = (
    "tests/ci",
    "tests/test_ready_label_ci.py",
    "tests/test_benchmark_branch_policy.py",
    "tests/test_v1_gate_validation.py",
    "tests/test_v1_recall_summary.py",
    "tests/test_beam_search_v1_benchmark.py",
    "tests/test_beam_v1_performance_ab.py",
)


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                "--durations=20",
                *CPU_TESTS,
                *sys.argv[1:],
            ]
        )
    )
