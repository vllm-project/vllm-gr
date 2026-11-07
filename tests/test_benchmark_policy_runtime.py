# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run policy runtime behavior tests as part of the Python test suite."""

import shutil
import subprocess
import unittest
from pathlib import Path


class BenchmarkPolicyRuntimeTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        subprocess.run(
            ["node", "--test", "tests/test_benchmark_policy_runtime.js"],
            cwd=root,
            check=True,
        )
