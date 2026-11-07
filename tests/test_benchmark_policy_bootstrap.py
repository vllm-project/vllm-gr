# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the workflow bootstrap behavior tests during the regular pytest suite."""

import shutil
import subprocess
import unittest
from pathlib import Path


class BenchmarkPolicyBootstrapTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for github-script tests")
    def test_workflow_bootstrap(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["node", "--test", "tests/test_benchmark_policy_bootstrap.cjs"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
