# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run documentation classifier regression tests alongside the Python suite."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_docs_only_classifier() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the documentation classifier tests")
    assert node is not None
    root = Path(__file__).resolve().parents[1]
    subprocess.run([node, "--test", "tests/test_docs_only.js"], cwd=root, check=True)
