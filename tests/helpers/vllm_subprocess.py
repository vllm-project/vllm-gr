# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import subprocess
import sys

import pytest

from tests.helpers.paths import REPO_ROOT


def run_vllm_script(script: str, *args: str) -> None:
    """Run real-vLLM checks without re-registering Torch's opaque types."""
    prelude = """
import importlib.util

if importlib.util.find_spec("vllm") is None:
    raise SystemExit(77)
"""
    result = subprocess.run(
        [sys.executable, "-c", prelude + script, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip("vllm is not installed")
    assert result.returncode == 0, result.stdout + result.stderr
