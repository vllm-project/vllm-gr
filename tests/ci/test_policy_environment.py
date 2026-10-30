# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the single job receives every normalized benchmark policy field."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci.normalize_benchmark_policy_env import (
    normalize_environment,
    read_github_env,
)
from tools.ci.resolve_benchmark_policy import load_policy, resolve_policy


@pytest.mark.parametrize("device", ["gpu", "npu"])  # type: ignore[misc]
def test_normalized_policy_is_exported_to_step_environment(tmp_path: Path, device: str) -> None:
    root = Path(__file__).resolve().parents[2]
    values = resolve_policy(load_policy(root / ".github/benchmark-gate.yaml"), device, "unknown")
    source, output, environment = (tmp_path / name for name in ("source", "output", "env"))
    source.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/ci/normalize_benchmark_policy_env.py"),
            "--env-file",
            str(source),
            "--github-output",
            str(output),
            "--github-env",
            str(environment),
        ],
        check=True,
    )
    expected = normalize_environment(read_github_env(source))
    assert read_github_env(environment) == expected
    assert json.loads(output.read_text().split("=", 1)[1]) == expected
    assert expected["ONLINE_NON_ACCURACY_FAILURE_MODE"] == "blocking"
