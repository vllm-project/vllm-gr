# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_beamproc_mode.sh"


def _run_with_dataset_path(dataset_path: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if dataset_path is None:
        env.pop("DATASET_PATH", None)
    else:
        env["DATASET_PATH"] = dataset_path
    return subprocess.run(
        [str(SCRIPT), "rust"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_beamproc_mode_requires_external_dataset_path() -> None:
    result = _run_with_dataset_path(None)

    assert result.returncode == 2
    assert "DATASET_PATH must point to the OpenOneRec benchmark data" in result.stderr


def test_run_beamproc_mode_rejects_missing_dataset_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-benchmark-data"

    result = _run_with_dataset_path(str(missing_path))

    assert result.returncode == 2
    assert f"DATASET_PATH is not a directory: {missing_path}" in result.stderr
