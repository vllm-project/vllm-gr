# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT

pytestmark = pytest.mark.cpu_test

VERIFIER = REPO_ROOT / "examples" / "online_serving" / "catalog" / "verify_catalog.py"
SID_TOKENS = ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"]
SID = "".join(SID_TOKENS)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("response", "trailer", "count", "exit_code"),
    [
        ({"choices": [{"text": SID}]}, "", 1, 0),
        ({"choices": [{"message": {"content": SID}}]}, " total time:0.35s", 1, 0),
        ({"error": {"message": "bad request"}}, "", 1, 1),
        ({}, "", 1, 1),
        ({"choices": []}, "", 1, 1),
        ({"choices": [{"message": {"content": ""}}]}, "", 1, 1),
        ({"choices": [{"text": "<s_a_1>"}]}, "", 1, 1),
        ({"choices": [{"text": SID.replace("<s_c_3>", "<s_c_4>")}]}, "", 1, 1),
        ({"choices": [{"text": SID}]}, "", 2, 1),
    ],
)
def test_catalog_verifier_reports_failure_for_invalid_results(
    tmp_path: Path, response: dict[str, object], trailer: str, count: int, exit_code: int
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(response) + trailer, encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps([SID_TOKENS]), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            str(result_path),
            str(catalog_path),
            "--expected-count",
            str(count),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == exit_code, completed.stderr
    assert ("SUCCESS:" in completed.stdout) == (exit_code == 0)
    assert ("FAILURE:" in completed.stderr) == (exit_code != 0)
