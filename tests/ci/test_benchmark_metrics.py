# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
READER = REPO_ROOT / "tools/ci/read_benchmark_metrics.py"


def test_accuracy_metrics_require_exact_numeric_fields(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    command = [sys.executable, str(READER), "accuracy", str(path)]
    fields = '"pass@1": 0.1, "position1_pass@32": 0.2, "recall@32": 0.3, "elapsed_time": 2.5'
    for suffix, good in [
        (', "pass@32": 0.8', True),
        ("", False),
        (', "pass@32": "n/a"', False),
        (', "pass@32": true', False),
        (', "pass@32": -1', False),
        (', "pass@32": 32', False),
        (', "pass@32": NaN', False),
        (', "pass@32": 0.5, "pass@32": 1', False),
    ]:
        path.write_text("{" + fields + suffix + "}")
        result = subprocess.run(command, capture_output=True, text=True)
        assert (result.returncode == 0) is good, result.stderr
        if good:
            assert list(map(float, result.stdout.split())) == [0.1, 0.8, 0.3, 2.5]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("failed", "ttft", "tpot", "expected"),
    [
        (0, 12.406, 0.024, ["0", "12.41", "0.02"]),
        (0, 12.406, 0.026, ["0", "12.41", "0.03"]),
        (1, 0, 0, ["1", "0.00", "0.00"]),
    ],
)
def test_serving_metrics_keep_console_precision(
    tmp_path: Path, failed: int, ttft: float, tpot: float, expected: list[str]
) -> None:
    path = tmp_path / "serving.json"
    contents = f'{{"p99_tpot_ms": {tpot}, "p99_ttft_ms": {ttft}, "failed": {failed}}}'
    path.write_text(contents, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, READER, "serving", path], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == expected
    assert path.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize(  # type: ignore[misc]
    ("contents", "error"),
    [
        (None, "serving.json"),
        ("{", "Expecting"),
        ("[]", "metrics object"),
        ('{"p99_ttft_ms": 1, "p99_tpot_ms": 2}', "failed"),
        ('{"failed": true, "p99_ttft_ms": 1, "p99_tpot_ms": 2}', "failed"),
        ('{"failed": -1, "p99_ttft_ms": 1, "p99_tpot_ms": 2}', "failed"),
        ('{"failed": 1.0, "p99_ttft_ms": 1, "p99_tpot_ms": 2}', "failed"),
        ('{"failed": "1", "p99_ttft_ms": 1, "p99_tpot_ms": 2}', "failed"),
        ('{"failed": 0, "p99_tpot_ms": 2}', "p99_ttft_ms"),
        ('{"failed": 0, "p99_ttft_ms": 1}', "p99_tpot_ms"),
        ('{"failed": 0, "p99_ttft_ms": 1, "p99_ttft_ms": 2}', "duplicate field p99_ttft_ms"),
        ('{"failed": 0, "p99_ttft_ms": false, "p99_tpot_ms": 2}', "p99_ttft_ms"),
        ('{"failed": 0, "p99_ttft_ms": "1", "p99_tpot_ms": 2}', "p99_ttft_ms"),
        ('{"failed": 0, "p99_ttft_ms": -0.001, "p99_tpot_ms": 2}', "p99_ttft_ms"),
        ('{"failed": 0, "p99_ttft_ms": 1, "p99_tpot_ms": NaN}', "p99_tpot_ms"),
        ('{"failed": 0, "p99_ttft_ms": 1, "p99_tpot_ms": Infinity}', "p99_tpot_ms"),
    ],
)
def test_serving_metrics_reject_invalid_json(
    tmp_path: Path, contents: str | None, error: str
) -> None:
    path = tmp_path / "serving.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, READER, "serving", path], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert not result.stdout
    assert str(path) in result.stderr
    assert error in result.stderr
