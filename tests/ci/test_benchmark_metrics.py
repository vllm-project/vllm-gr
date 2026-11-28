# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import subprocess
import sys
import warnings
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


@pytest.mark.parametrize("device", ["gpu", "npu"])  # type: ignore[misc]
def test_video_baseline_unknown_model_warns_and_uses_defaults(device: str) -> None:
    from tests.e2e.test_onerec_video_benchmark import resolve_device_baseline

    with pytest.warns(UserWarning, match="Unknown accelerator.*using the default baseline"):
        policy = resolve_device_baseline(device, "Unknown accelerator")

    assert policy["BENCHMARK_BOARD_ID"] == "Unknown accelerator"
    assert policy["BENCHMARK_DEVICE"] == device
    assert policy["PASS_AT_1_THRESHOLD"] == "0.09"
    assert policy["PASS_AT_32_THRESHOLD"] == "0.29"
    assert policy["RECALL_AT_32_THRESHOLD"] == "0.04"
    assert policy["P99_TTFT_THRESHOLD"] == "300"
    assert policy["ACCURACY_FAILURE_MODE"] == "blocking"
    assert policy["ONLINE_NON_ACCURACY_FAILURE_MODE"] == "blocking"
    assert policy["P99_TTFT_FAILURE_MODE"] == "warning"


@pytest.mark.parametrize(  # type: ignore[misc]
    ("device", "model", "board", "accuracy", "ttft", "offline_elapsed"),
    [
        ("gpu", " nvidia l4 ", "NVIDIA L4", ("0.09", "0.29", "0.04"), "300", "20"),
        ("npu", " ascend910_9362 ", "Ascend910_9362", ("0.05", "0.21", "0.03"), "280", "25"),
    ],
)
def test_video_baseline_known_model_preserves_overrides(
    device: str,
    model: str,
    board: str,
    accuracy: tuple[str, str, str],
    ttft: str,
    offline_elapsed: str,
) -> None:
    from tests.e2e.test_onerec_video_benchmark import resolve_device_baseline

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        policy = resolve_device_baseline(device, model)

    assert not emitted
    assert policy["BENCHMARK_BOARD_ID"] == board
    accuracy_keys = ("PASS_AT_1_THRESHOLD", "PASS_AT_32_THRESHOLD", "RECALL_AT_32_THRESHOLD")
    assert tuple(policy[key] for key in accuracy_keys) == accuracy
    assert policy["P99_TTFT_THRESHOLD"] == ttft
    assert policy["ELAPSED_TIME_OFFLINE_THRESHOLD"] == offline_elapsed
