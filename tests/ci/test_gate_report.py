# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-def"
"""Exercise strict reporting through real nested pytest sessions."""

import json
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT
from tools.ci import run_gate

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
pytest_plugins = ("pytester",)


@pytest.fixture(autouse=True)
def reset_vocab_slice_config() -> None:
    pass


@pytest.mark.parametrize(
    "body,passed",
    [
        ("def test_ok(): pass", True),
        ("import pytest\n@pytest.mark.xfail(strict=False)\ndef test_xpass(): pass", False),
        ("def test_fail(): assert False", False),
        ("import pytest\ndef test_skip(): pytest.skip('unavailable')", False),
        ("import pytest\n@pytest.mark.xfail\ndef test_xfail(): assert False", False),
        (
            "import pytest\n@pytest.fixture(autouse=True)\ndef setup(): raise RuntimeError('bad')\n"
            "def test_setup(): pass",
            False,
        ),
        (
            "import pytest\n@pytest.fixture(autouse=True)\ndef setup():\n yield\n"
            " raise RuntimeError('cleanup failed')\ndef test_teardown(): pass",
            False,
        ),
        ("def test_cancel(): raise KeyboardInterrupt", False),
        ("# no tests", False),
    ],
)
def test_report_rejects_incomplete_execution(
    pytester: pytest.Pytester, body: str, passed: bool
) -> None:
    pytester.makeconftest(
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\npytest_plugins = ('tools.ci.strict_gate',)"
    )
    pytester.makepyfile(body)
    assert (pytester.runpytest_subprocess().ret == 0) is passed


@pytest.mark.parametrize(
    "mode,expected",
    [("pass", 0), ("fail", 1), ("timeout", 1), ("cancel", 130)],
)
def test_runner_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, expected: int
) -> None:
    import subprocess

    def execute(command, **kwargs):
        if mode == "cancel":
            raise KeyboardInterrupt
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        return 1 if mode == "fail" else 0

    monkeypatch.setenv("PYTEST_ADDOPTS", "-k test_good")
    monkeypatch.setattr(run_gate, "run_gate_process", execute)
    output = tmp_path / "run"
    assert (
        run_gate.main(["--suite", "unit", "--hardware", "host", "--output", str(output)])
        == expected
    )
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == (
        "passed" if not expected else "cancelled" if expected == 130 else "failed"
    )
    with pytest.raises(FileExistsError):
        run_gate.main(["--suite", "unit", "--hardware", "host", "--output", str(output)])


@pytest.mark.parametrize(
    "hardware,cards,expected",
    [("cuda:H100", 1, ["single"]), ("cuda:H100", 2, ["pair", "masked"]), ("npu:A3", 2, ["pair"])],
)
def test_resource_lane_runs_only_owned_scenarios(monkeypatch, tmp_path, hardware, cards, expected):
    from tests.helpers.hardware import DeviceInventory

    backend, sku = hardware.split(":")
    name = "Ascend910_9391" if backend == "npu" else f"NVIDIA {sku}"
    monkeypatch.setattr(
        run_gate, "probe_devices", lambda _: DeviceInventory(backend, (name,) * cards)
    )

    def execute(command, **kwargs):
        assert [
            node.split("[")[-1][:-1] for node in command if "::test_resource_probe" in node
        ] == expected
        return 0

    monkeypatch.setattr(run_gate, "run_gate_process", execute)
    assert (
        run_gate.main(
            [
                "--suite",
                "resource",
                "--hardware",
                hardware,
                "--cards",
                str(cards),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
