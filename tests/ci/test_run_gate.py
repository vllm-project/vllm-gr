# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-def,no-any-return"
"""Entrypoint regressions for native selection and strict gate outcomes."""

from __future__ import annotations

import pytest

from tests.helpers.hardware import DeviceInventory
from tools.ci import run_gate

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
pytest_plugins = ("pytester",)


def test_entrypoint_selects_levels_without_reports(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return 0

    monkeypatch.setattr(run_gate, "run_gate_process", run)
    assert run_gate.main(["--suite", "unit", "--hardware", "host"]) == 0
    command, options = calls[0]
    assert command[command.index("-m") + 1] == "pytest"
    assert command[-2:] == ["-m", "l1 or l2 or l3"]
    assert "--strict-gate" not in command
    assert "log" not in options
    assert "--junitxml" not in command


def test_card_count_and_failure_propagation(monkeypatch):
    commands = []
    monkeypatch.setattr(
        run_gate, "probe_devices", lambda _: DeviceInventory("cuda", ("NVIDIA H100",))
    )

    def run(cmd, **kw):
        commands.append(cmd)
        return 7

    monkeypatch.setattr(run_gate, "run_gate_process", run)
    assert run_gate.main(["--suite", "functional", "--hardware", "cuda:H100", "--cards", "1"]) == 7
    assert "--strict-gate" in commands[0]
    with pytest.raises(SystemExit):
        run_gate.main(["--suite", "unit", "--hardware", "cuda:H100", "--cards", "2"])


@pytest.mark.parametrize("mark", ["", "@pytest.mark.skip", "@pytest.mark.xfail"])
def test_strict_gate_uses_pytest_outcomes(pytester, mark):
    pytester.makeconftest("from tests.conftest import pytest_addoption, pytest_sessionfinish")
    pytester.makepyfile(
        f"import pytest\n{mark}\ndef test_gate():\n    assert {mark != '@pytest.mark.xfail'}\n"
    )
    result = pytester.runpytest("--strict-gate")
    assert result.ret == (0 if not mark else 1)
