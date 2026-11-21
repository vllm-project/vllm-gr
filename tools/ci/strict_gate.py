# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reject incomplete execution and unexpected pytest outcomes in gate runs."""

import pytest


class GateOutcomes:
    def __init__(self) -> None:
        self.phases: dict[str, set[str]] = {phase: set() for phase in ("setup", "call", "teardown")}
        self.unexpected = False

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if not report.passed or hasattr(report, "wasxfail"):
            self.unexpected = True
        if report.passed:
            self.phases[report.when].add(report.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if exitstatus == 0 and (
            self.unexpected
            or not session.testscollected
            or any(len(items) != session.testscollected for items in self.phases.values())
        ):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(GateOutcomes())
