# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Observe pytest without changing test selection, fixtures, or assertions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_report: dict[str, Any] = {
    "selected": [],
    "reports": [],
    "collection_errors": [],
    "collection_skips": [],
}


def _save() -> None:
    path = Path(os.environ["VLLM_GR_NPU_TEST_REPORT"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pytest_collection_finish(session: Any) -> None:
    _report["selected"] = [
        {"nodeid": item.nodeid, "markers": sorted({mark.name for mark in item.iter_markers()})}
        for item in session.items
    ]
    _report["collect_only"] = bool(session.config.option.collectonly)
    _report["plugins"] = sorted(
        str(name) for name, _plugin in session.config.pluginmanager.list_name_plugin()
    )
    _save()


def pytest_collectreport(report: Any) -> None:
    if report.skipped:
        _report["collection_skips"].append(
            {"nodeid": report.nodeid, "detail": str(report.longrepr)}
        )
        _save()
    if report.failed:
        _report["collection_errors"].append(
            {"nodeid": report.nodeid, "detail": str(report.longrepr)}
        )
        _save()


def pytest_runtest_logreport(report: Any) -> None:
    _report["reports"].append(
        {
            "nodeid": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "wasxfail": getattr(report, "wasxfail", None),
            "duration_seconds": report.duration,
            "detail": str(report.longrepr) if report.longrepr else None,
        }
    )
    _save()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    _report["exitstatus"] = int(exitstatus)
    _save()
