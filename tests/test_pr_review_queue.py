# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for review events and repository-wide device locking."""

from pathlib import Path

import pytest
import yaml

import tests.test_ready_label_ci as ready_tests

pytestmark = pytest.mark.cpu_test
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("base", ["main", "decode_graph"])
@pytest.mark.parametrize("action", ["submitted", "edited", "dismissed"])
def test_ready_review_runs_wheel_and_requires_success(base: str, action: str) -> None:
    test = ready_tests.TestReadyLabelCI()
    test.setUp()
    workflow = test.workflows["pre-commit.yml"]
    review = ready_tests.context(action, event="pull_request_review", ready=True, base_ref=base)
    running = test.scheduled(workflow, review)
    assert "wheel-scope" in running
    assert "build" in running
    gate = workflow["jobs"]["benchmark-blocking"]
    results = {name: "success" if name in running else "skipped" for name in gate["needs"]}
    accepted = test.run_gate(results, review)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    for failure in ("failure", "cancelled", "skipped"):
        rejected = test.run_gate({**results, "build": failure}, review)
        assert rejected.returncode == 1, rejected.stdout + rejected.stderr


def test_device_jobs_share_repository_lock_distinct_from_workflow_lock() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/pre-commit.yml").read_text())
    outer = ready_tests.render(workflow["concurrency"]["group"], ready_tests.context())
    for name in (
        "pre-commit",
        "decode-benchmark",
        "pangu-offline",
        "pangu-online",
        "beam-performance-ab-only",
    ):
        lock = workflow["jobs"][name]["concurrency"]
        assert lock["group"] == "vllm-gr-accelerator-${{ github.repository }}"
        assert lock["queue"] == "max"
        assert lock.get("cancel-in-progress", False) is False
        assert (
            ready_tests.render(lock["group"], ready_tests.context()).casefold() != outer.casefold()
        )
