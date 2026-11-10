# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the PR workflow queue, independent of a live Actions runner."""

from pathlib import Path

import pytest
import yaml

from tests.test_ready_label_ci import context, render

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def test_gate_runs_for_different_prs_share_one_preserving_queue() -> None:
    queue = _workflow("pre-commit.yml")["concurrency"]
    assert queue["queue"] == "max"
    assert queue.get("cancel-in-progress", False) is False
    events = [
        context("synchronize", number=481, base_ref="main"),
        context("synchronize", number=482, base_ref="main"),
        context("labeled", number=483, base_ref="main", label="ready"),
        context("submitted", number=484, event="pull_request_review", base_ref="main"),
        context("synchronize", number=485, base_ref="decode_graph"),
        context("labeled", number=486, abtest=True, label="abtest", base_ref="main"),
        context(event="workflow_dispatch", ready=False, draft=True),
    ]
    groups = {render(queue["group"], github).casefold() for github in events}
    assert len(groups) == 1, groups


def test_irrelevant_label_event_does_not_block_pr_gate_queue() -> None:
    queue = _workflow("pre-commit.yml")["concurrency"]
    reference = render(queue["group"], context()).casefold()
    unrelated = context("labeled", label="bug")
    first = render(queue["group"], unrelated).casefold()
    unrelated.run_id += 1
    second = render(queue["group"], unrelated).casefold()
    assert first != reference
    assert first != second


def test_workflow_lock_is_distinct_from_its_owned_job_locks() -> None:
    workflow = _workflow("pre-commit.yml")
    github = context()
    workflow_group = render(workflow["concurrency"]["group"], github).casefold()
    for name, job in workflow["jobs"].items():
        if "concurrency" in job:
            job_group = render(job["concurrency"]["group"], github).casefold()
            assert job_group != workflow_group, f"{name} waits on its own workflow lock"


def test_wheel_and_decode_remain_in_full_gate() -> None:
    workflow = _workflow("pre-commit.yml")
    wheel = workflow["jobs"]["build"]
    assert wheel["runs-on"] == "ubuntu-latest"
    assert set(wheel["strategy"]["matrix"]["python-version"]) == {"3.11", "3.12", "3.13"}
    assert set(wheel["strategy"]["matrix"]["target-device"]) == {"cpu", "cuda"}
    assert any("tools/run_wheel_smoke_test.sh" in step.get("run", "") for step in wheel["steps"])
    assert any(
        step.get("uses", "").startswith("actions/upload-artifact@") for step in wheel["steps"]
    )
    assert set(workflow["jobs"]["decode-benchmark"]["needs"]) == {"benchmark-policy", "pre-commit"}
    assert workflow["jobs"]["decode-benchmark"]["runs-on"] == "self-hosted"
    assert "build" in workflow["jobs"]["benchmark-blocking"]["needs"]


def test_standalone_wheel_remains_manually_dispatchable() -> None:
    workflow = _workflow("build_wheel.yml")
    triggers = workflow.get("on", workflow.get(True))
    assert "workflow_dispatch" in triggers
    assert "pull_request" not in triggers
