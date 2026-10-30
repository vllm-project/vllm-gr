# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep upstream A/B validation compatible with queued device execution."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tests.test_ready_label_ci import context, evaluate, render

ROOT = Path(__file__).resolve().parents[2]


def workflow(name: str) -> dict[str | bool, Any]:
    return cast(
        dict[str | bool, Any], yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
    )


def test_ab_label_uses_shared_device_lock_without_becoming_cpu_evidence() -> None:
    ab = workflow("beam-performance-ab.yml")
    job = ab["jobs"]["beam-performance-ab-only"]
    device_lock = render(
        workflow("device-validation.yml")["concurrency"]["group"],
        context(event="workflow_dispatch"),
        inputs={"pr_number": "481", "base_ref": "decode_graph"},
    )
    for ready in (False, True):
        for draft in (False, True):
            for label in ("abtest", "ready", "ci-priority:high"):
                event = context("labeled", ready=ready, draft=draft, label=label)
                assert bool(evaluate(job["if"], event)) == (label == "abtest" and not draft)
                if label == "abtest":
                    assert render(ab["concurrency"]["group"], event) == device_lock
                else:
                    assert render(ab["concurrency"]["group"], event) != "device-pr-481"
    assert ab["concurrency"]["cancel-in-progress"] is False
    assert render(ab["run-name"], context()) == "Beam V1 A/B #481 @merge-sha"
    triggers = workflow("ci-queue.yml")
    assert ab["name"] in triggers.get("on", triggers.get(True))["workflow_run"]["workflows"]
    assert not evaluate(job["if"], context("synchronize", label="abtest"))


@pytest.mark.parametrize(  # type: ignore[misc]
    "scenario,expected",
    [
        ("current", {"fresh": "true", "passed": "true"}),
        ("failed", {"fresh": "true", "passed": "false"}),
        ("base", {"fresh": "false", "passed": "false"}),
        ("merge", {"fresh": "false", "passed": "false"}),
        ("expired", {"fresh": "false", "passed": "false"}),
        ("missing", {"fresh": "false", "passed": "false"}),
        ("unavailable", {"fresh": "false", "passed": "false"}),
    ],
)
def test_freshness_checks_the_queued_revision(scenario: str, expected: dict[str, str]) -> None:
    steps = workflow("device-validation.yml")["jobs"]["validate"]["steps"]
    script = next(step for step in steps if step.get("id") == "ab-freshness")["with"]["script"]
    harness = """
const {script, scenario} = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const outputs = {};
const context = {repo: {owner: 'owner', repo: 'repo'}, runId: 124, sha: 'dispatcher-sha'};
const github = {rest: {actions: {
  listWorkflowRuns: async (args) => {
    if (args.workflow_id !== 'beam-performance-ab.yml' || args.head_sha !== 'base-sha' ||
        args.event !== 'pull_request_target')
      throw new Error('Lookup did not use the queued PR identity');
    if (scenario === 'unavailable') throw new Error('HTTP 403');
    return {data: {workflow_runs: [{id: 123, status: 'completed', conclusion: 'success',
      head_sha: scenario === 'base' ? 'old-base' : 'base-sha',
      display_title: `Beam V1 A/B #481 @${scenario === 'merge' ? 'old-merge' : 'current-merge'}`,
      run_attempt: 1}]}};
  },
  listJobsForWorkflowRun: async () => ({data: {jobs: [
    {name: 'Beam V1 performance A/B only', conclusion: scenario === 'failed' ? 'failure' : 'success'},
  ]}}),
  listWorkflowRunArtifacts: async () => ({data: {artifacts: scenario === 'missing' ? [] : [
    {name: 'beam-v1-performance-report-123-1', expired: scenario === 'expired'},
  ]}}),
}}};
const core = {setOutput: (key, value) => {outputs[key] = value;}, warning: () => {}, info: () => {}};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
new AsyncFunction('github', 'context', 'core', script)(github, context, core)
  .then(() => process.stdout.write(JSON.stringify(outputs)))
  .catch(error => {console.error(error); process.exitCode = 1;});
"""
    result = subprocess.run(
        ["node", "-e", harness],
        input=json.dumps({"script": script, "scenario": scenario}),
        text=True,
        capture_output=True,
        check=True,
        env={
            **os.environ,
            "PR_NUMBER": "481",
            "CI_HEAD_SHA": "current-head",
            "CI_BASE_SHA": "base-sha",
            "CI_MERGE_SHA": "current-merge",
            "CI_BASE_REF": "decode_graph",
        },
    )
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize(  # type: ignore[misc]
    "pr_number,script", [("0", "performance.sh"), ("481", "performance_ab.sh")]
)
def test_performance_runs_ab_only_for_prs(tmp_path: Path, pr_number: str, script: str) -> None:
    steps = workflow("device-validation.yml")["jobs"]["validate"]["steps"]
    step = next(item for item in steps if item.get("name", "").startswith("Measure Beam V1"))
    # Execute the workflow's shell branch with a harmless command recorder.
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", "-eu", "-c", step["run"]],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PR_NUMBER": pr_number, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )
    assert result.stdout.splitlines()[-1] == f"tests/system/run_beam_v1_{script}"
    for key, source in (("BASE", "base"), ("HEAD", "head"), ("CANDIDATE", "merge")):
        assert step["env"][f"V1_PERF_{key}_SHA"] == "${{ inputs." + source + "_sha }}"
    checkout = next(item for item in steps if item.get("uses") == "actions/checkout@v4")
    assert checkout["with"]["fetch-depth"] == 2


def test_fresh_failed_ab_report_blocks_full_validation() -> None:
    steps = workflow("device-validation.yml")["jobs"]["validate"]["steps"]
    reject = next(
        step for step in steps if step.get("name") == "Require a passing current A/B report"
    )
    for fresh, passed in ((False, False), (True, False), (True, True)):
        condition = reject["if"].replace(
            "steps.ab-freshness.outputs.fresh", repr(str(fresh).lower())
        )
        condition = condition.replace(
            "steps.ab-freshness.outputs.passed", repr(str(passed).lower())
        )
        assert bool(evaluate(condition, context())) == (fresh and not passed)
    result = subprocess.run(["bash", "-eu", "-c", reject["run"]], capture_output=True, text=True)
    assert result.returncode == 1
