# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the workflow guards against PR lifecycle events without a runner."""

import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import regex as re
import yaml

from tools.ci.run_cpu_tests import CPU_TESTS

ROOT = Path(__file__).resolve().parents[1]


def namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    return value


def context(
    action: str = "synchronize",
    *,
    ready: bool = True,
    draft: bool = False,
    label: str = "",
    event: str = "pull_request",
) -> Any:
    return namespace(
        {
            "workflow": "test",
            "event_name": event,
            "run_id": 123,
            "ref": "refs/pull/481/merge",
            "ref_name": "481/merge",
            "sha": "merge-sha",
            "event": {
                "action": action,
                "repository": {"id": 1},
                "label": {"name": label},
                "pull_request": {
                    "number": 481,
                    "draft": draft,
                    "labels": [{"name": "ready"}] if ready else [],
                    "head": {"sha": "head-sha", "repo": {"id": 1}},
                    "base": {"ref": "decode_graph", "sha": "base-sha", "repo": {"id": 1}},
                    "merge_commit_sha": "merge-sha",
                },
            },
        }
    )


def evaluate(
    expression: str,
    github: Any,
    results: dict[str, str] | None = None,
    *,
    cancelled: bool = False,
    inputs: dict[str, str] | None = None,
) -> Any:
    # Evaluate only the small expression subset used by these tracked workflow
    # guards. Keep the expressions in YAML as the source of truth.
    expression = expression.strip().removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("github.event.pull_request.labels.*.name", "labels")
    expression = re.sub(r"needs\.([\w-]+)\.result", r"results['\1']", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    expression = re.sub(r"\btrue\b", "True", expression)
    expression = re.sub(r"\bfalse\b", "False", expression)
    return eval(
        " ".join(expression.split()),
        {"__builtins__": {}},
        {
            "github": github,
            "inputs": namespace(inputs or {}),
            "labels": [item["name"] for item in github.event.pull_request.labels],
            "results": results or {},
            "always": lambda: True,
            "cancelled": lambda: cancelled,
            "contains": lambda items, value: value in items,
            "format": lambda template, *args: template.format(*args),
        },
    )


def render(template: str, github: Any, *, inputs: dict[str, str] | None = None) -> str:
    rendered = re.sub(
        r"\$\{\{(.*?)\}\}",
        lambda match: str(evaluate(match.group(1), github, inputs=inputs)),
        template,
        flags=re.DOTALL,
    )
    assert isinstance(rendered, str)
    return rendered


class TestReadyLabelCI(unittest.TestCase):
    def setUp(self) -> None:
        self.workflows = {
            path.name: yaml.safe_load(path.read_text())
            for path in (ROOT / ".github/workflows").glob("*.yml")
        }

    def test_cpu_lifecycle_and_priority_label_isolation(self) -> None:
        workflow = self.workflows["pre-commit.yml"]
        job = workflow["jobs"]["fast-checks"]
        triggers = workflow.get("on", workflow.get(True))["pull_request"]["types"]
        for action in (
            "opened",
            "synchronize",
            "reopened",
            "ready_for_review",
            "converted_to_draft",
        ):
            self.assertIn(action, triggers)
            for ready in (False, True):
                for draft in (False, True):
                    self.assertTrue(evaluate(job["if"], context(action, ready=ready, draft=draft)))
        for action in ("labeled", "unlabeled"):
            normal = context(action, label="ready")
            self.assertTrue(evaluate(job["if"], normal))
            for label in (
                "bug",
                "ci-priority:high",
                "ci-priority:normal",
                "ci-priority:low",
                "automerge",
                "ci-preview",
            ):
                ignored = context(action, label=label)
                self.assertFalse(evaluate(job["if"], ignored))
                self.assertNotEqual(
                    render(workflow["concurrency"]["group"], normal),
                    render(workflow["concurrency"]["group"], ignored),
                )
                self.assertTrue(render(workflow["run-name"], ignored).startswith("ci-ignored:"))
        self.assertEqual(job["runs-on"], "ubuntu-latest")

    def test_existing_wheel_ready_gate(self) -> None:
        workflow = self.workflows["build_wheel.yml"]
        job = workflow["jobs"]["build"]
        for ready in (False, True):
            for draft in (False, True):
                self.assertEqual(
                    bool(evaluate(job["if"], context(ready=ready, draft=draft))),
                    ready and not draft,
                )
        self.assertEqual(job["runs-on"], "ubuntu-latest")

    def test_one_non_preemptible_device_job(self) -> None:
        workflow = self.workflows["device-validation.yml"]
        self.assertEqual(set(workflow["jobs"]), {"preflight", "validate"})
        self.assertEqual(workflow["jobs"]["preflight"]["runs-on"], "ubuntu-latest")
        self.assertEqual(workflow["jobs"]["validate"]["needs"], "preflight")
        self.assertIs(workflow["concurrency"]["cancel-in-progress"], False)
        self.assertEqual(
            set(workflow.get("on", workflow.get(True))), {"workflow_dispatch", "workflow_call"}
        )
        job = workflow["jobs"]["validate"]
        self.assertNotIn("strategy", job)
        steps = job["steps"]
        names = [step.get("name") for step in steps]
        required = [
            "Device unit and end-to-end tests",
            "Gate Beam V1 B=4 OneRec Recall on fixed samples",
            "Measure Beam V1 offline batch and online concurrency",
            "Run Pangu offline accuracy gate",
            "Pangu online accuracy and API contract",
            "Check clear device",
            "Complete device validation",
        ]
        self.assertEqual([name for name in names if name in required], required)
        self.assertEqual(names[-1], "Complete device validation")
        device_command = next(step["run"] for step in steps if step.get("id") == "device-tests")
        for path in CPU_TESTS:
            self.assertIn(f"--ignore={path}", device_command)
        self.assertNotIn("if", steps[-1])  # implicit success() rejects failures and cancellation
        self.assertTrue(all(not step.get("continue-on-error", False) for step in steps))
        self.assertEqual(
            sum(step.get("uses") == "./.github/actions/setup-device-env" for step in steps), 1
        )
        cleanup = next(step for step in steps if step.get("name") == "Check clear device")
        for outcome in ("success", "failure", "skipped"):
            guard = cleanup["if"].replace("steps.cuda-mapping.outcome", repr(outcome))
            self.assertEqual(bool(evaluate(guard, context(), cancelled=True)), outcome == "success")

    def test_privileged_queue_uses_only_default_branch(self) -> None:
        workflow = self.workflows["ci-queue.yml"]
        self.assertIs(workflow["concurrency"]["cancel-in-progress"], False)
        completed = workflow.get("on", workflow.get(True))["workflow_run"]
        self.assertIn(
            self.workflows["device-validation-preview.yml"]["name"], completed["workflows"]
        )
        self.assertIn("completed", completed["types"])
        checkout = workflow["jobs"]["dispatch"]["steps"][0]
        self.assertEqual(checkout["with"]["ref"], "${{ github.event.repository.default_branch }}")
        self.assertIs(checkout["with"]["persist-credentials"], False)
        retired = self.workflows["retest-router.yml"]
        self.assertNotIn("issue_comment", retired.get("on", retired.get(True)))
        queue = (ROOT / "tools/ci/queue.cjs").read_text()
        self.assertNotIn("cancelWorkflowRun", queue)
        self.assertNotIn("pulls.merge", queue)

    def test_device_lock_covers_all_snapshots_and_entry_points_of_one_pr(self) -> None:
        workflow = self.workflows["device-validation.yml"]
        group = workflow["concurrency"]["group"]
        first = {
            "pr_number": "481",
            "base_ref": "decode_graph",
            "head_sha": "1" * 40,
            "base_sha": "2" * 40,
        }
        changed = {**first, "base_ref": "main", "head_sha": "3" * 40, "base_sha": "4" * 40}
        original = render(group, context(event="workflow_dispatch"), inputs=first)
        self.assertEqual(original, render(group, context(), inputs=changed))
        self.assertNotEqual(
            original, render(group, context(), inputs={**first, "pr_number": "482"})
        )
        self.assertNotEqual(
            render(group, context(), inputs={**first, "pr_number": "0"}),
            render(group, context(), inputs={**changed, "pr_number": "0"}),
        )

    def test_preview_priority_label_changes_cannot_replace_a_pending_preview(self) -> None:
        group = self.workflows["device-validation-preview.yml"]["concurrency"]["group"]
        original = render(group, context("labeled", label="ci-preview"))
        for label in ("ci-priority:high", "ci-priority:normal", "ci-priority:low", "ready", "bug"):
            ignored = context("labeled", label=label)
            self.assertNotEqual(original, render(group, ignored))

    def test_preview_priority_defaults_to_low_and_highest_label_wins(self) -> None:
        priority = self.workflows["device-validation-preview.yml"]["jobs"]["preview"]["with"][
            "priority"
        ]
        for labels, expected in (
            ([], "low"),
            (["ci-priority:low"], "low"),
            (["ci-priority:normal", "ci-priority:low"], "normal"),
            (["ci-priority:high", "ci-priority:normal", "ci-priority:low"], "high"),
        ):
            github = context("labeled", label="ci-preview")
            github.event.pull_request.labels.extend({"name": label} for label in labels)
            self.assertEqual(evaluate(priority, github), expected)

    def test_preview_requires_an_explicit_label_and_keeps_read_only_permissions(self) -> None:
        workflow = self.workflows["device-validation-preview.yml"]
        self.assertTrue(all(value == "read" for value in workflow["permissions"].values()))
        self.assertIs(workflow["concurrency"]["cancel-in-progress"], False)
        triggers = workflow.get("on", workflow.get(True))
        self.assertEqual(triggers["pull_request_target"]["types"], ["labeled"])
        job = workflow["jobs"]["preview"]
        self.assertEqual(job["uses"], "./.github/workflows/device-validation.yml")
        for ready in (False, True):
            for draft in (False, True):
                for label in ("ready", "ci-preview", "ci-priority:high"):
                    github = context("labeled", ready=ready, draft=draft, label=label)
                    github.event.pull_request.labels.append({"name": "ci-preview"})
                    self.assertEqual(
                        bool(evaluate(job["if"], github)),
                        ready and not draft and label == "ci-preview",
                    )

    def test_snapshot_preflight_can_read_pr_metadata_through_every_entry_point(self) -> None:
        # A reusable workflow cannot acquire permissions omitted by its caller.
        for filename, job_name in (
            ("device-validation.yml", "validate"),
            ("device-validation-preview.yml", "preview"),
        ):
            workflow = self.workflows[filename]
            permissions = workflow["jobs"][job_name].get("permissions", workflow["permissions"])
            for scope in ("actions", "contents", "pull-requests"):
                self.assertEqual(permissions.get(scope), "read", f"{filename}: {scope}")

    def test_fork_entry_points_are_rejected_before_self_hosted_allocation(self) -> None:
        for filename, job_name, label in (
            ("device-validation-preview.yml", "preview", "ci-preview"),
            ("beam-performance-ab.yml", "beam-performance-ab-only", "abtest"),
        ):
            workflow = self.workflows[filename]
            self.assertEqual(set(workflow.get("on", workflow.get(True))), {"pull_request_target"})
            github = context("labeled", label=label, event="pull_request_target")
            github.event.pull_request.labels.append({"name": label})
            job = workflow["jobs"][job_name]
            self.assertTrue(evaluate(job["if"], github))
            github.event.pull_request.head.repo.id = 2
            self.assertFalse(evaluate(job["if"], github))
            github.sha = "base-sha"
            if job_name == "preview":
                merge = job["with"]["merge_sha"]
            else:
                merge = next(
                    step["with"]["ref"]
                    for step in job["steps"]
                    if step.get("uses") == "actions/checkout@v4"
                )
            self.assertEqual(render(merge, github), "merge-sha")
        device = self.workflows["device-validation.yml"]["jobs"]
        self.assertEqual(device["validate"]["needs"], "preflight")
        self.assertEqual(device["preflight"]["runs-on"], "ubuntu-latest")
        self.assertEqual(device["preflight"]["steps"], [device["validate"]["steps"][0]])

    def test_cpu_refresh_names_the_exact_snapshot_and_checks_it_before_checkout(self) -> None:
        workflow = self.workflows["pre-commit.yml"]
        github = context(event="workflow_dispatch")
        github.sha = "trusted-dispatcher-sha"
        inputs = {
            "pr_number": "481",
            "head_sha": "head",
            "base_ref": "decode_graph",
            "base_sha": "base",
            "merge_sha": "merge",
        }
        self.assertEqual(
            render(workflow["run-name"], github, inputs=inputs),
            "ci-meta:v1;run=123;pr=481;base=decode_graph;base_sha=base;merge_sha=merge",
        )
        steps = workflow["jobs"]["fast-checks"]["steps"]
        self.assertEqual(steps[0]["name"], "Verify queued snapshot")
        self.assertTrue(evaluate(steps[0]["if"], github, inputs=inputs))
        self.assertFalse(evaluate(steps[0]["if"], github, inputs={"pr_number": "0"}))
        self.assertEqual(steps[1]["uses"], "actions/checkout@v4")
        self.assertEqual(render(steps[1]["with"]["ref"], github, inputs=inputs), "merge")
        previous = render(workflow["concurrency"]["group"], github, inputs=inputs)
        github.run_id += 1
        self.assertNotEqual(
            previous, render(workflow["concurrency"]["group"], github, inputs=inputs)
        )

    def test_queue_retains_pending_ticks_and_can_read_the_comment_inbox(self) -> None:
        workflow = self.workflows["ci-queue.yml"]
        self.assertEqual(workflow["permissions"]["issues"], "read")
        self.assertEqual(workflow["concurrency"]["queue"], "max")
        self.assertIs(workflow["concurrency"]["cancel-in-progress"], False)

    def test_preview_without_data_access_fails_before_device_setup(self) -> None:
        steps = self.workflows["device-validation.yml"]["jobs"]["validate"]["steps"]
        guard = next(
            step
            for step in steps
            if step.get("name") == "Verify dataset access before preparing the device"
        )
        setup = next(
            step for step in steps if step.get("uses") == "./.github/actions/setup-device-env"
        )
        self.assertLess(steps.index(guard), steps.index(setup))
        result = subprocess.run(
            ["bash", "-eu", "-c", guard["run"]],
            env={**os.environ, "HF_TOKEN": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("prepared OpenOneRec cache", result.stderr)

    def test_pangu_model_cache_preserves_the_pre_queue_worker_cache(self) -> None:
        steps = self.workflows["device-validation.yml"]["jobs"]["validate"]["steps"]
        cache = next(step for step in steps if step.get("id") == "cache-pangu1b")
        # These identify the already-warmed, read-only model on the same machine.
        self.assertEqual(cache["with"]["base"], "/data/actions/cache")
        self.assertEqual(cache["with"]["path"], "openPangu-Embedded-1B-V1.1")
        self.assertEqual(cache["with"]["key"], "pangu1b-model-${{ runner.os }}")

    def test_push_is_cpu_only_and_lint_does_not_launch_pytest(self) -> None:
        workflow = self.workflows["pre-commit.yml"]
        self.assertEqual(set(workflow["jobs"]), {"fast-checks"})
        hooks = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
        self.assertFalse(
            any(hook["id"] == "run-tests" for repo in hooks["repos"] for hook in repo["hooks"])
        )

    def test_queued_snapshot_guard_executes_against_changed_inputs(self) -> None:
        # Execute the actual workflow script against a small API double, not a copy of its logic.
        harness = """
const context = {repo: {owner: 'test', repo: 'repo'}};
const scenario = process.env.SCENARIO;
const pr = {state: 'open', draft: scenario === 'draft', labels: [{name: 'ready'}],
  head: {sha: process.env.HEAD_SHA, repo: {id: 1}}, base: {sha: process.env.BASE_SHA, ref: process.env.BASE_REF, repo: {id: 1}},
  merge_commit_sha: process.env.MERGE_SHA};
if (scenario === 'head') pr.head.sha = 'a'.repeat(40);
if (scenario === 'unready') pr.labels = [];
if (scenario === 'fork') pr.head.repo.id = 2;
if (scenario === 'deleted-repo') pr.head.repo = null;
if (scenario === 'wrong-base-repo') pr.base.repo.id = 2;
const github = {rest: {repos: {
  get: async () => ({data: {id: 1}}),
  getBranch: async () => ({data: {commit: {sha: scenario === 'base' ? 'a'.repeat(40) : process.env.BASE_SHA}}}),
  getCommit: async () => ({data: {parents: scenario === 'parents' ? [] :
    [{sha: process.env.BASE_SHA}, {sha: process.env.HEAD_SHA}]}})
}, pulls: {get: async () => ({data: pr})}}};
(async () => { SCRIPT })().catch((error) => { console.error(error.message); process.exitCode = 1; });
"""
        for filename, job in (
            ("device-validation.yml", "preflight"),
            ("pre-commit.yml", "fast-checks"),
            ("beam-performance-ab.yml", "beam-performance-ab-only"),
        ):
            guard = self.workflows[filename]["jobs"][job]["steps"][0]
            script = guard["with"]["script"]
            for scenario in (
                "valid",
                "head",
                "base",
                "draft",
                "unready",
                "parents",
                "fork",
                "deleted-repo",
                "wrong-base-repo",
            ):
                result = subprocess.run(
                    ["node", "-e", harness.replace("SCRIPT", script)],
                    text=True,
                    capture_output=True,
                    check=False,
                    env={
                        **os.environ,
                        "PR_NUMBER": "1",
                        "HEAD_SHA": "1" * 40,
                        "BASE_SHA": "2" * 40,
                        "MERGE_SHA": "3" * 40,
                        "BASE_REF": "decode_graph",
                        "REQUIRE_READY": guard["env"].get("REQUIRE_READY", "true"),
                        "SCENARIO": scenario,
                    },
                )
                allowed = scenario == "valid" or (
                    scenario == "unready" and filename == "beam-performance-ab.yml"
                )
                self.assertEqual(
                    result.returncode, int(not allowed), f"{filename}: {result.stderr}"
                )


if __name__ == "__main__":
    unittest.main()
