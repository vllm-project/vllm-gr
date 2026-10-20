# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the workflow guards against PR lifecycle events without a runner."""

import copy
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import regex as re
import yaml

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
                "label": {"name": label},
                "pull_request": {
                    "number": 481,
                    "draft": draft,
                    "labels": [{"name": "ready"}] if ready else [],
                    "base": {"ref": "decode_graph", "sha": "base-sha"},
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
    return eval(  # noqa: S307 - only repository-owned workflow expressions
        " ".join(expression.split()),
        {"__builtins__": {}},
        {
            "github": github,
            "labels": [item["name"] for item in github.event.pull_request.labels],
            "results": results or {},
            "always": lambda: True,
            "cancelled": lambda: cancelled,
            "contains": lambda items, value: value in items,
            "format": lambda template, *args: template.format(*args),
        },
    )


def render(template: str, github: Any) -> str:
    rendered = re.sub(
        r"\$\{\{(.*?)\}\}",
        lambda match: str(evaluate(match.group(1), github)),
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
            if path.name in {"pre-commit.yml", "build_wheel.yml"}
        }

    def scheduled(
        self,
        workflow: dict[str, Any],
        github: Any,
        failed: str | None = None,
        *,
        cancelled: bool = False,
    ) -> list[str]:
        results: dict[str, str] = {}
        running = []
        for name, job in workflow["jobs"].items():
            needs = job.get("needs", [])
            if isinstance(needs, str):
                needs = [needs]
            condition = job.get("if", "true")
            dependencies_ok = all(results[need] == "success" for need in needs)
            explicit_status = any(status in condition for status in ("always()", "cancelled()"))
            enabled = (explicit_status or (dependencies_ok and not cancelled)) and evaluate(
                condition, github, results, cancelled=cancelled
            )
            results[name] = ("failure" if name == failed else "success") if enabled else "skipped"
            if enabled:
                running.append(name)
        return running

    def test_pr_lifecycle(self) -> None:
        events = [
            ("opened", False, False, "", False),
            ("synchronize", False, False, "", False),
            ("reopened", False, False, "", False),
            ("ready_for_review", False, False, "", False),
            ("labeled", True, False, "ready", True),
            ("synchronize", True, False, "", True),
            ("reopened", True, False, "", True),
            ("ready_for_review", True, False, "", True),
            ("labeled", True, True, "ready", False),
            ("synchronize", True, True, "", False),
            ("converted_to_draft", True, True, "", False),
            ("unlabeled", False, False, "ready", False),
            ("labeled", True, False, "bug", False),
            ("unlabeled", True, False, "bug", False),
            ("labeled", False, False, "bug", False),
        ]
        for filename, workflow in self.workflows.items():
            triggers = workflow.get("on", workflow.get(True))["pull_request"]["types"]
            for action, ready, draft, label, expected in events:
                with self.subTest(
                    workflow=filename, action=action, ready=ready, draft=draft, label=label
                ):
                    self.assertIn(action, triggers)
                    github = context(action, ready=ready, draft=draft, label=label)
                    running = self.scheduled(workflow, github)
                    validation = [job for job in running if job != "benchmark-blocking"]
                    self.assertEqual(bool(validation), expected)
                    if expected:
                        self.assertEqual(set(running), set(workflow["jobs"]))
                    if "benchmark-blocking" in workflow["jobs"]:
                        ignored_label = action in {"labeled", "unlabeled"} and label != "ready"
                        self.assertEqual("benchmark-blocking" in running, not ignored_label)
                        gate_name = render(workflow["jobs"]["benchmark-blocking"]["name"], github)
                        self.assertEqual(gate_name == "Benchmark blocking gate", not ignored_label)
                    if "run-name" in workflow:
                        self.assertEqual(
                            render(workflow["run-name"], github).startswith("ci-meta:v1;"),
                            expected,
                        )

    def test_push_and_manual_dispatch_remain_enabled(self) -> None:
        for workflow in self.workflows.values():
            triggers = workflow.get("on", workflow.get(True))
            self.assertIn("workflow_dispatch", triggers)
            self.assertEqual(set(triggers["push"]["branches"]), {"main", "decode_graph"})
            for event in ("push", "workflow_dispatch"):
                github = context(event=event, ready=False, draft=True)
                self.assertEqual(set(self.scheduled(workflow, github)), set(workflow["jobs"]))

    def test_label_edits_do_not_cancel_unrelated_runs(self) -> None:
        for workflow in self.workflows.values():
            group = workflow["concurrency"]["group"]
            normal = render(group, context())
            for action, ready, draft, label in (
                ("labeled", True, False, "ready"),
                ("unlabeled", False, False, "ready"),
                ("converted_to_draft", True, True, ""),
            ):
                github = context(action, ready=ready, draft=draft, label=label)
                self.assertEqual(render(group, github), normal)
                self.assertTrue(evaluate(workflow["concurrency"]["cancel-in-progress"], github))
            for action in ("labeled", "unlabeled"):
                github = context(action, label="bug")
                self.assertNotEqual(render(group, github), normal)
                second = copy.deepcopy(github)
                second.run_id += 1
                self.assertNotEqual(render(group, second), render(group, github))

    def test_failed_validation_does_not_skip_blocking_gate(self) -> None:
        workflow = self.workflows["pre-commit.yml"]
        if "benchmark-blocking" not in workflow["jobs"]:
            self.skipTest("This branch uses a single validation job")
        for failed in workflow["jobs"]["benchmark-blocking"]["needs"]:
            with self.subTest(failed=failed):
                self.assertIn("benchmark-blocking", self.scheduled(workflow, context(), failed))

    def test_blocking_gate_still_rejects_unsuccessful_checks(self) -> None:
        job = self.workflows["pre-commit.yml"]["jobs"]["benchmark-blocking"]
        success = dict.fromkeys(job["needs"], "success")
        scenarios = [(success, 0)]
        for variable in success:
            for status in ("failure", "cancelled", "skipped"):
                scenarios.append(({**success, variable: status}, 1))
        for values, expected_code in scenarios:
            with self.subTest(values=values):
                result = self.run_gate(values)
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)

    def run_gate(self, results: dict[str, str]) -> subprocess.CompletedProcess[str]:
        step = self.workflows["pre-commit.yml"]["jobs"]["benchmark-blocking"]["steps"][0]
        env = {
            key: str(evaluate(value, context(), results)).lower()
            for key, value in step["env"].items()
        }
        return subprocess.run(
            ["bash", "-eu", "-o", "pipefail", "-c", step["run"]],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )

    def test_unready_and_draft_events_fail_required_gate(self) -> None:
        workflow = self.workflows["pre-commit.yml"]
        gate = workflow["jobs"]["benchmark-blocking"]
        for github in (
            context("opened", ready=False),
            context("synchronize", ready=False),
            context("unlabeled", ready=False, label="ready"),
            context("converted_to_draft", draft=True),
            context("labeled", draft=True, label="ready"),
        ):
            with self.subTest(event=github.event):
                running = self.scheduled(workflow, github)
                self.assertEqual(running, ["benchmark-blocking"])
                self.assertEqual(render(gate["name"], github), "Benchmark blocking gate")
                results = {
                    need: "success" if need in running else "skipped" for need in gate["needs"]
                }
                result = self.run_gate(results)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_cancellation_stops_pangu_but_keeps_cleanup_and_gate(self) -> None:
        workflow = self.workflows["pre-commit.yml"]
        jobs = workflow["jobs"]
        # A cancellation can arrive while a Pangu job is already running, after
        # its prerequisites have succeeded. GitHub re-evaluates the job guard.
        results = dict.fromkeys(jobs["benchmark-blocking"]["needs"], "success")
        for name in ("pangu-offline", "pangu-online"):
            with self.subTest(job=name):
                self.assertTrue(evaluate(jobs[name]["if"], context(), results))
                self.assertFalse(evaluate(jobs[name]["if"], context(), results, cancelled=True))
        # Upstream validation failures must still allow independent accuracy
        # checks to run, and must not hide a failed final gate.
        for failed in ("decode-benchmark", "pangu-offline"):
            with self.subTest(failed=failed):
                running = self.scheduled(workflow, context(), failed)
                self.assertIn("pangu-offline", running)
                self.assertIn("pangu-online", running)
                self.assertIn("benchmark-blocking", running)
        self.assertEqual(
            self.scheduled(workflow, context(), cancelled=True), ["benchmark-blocking"]
        )
        reject_cancelled = jobs["benchmark-blocking"]["steps"][-1]
        self.assertFalse(evaluate(reject_cancelled["if"], context()))
        self.assertTrue(evaluate(reject_cancelled["if"], context(), cancelled=True))
        rejected = subprocess.run(
            ["bash", "-eu", "-o", "pipefail", "-c", reject_cancelled["run"]],
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 1)
        for name in ("decode-benchmark", "pangu-offline", "pangu-online"):
            cleanup = next(
                step for step in jobs[name]["steps"] if step.get("name") == "Check clear device"
            )
            # Evaluate the guard after successful device mapping, even during
            # cancellation. Failed mapping must still prevent unsafe cleanup.
            for outcome in ("success", "failure", "skipped"):
                condition = cleanup["if"].replace("steps.cuda-mapping.outcome", repr(outcome))
                self.assertEqual(
                    bool(evaluate(condition, context(), cancelled=True)), outcome == "success"
                )


if __name__ == "__main__":
    unittest.main()
