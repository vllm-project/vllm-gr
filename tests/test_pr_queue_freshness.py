# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the exact github-script freshness guards in the checked-in workflow."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.cpu_test
ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/pre-commit.yml"
RUNNER = ROOT / "tests/pr_queue_script_runner.cjs"
HEAD, BASE, MERGE = "a" * 40, "b" * 40, "c" * 40


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text())


def _script(job: str, name: str) -> str:
    steps = _workflow()["jobs"][job]["steps"]
    found = [s["with"]["script"] for s in steps if s.get("name") == name]
    assert len(found) == 1
    return found[0]


def _pr() -> dict[str, Any]:
    return {
        "number": 42,
        "state": "open",
        "draft": False,
        "head": {"sha": HEAD},
        "base": {"sha": BASE, "ref": "main"},
        "merge_commit_sha": MERGE,
        "labels": [{"name": "ready"}],
    }


def _run(
    script: str,
    pr: dict,
    live: Any,
    *,
    event: str = "pull_request",
    action: str = "synchronize",
    label: str | None = None,
    sha: str = MERGE,
    api_error: bool = False,
    files: list | None = None,
    live2: Any = None,
    file_api_error: bool = False,
) -> dict:
    if not shutil.which("node"):
        pytest.skip("node is unavailable")
    payload = {"pull_request": pr, "action": action}
    if label is not None:
        payload["label"] = {"name": label}
    params = {
        "script": script,
        "context": {
            "eventName": event,
            "sha": sha,
            "repo": {"owner": "JiusiServe", "repo": "vllm-gr"},
            "payload": payload,
        },
        "live": live,
        "api_error": api_error,
        "files": files or [],
        "file_api_error": file_api_error,
    }
    if live2 is not None:
        params["live2"] = live2
    p = subprocess.run(
        ["node", str(RUNNER)], input=json.dumps(params), capture_output=True, text=True, check=True
    )
    return json.loads(p.stdout)


@pytest.fixture(
    params=[
        ("pr-admission", "Check queued PR freshness"),
        ("benchmark-blocking", "Recheck queued PR freshness at completion"),
    ]
)
def script(request: pytest.FixtureRequest) -> str:
    return _script(*request.param)


def test_accepts_current_ready_and_review_pr(script: str) -> None:
    pr = _pr()
    for event in ("pull_request", "pull_request_review"):
        result = _run(script, pr, _pr(), event=event)
        assert result["ok"] and result["calls"] == [
            {"owner": "JiusiServe", "repo": "vllm-gr", "pull_number": 42}
        ]


def test_accepts_current_abtest_only_pr(script: str) -> None:
    pr = _pr()
    pr["labels"] = [{"name": "abtest"}]
    assert _run(script, pr, pr, action="labeled", label="abtest")["ok"]


@pytest.mark.parametrize(
    ("event", "action", "label"),
    [
        ("pull_request", "synchronize", None),
        ("pull_request", "labeled", "ready"),
        ("pull_request_review", "submitted", None),
    ],
)
@pytest.mark.parametrize("had_abtest", [False, True])
def test_full_ready_run_ignores_abtest_label_changes(
    script: str, event: str, action: str, label: str | None, had_abtest: bool
) -> None:
    pr = _pr()
    if had_abtest:
        pr["labels"].append({"name": "abtest"})
    live = {**pr, "labels": [{"name": "ready"}]}
    if not had_abtest:
        live["labels"].append({"name": "abtest"})
    assert _run(script, pr, live, event=event, action=action, label=label)["ok"]


@pytest.mark.parametrize("ready", [False, True])
def test_abtest_request_rejects_removed_abtest_label(script: str, ready: bool) -> None:
    pr = _pr()
    pr["labels"] = ([{"name": "ready"}] if ready else []) + [{"name": "abtest"}]
    live = {**pr, "labels": [{"name": "ready"}] if ready else []}
    assert _run(script, pr, pr, action="labeled", label="abtest")["ok"]
    assert not _run(script, pr, live, action="labeled", label="abtest")["ok"]


def test_full_ready_rejects_ready_removal_even_when_abtest_changes(script: str) -> None:
    pr = _pr()
    live = {**pr, "labels": [{"name": "abtest"}]}
    assert not _run(script, pr, live)["ok"]


@pytest.mark.parametrize("ready", [False, True])
def test_abtest_request_rejects_ready_label_changes(script: str, ready: bool) -> None:
    pr = _pr()
    pr["labels"] = ([{"name": "ready"}] if ready else []) + [{"name": "abtest"}]
    live = {**pr, "labels": ([] if ready else [{"name": "ready"}]) + [{"name": "abtest"}]}
    assert not _run(script, pr, live, action="labeled", label="abtest")["ok"]


def test_full_ready_run_ignores_unrelated_label_changes(script: str) -> None:
    pr = _pr()
    live = {**pr, "labels": [{"name": "ready"}, {"name": "documentation"}]}
    assert _run(script, pr, live)["ok"]


@pytest.mark.parametrize(
    "change",
    [
        {"state": "closed"},
        {"draft": True},
        {"draft": None},
        {"head": {"sha": "d" * 40}},
        {"base": {"sha": "e" * 40, "ref": "main"}},
        {"base": {"sha": BASE, "ref": "decode_graph"}},
        {"merge_commit_sha": "f" * 40},
        {"merge_commit_sha": None},
        {"labels": []},
        {"labels": None},
        {"head": None},
        {"base": None},
    ],
)
def test_rejects_stale_or_invalid_live_snapshot(script: str, change: dict) -> None:
    live = {**_pr(), **change}
    assert not _run(script, _pr(), live)["ok"]


def test_rejects_missing_live_draft_and_api_errors(script: str) -> None:
    live = _pr()
    live.pop("draft")
    assert not _run(script, _pr(), live)["ok"]
    assert not _run(script, _pr(), None, api_error=True)["ok"]


def test_rejects_invalid_event_revision(script: str) -> None:
    assert not _run(script, _pr(), _pr(), sha="not-a-commit")["ok"]


def test_dispatch_and_ignored_label_do_not_fetch_pr(script: str) -> None:
    for args in ({"event": "workflow_dispatch"}, {"action": "labeled", "label": "bug"}):
        result = _run(script, _pr(), _pr(), **args)
        assert result["ok"] and result["calls"] == []


def test_admission_is_ancestor_of_all_heavy_jobs_and_final_gate_checks_it() -> None:
    jobs = _workflow()["jobs"]

    def deps(name: str) -> set[str]:
        value = jobs[name].get("needs", [])
        names = [value] if isinstance(value, str) else value
        return set(names) | set().union(*(deps(n) for n in names))

    for name in (
        "build",
        "benchmark-policy",
        "pre-commit",
        "decode-benchmark",
        "pangu-offline",
        "pangu-online",
        "beam-performance-ab-only",
    ):
        assert "pr-admission" in deps(name), name
    gate = jobs["benchmark-blocking"]
    assert "pr-admission" in gate["needs"]
    assert any(step.get("name") == "Require queued PR admission" for step in gate["steps"])


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ([{"filename": "docs/usage.md", "status": "modified"}], "false"),
        ([{"filename": "examples/demo.py", "status": "modified"}], "false"),
        ([{"filename": ".github/workflows/x.yml", "status": "modified"}], "false"),
        ([{"filename": ".buildkite/pipeline", "status": "modified"}], "false"),
        ([{"filename": "abc.yml", "status": "modified"}], "false"),
        ([{"filename": "nested/abc.yml", "status": "modified"}], "true"),
        ([{"filename": "pkg/guide.md", "status": "modified"}], "false"),
        ([{"filename": "vllm_gr/model.py", "status": "modified"}], "true"),
        (
            [
                {
                    "filename": "docs/new.md",
                    "previous_filename": "vllm_gr/old.py",
                    "status": "renamed",
                }
            ],
            "true",
        ),
        (
            [
                {
                    "filename": "vllm_gr/new.py",
                    "previous_filename": "docs/old.md",
                    "status": "renamed",
                }
            ],
            "true",
        ),
    ],
)
def test_wheel_scope_matches_existing_paths_ignore(files: list, expected: str) -> None:
    live = _pr()
    live["changed_files"] = len(files)
    result = _run(_script("wheel-scope", "Check wheel path filters"), _pr(), live, files=files)
    assert result["ok"] and result["outputs"]["required"] == expected


def test_wheel_scope_fails_safe_on_errors_or_changed_pr() -> None:
    script = _script("wheel-scope", "Check wheel path filters")
    files = [{"filename": "docs/guide.md", "status": "modified"}]
    live = {**_pr(), "changed_files": 1}
    for params in (
        {"api_error": True},
        {"file_api_error": True},
        {"live2": {**live, "head": {"sha": "d" * 40}}},
    ):
        result = _run(script, _pr(), live, files=files, **params)
        assert result["ok"] and result["outputs"]["required"] == "true"
