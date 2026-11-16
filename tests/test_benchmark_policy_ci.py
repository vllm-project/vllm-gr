# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the policy job against real checkouts without an accelerator or GitHub."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import regex as re
import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path(".github/benchmark-gate.yaml")
RESOLVER = Path("tools/ci/resolve_benchmark_policy.py")
WORKFLOW = ROOT / ".github/workflows/pre-commit.yml"
pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def commit(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=CI Test",
        "-c",
        "user.email=ci@example.invalid",
        "commit",
        "-qm",
        message,
    )
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture  # type: ignore[misc]
def checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "checkout"
    repo.mkdir()
    git(repo, "init", "-qb", "base")
    for name in (POLICY, RESOLVER):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, repo / name)
    commit(repo, "Initial policy and resolver")
    return repo


def run_policy(
    repo: Path, output_dir: Path, *, device: str = "gpu", board: str = "unknown"
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["benchmark-policy"]
    step = next(item for item in job["steps"] if item.get("id") == "benchmark-policy")
    output = output_dir / "output"
    summary = output_dir / "summary"
    # Stub only hardware detection. Execute the workflow's actual Python CLI.
    script = """
command() { [[ "$*" == "-v npu-smi" && "$TEST_DEVICE" == "npu" ]]; }
python() {
  if [[ "$*" == *"import acl"* ]]; then
    printf '%s\n' "$TEST_BOARD"
  else
    "$TEST_PYTHON" "$@"
  fi
}
""" + step["run"]
    result = subprocess.run(
        ["bash", "-eu", "-o", "pipefail", "-c", script],
        cwd=repo,
        env={
            **os.environ,
            "TEST_DEVICE": device,
            "TEST_BOARD": board,
            "TEST_PYTHON": sys.executable,
            "POLICY_REF": git(repo, "rev-parse", "HEAD"),
            "POLICY_TARGET_BRANCH": "main",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output, summary


def read_output(output: Path) -> dict[str, str]:
    key, value = output.read_text().strip().split("=", 1)
    assert key == "policy-json"
    values: dict[str, str] = json.loads(value)
    return values


@pytest.mark.parametrize(  # type: ignore[misc]
    "device,board,pangu,ttft",
    [
        ("gpu", "unknown", "1.0", "300"),
        ("npu", "Ascend910_9362", "0.875", "280"),
    ],
)
def test_policy_job_emits_every_consumer_field(
    checkout: Path, tmp_path: Path, device: str, board: str, pangu: str, ttft: str
) -> None:
    result, output, summary = run_policy(checkout, tmp_path, device=device, board=board)
    assert result.returncode == 0, result.stdout + result.stderr
    values = read_output(output)
    consumed = re.findall(
        r"fromJSON\(needs\.benchmark-policy\.outputs\.policy-json\)\.(\w+)",
        WORKFLOW.read_text(),
    )
    assert consumed and set(consumed) <= values.keys()
    assert values["PASS_AT_1_THRESHOLD"] == "0.05"
    assert values["PASS_AT_32_THRESHOLD"] == "0.21"
    assert values["RECALL_AT_32_THRESHOLD"] == "0.03"
    assert values["P99_TTFT_THRESHOLD"] == ttft
    assert values["PANGU1B_OFFLINE_ACCURACY_THRESHOLD"] == pangu
    assert values["PANGU1B_ONLINE_ACCURACY_THRESHOLD"] == pangu
    assert values["ACCURACY_FAILURE_MODE"] == "blocking"
    assert values["PANGU1B_OFFLINE_ACCURACY_FAILURE_MODE"] == "blocking"
    assert values["ONLINE_NON_ACCURACY_FAILURE_MODE"] == "blocking"
    assert values["P99_TTFT_FAILURE_MODE"] == "warning"
    assert git(checkout, "rev-parse", "HEAD") in summary.read_text()
    digest = hashlib.sha256((checkout / POLICY).read_bytes()).hexdigest()
    assert digest in summary.read_text()


def test_policy_uses_the_pinned_merge_even_after_base_advances(
    checkout: Path, tmp_path: Path
) -> None:
    policy = checkout / POLICY
    original = policy.read_text()
    policy.write_text(original.replace("offline_accuracy_min: 1.00", "offline_accuracy_min: 0.90"))
    ancestor = commit(checkout, "Old accuracy floor")
    policy.write_text(original)
    commit(checkout, "Base raises accuracy floor")
    git(checkout, "checkout", "-qb", "candidate", ancestor)
    policy.write_text(policy.read_text().replace("name: development", "name: candidate-policy"))
    commit(checkout, "PR changes policy name")
    git(checkout, "checkout", "-q", "base")
    git(
        checkout,
        "-c",
        "user.name=CI Test",
        "-c",
        "user.email=ci@example.invalid",
        "merge",
        "--no-ff",
        "-qm",
        "Test merge",
        "candidate",
    )
    merge = git(checkout, "rev-parse", "HEAD")
    # A queued run still tests this merge after the base advances again.
    policy.write_text(
        policy.read_text().replace("offline_accuracy_min: 1.00", "offline_accuracy_min: 0.80")
    )
    commit(checkout, "Later base update")
    git(checkout, "checkout", "-q", "--detach", merge)

    result, output, summary = run_policy(checkout, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    values = read_output(output)
    assert values["BENCHMARK_POLICY_NAME"] == "candidate-policy"
    assert values["PANGU1B_OFFLINE_ACCURACY_THRESHOLD"] == "1.0"
    assert merge in summary.read_text()
    # The real gate must reject 95%, which incorrectly passed with the head policy.
    gate = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; validate_accuracy_metric Pangu 0.95 "$2" blocking gpu 90',
            "gate",
            str(ROOT / "tools/ci/benchmark_gate_helpers.sh"),
            values["PANGU1B_OFFLINE_ACCURACY_THRESHOLD"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert gate.returncode == 90, gate.stdout + gate.stderr


@pytest.mark.parametrize("invalid", ["missing", "yaml", "schema"])  # type: ignore[misc]
def test_invalid_policy_cannot_publish_outputs(
    checkout: Path, tmp_path: Path, invalid: str
) -> None:
    policy = checkout / POLICY
    if invalid == "missing":
        policy.unlink()
    elif invalid == "yaml":
        policy.write_text("thresholds: [\n")
    else:
        policy.write_text(policy.read_text().replace("pass_at_1_min: 0.09", "pass_at_1_min: wrong"))
    result, output, _ = run_policy(checkout, tmp_path)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("version", [1, 2])  # type: ignore[misc]
def test_shell_and_job_outputs_agree_for_supported_schemas(
    checkout: Path, tmp_path: Path, version: int
) -> None:
    policy_path = checkout / POLICY
    if version == 1:
        policy = yaml.safe_load(policy_path.read_text())
        policy["version"] = 1
        policy["failure_modes"]["performance"] = "warning"
        del policy["pangu1b"]
        policy_path.write_text(yaml.safe_dump(policy))
    env_file, output, summary = (tmp_path / name for name in ("env", "output", "summary"))
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(checkout / RESOLVER),
            "--policy-file",
            str(policy_path),
            "--device",
            "gpu",
            "--board-id",
            "decode_graph_gpu",
            "--source-ref",
            git(checkout, "rev-parse", "HEAD"),
            "--github-env",
            str(env_file),
            "--github-output",
            str(output),
            "--github-step-summary",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = read_output(output)
    assert values == dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert values["PANGU1B_OFFLINE_ACCURACY_THRESHOLD"] == "1.0"
    assert values["PANGU1B_OFFLINE_ACCURACY_FAILURE_MODE"] == "blocking"
    assert values["P99_TTFT_FAILURE_MODE"] == "warning"
    assert values["PANGU1B_POLICY_SOURCE"] == ("strict-bootstrap" if version == 1 else "policy")


def test_all_jobs_checkout_the_same_workflow_commit() -> None:
    workflow: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text())
    for name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["ref"] == "${{ github.sha }}", name
    policy_job = workflow["jobs"]["benchmark-policy"]
    assert policy_job["steps"][0]["uses"].startswith("actions/checkout@")
    assert all("github-script" not in step.get("uses", "") for step in policy_job["steps"])
