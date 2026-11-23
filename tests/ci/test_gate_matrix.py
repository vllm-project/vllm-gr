# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Runner-registration and final-summary failure contracts, without device access."""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.paths import REPO_ROOT
from tools.ci.gate_matrix import plan, verify

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture(autouse=True)
def reset_vocab_slice_config() -> None:
    pass


def registry() -> dict[str, Any]:
    return {
        "version": 2,
        "targets": {
            "cuda:H100": {
                "enabled": True,
                "allocations": {
                    "1": {
                        "routing_label": "test-only-device",
                        "labels": ["self-hosted", "test-only-device"],
                        "python": "/test-only-env/bin/python",
                        "isolated": True,
                        "policy_board": "default",
                    }
                },
            }
        },
    }


def test_unregistered_inventory_cannot_schedule_devices() -> None:
    config = json.loads((REPO_ROOT / ".github/gate-runners.json").read_text())
    assert len(config["targets"]) == 9
    for target in config["targets"]:
        with pytest.raises(ValueError, match="not registered"):
            plan(config, [target], ["unit"], "a" * 40)


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels", []),
        ("labels", ["self-hosted"]),
        ("routing_label", "linux"),
        ("labels", ["self-hosted", "test-only-device", "test-only-device"]),
        ("policy_board", "typo"),
        ("python", None),
        ("python", "python"),
        ("policy_board", None),
    ],
)
def test_enabled_runner_needs_concrete_configuration(field: str, value: Any) -> None:
    config = registry()
    config["targets"]["cuda:H100"]["allocations"]["1"][field] = value
    with pytest.raises(ValueError):
        plan(config, ["cuda:H100"], ["unit"], "a" * 40)


def test_resource_and_backend_eligibility() -> None:
    config = registry()
    config["targets"]["cuda:H100"]["allocations"]["1"]["isolated"] = False
    with pytest.raises(ValueError, match="isolated"):
        plan(config, ["cuda:H100"], ["resource"], "a" * 40)
    config["targets"]["npu:A3"] = config["targets"].pop("cuda:H100")
    with pytest.raises(ValueError, match="CUDA-only"):
        plan(config, ["npu:A3"], ["functional"], "a" * 40)
    with pytest.raises(ValueError, match="Unsupported"):
        plan(config, ["npu:A3"], ["performance-ab"], "a" * 40)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "failed",
        "cancelled",
        "skipped",
        "missing",
        "stale",
        "wrong-run",
        "wrong-policy",
        "wrong-device",
        "wrong-cards",
        "missing-junit",
        "test-skip",
        "missing-job",
    ],
)
def test_final_gate_requires_every_current_report(tmp_path: Path, failure: str | None) -> None:
    expected = plan(registry(), ["cuda:H100"], ["unit"], "a" * 40)
    root = tmp_path / "gate-result-cuda-H100--cards-1--unit"
    (root / "unit").mkdir(parents=True)
    summary = {
        "status": "passed",
        "commit": "a" * 40,
        "hardware": "cuda:H100",
        "workflow_run": expected["workflow_run"],
        "cards": 1,
        "policy_board": "default",
        "required_suites": ["unit"],
        "results": {"unit": {"status": "passed", "exit_code": 0}},
    }
    if failure == "stale":
        summary["commit"] = "b" * 40
    if failure == "wrong-run":
        summary["workflow_run"] = "another-workflow-run"
    if failure == "wrong-policy":
        summary["policy_board"] = "typo"
    if failure == "wrong-device":
        summary["hardware"] = "cuda:L4"
    if failure == "wrong-cards":
        summary["cards"] = 4
    if failure != "missing":
        (root / "summary.json").write_text(json.dumps(summary))
    if failure != "missing-junit":
        outcome_xml = "<skipped/>" if failure == "test-skip" else ""
        (root / "unit/junit.xml").write_text(
            f"<testsuite><testcase>{outcome_xml}</testcase></testsuite>"
        )
    if failure == "missing-job":
        expected = plan(registry(), ["cuda:H100"], ["unit", "resource"], "a" * 40)
    outcome = failure if failure in {"failed", "cancelled", "skipped"} else "success"
    assert bool(verify(expected, tmp_path, str(outcome))) is (failure is not None)


def test_matrix_separates_card_allocations() -> None:
    config = registry()
    lanes = config["targets"]["cuda:H100"]["allocations"]
    lanes["4"] = {**lanes["1"], "labels": ["self-hosted", "four"], "routing_label": "four"}
    jobs = plan(config, ["cuda:H100"], ["unit", "resource"], "a" * 40)["include"]
    assert [(j["cards"], j["suite"]) for j in jobs] == [(1, "unit"), (1, "resource"), (4, "unit")]
    assert len({j["id"] for j in jobs}) == 3


@pytest.mark.parametrize("rerun_all", [False, True])
def test_reruns_replace_only_retried_artifacts(tmp_path: Path, rerun_all: bool) -> None:
    expected = plan(registry(), ["cuda:H100"], ["unit", "resource"], "a" * 40)

    def publish(job: dict[str, Any], success: bool, attempt: int) -> None:
        suite = job["suite"]
        root = tmp_path / ("gate-result-" + job["id"])
        (root / suite).mkdir(parents=True, exist_ok=True)
        status = "passed" if success else "failed"
        summary = {
            "commit": expected["commit"],
            "workflow_run": expected["workflow_run"],
            "hardware": job["hardware"],
            "cards": job["cards"],
            "policy_board": job["policy_board"],
            "required_suites": [suite],
            "status": status,
            "attempt": attempt,
            "results": {suite: {"status": status, "exit_code": 0 if success else 1}},
        }
        (root / "summary.json").write_text(json.dumps(summary))
        (root / suite / "junit.xml").write_text("<testsuite><testcase/></testsuite>")

    unit, resource = expected["include"]
    publish(unit, True, 1)
    publish(resource, False, 1)
    assert verify(expected, tmp_path, "failure")
    if rerun_all:
        publish(unit, True, 2)
    publish(resource, True, 2)
    assert verify(expected, tmp_path, "success") == []
