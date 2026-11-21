# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Plan only registered runners; require complete, current reports from every job."""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def plan(
    registry: dict[str, Any], targets: list[str], suites: list[str], commit: str
) -> dict[str, Any]:
    from tools.ci.resolve_benchmark_policy import load_policy

    policy = load_policy(Path(__file__).resolve().parents[2] / ".github/benchmark-gate.yaml")
    supported = {
        "unit",
        "resource",
        "functional",
        "recall",
        "benchmark",
        "performance",
    }
    if registry.get("version") != 2 or not targets or not suites:
        raise ValueError("A versioned runner registry, targets and suites are required")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("A full checkout commit SHA is required")
    jobs = []
    for target in dict.fromkeys(targets):
        runner = registry.get("targets", {}).get(target)
        if not runner or runner.get("enabled") is not True:
            raise ValueError(f"Runner is not registered/enabled: {target}")
        allocations = runner.get("allocations")
        if not isinstance(allocations, dict) or not allocations:
            raise ValueError(f"Missing card allocations: {target}")
        for count, runner in allocations.items():
            if count not in {str(n) for n in range(1, 9)}:
                raise ValueError(f"Invalid allocation size: {count}")
            cards = int(count)
            routing = runner.get("routing_label")
            labels = runner.get("labels")
            python = runner.get("python")
            board = runner.get("policy_board")
            if (
                not isinstance(labels, list)
                or "self-hosted" not in labels
                or not isinstance(routing, str)
                or routing not in labels
                or routing.casefold()
                in {"self-hosted", "linux", "windows", "macos", "x64", "arm", "arm64"}
                or not all(isinstance(label, str) and label.strip() for label in labels)
                or len({label.casefold() for label in labels}) != len(labels)
                or not isinstance(python, str)
                or not python.startswith("/")
                or not isinstance(board, str)
                or not board
            ):
                raise ValueError(
                    f"Real runner labels and an absolute Python path are required: {target}"
                )
            if target not in {
                f"cuda:{sku}" for sku in ("L20", "A100", "H800", "H100", "L4", "B200")
            } | {"npu:A2", "npu:A3", "npu:A5"}:
                raise ValueError(f"Unknown hardware target: {target}")
            for suite in dict.fromkeys(suites):
                if suite not in supported:
                    raise ValueError(f"Unsupported dispatch suite: {suite}")
                if target.startswith("npu:") and suite in {"functional", "recall", "performance"}:
                    raise ValueError(f"CUDA-only suite {suite} cannot run on {target}")
                if board != "default" and board not in policy["device_overrides"].get(
                    "npu" if target.startswith("npu:") else "gpu", {}
                ).get("boards", {}):
                    raise ValueError(f"Unknown policy board {board} for {target}")
                if cards > 1 and suite != "unit" and not (cards == 2 and suite == "resource"):
                    continue
                if suite == "resource" and runner.get("isolated") is not True:
                    raise ValueError("Resource probes require an isolated runner")
                jobs.append(
                    {
                        "id": target.replace(":", "-") + f"--cards-{cards}--{suite}",
                        "hardware": target,
                        "suite": suite,
                        "cards": cards,
                        "labels": labels,
                        "python": python,
                        "policy_board": board,
                        "isolated": "true" if runner.get("isolated") is True else "false",
                    }
                )
        for suite in suites:
            if not any(job["hardware"] == target and job["suite"] == suite for job in jobs):
                raise ValueError(f"No compatible card allocation for {target} {suite}")
    return {
        "schema": 1,
        "commit": commit,
        "include": jobs,
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
    }


def verify(plan_data: dict[str, Any], reports: Path, job_result: str) -> list[str]:
    errors = []
    if job_result != "success":
        errors.append(f"Matrix result is {job_result}, not success")
    jobs = plan_data.get("include", [])
    if plan_data.get("schema") != 1 or not jobs:
        return [*errors, "Missing/invalid expected job plan"]
    for job in jobs:
        root = reports / ("gate-result-" + job["id"])
        try:
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            suite = job["suite"]
            if (
                summary["status"] != "passed"
                or summary["commit"] != plan_data["commit"]
                or summary["hardware"] != job["hardware"]
                or summary["required_suites"] != [suite]
                or summary["results"][suite]["status"] != "passed"
                or summary["cards"] != job["cards"]
                or summary["policy_board"] != job["policy_board"]
                or summary["workflow_run"] != plan_data["workflow_run"]
                or summary["results"][suite]["exit_code"] != 0
            ):
                raise ValueError("incomplete, stale or mismatched result")
            cases = ET.parse(root / suite / "junit.xml").findall(".//testcase")
            if not cases or any(
                case.find(tag) is not None
                for case in cases
                for tag in ("skipped", "failure", "error")
            ):
                raise ValueError("JUnit has missing or unsuccessful cases")
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
            errors.append(f"{job['id']}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan")
    prepare.add_argument("--registry", type=Path, default=Path(".github/gate-runners.json"))
    prepare.add_argument("--targets", required=True)
    prepare.add_argument("--suites", required=True)
    prepare.add_argument("--commit", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    finish = commands.add_parser("verify")
    finish.add_argument("--plan", type=Path, required=True)
    finish.add_argument("--reports", type=Path, required=True)
    finish.add_argument("--job-result", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(
            json.loads(args.registry.read_text()),
            args.targets.split(),
            args.suites.split(),
            args.commit,
        )
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"include": result["include"]}))
        return 0
    errors = verify(json.loads(args.plan.read_text()), args.reports, args.job_result)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
