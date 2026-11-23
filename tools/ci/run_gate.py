# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""One explicit entrypoint for device-scoped pytest suites and durable gate reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from tests.helpers.gate_process import run_gate_process
from tests.helpers.hardware import (
    CARD_COUNTS,
    SKUS,
    parse_target,
    probe_devices,
    validate_inventory,
)
from tests.helpers.paths import REPO_ROOT

SUITES = {
    "unit": ["tests", "--ignore=tests/development/test_resource_probe.py", "-m", "l1 or l2"],
    "resource": ["tests/development/test_resource_probe.py"],
    "functional": ["tests/st/test_beam_functional_gate.py"],
}
TARGETS = ["host", *(f"{backend}:{sku}" for backend, skus in SKUS.items() for sku in skus)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", choices=sorted(SUITES), required=True)
    parser.add_argument("--hardware", choices=TARGETS, required=True)
    parser.add_argument("--cards", type=int, choices=CARD_COUNTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    cards = args.cards
    resource = "resource" in args.suite
    if cards is not None or resource:
        target = parse_target(args.hardware)
        if target.backend == "host":
            parser.error("An accelerator is required for device allocations and resource probes")
        inventory = probe_devices(target.backend)
        validate_inventory(target, inventory)
        if cards is not None and len(inventory.names) != cards:
            parser.error("--cards must equal the allocated runtime-visible device count")
    if resource:
        available = len(inventory.names)
        scenarios = ["single"] if cards in (None, 1) else []
        if available >= 2 and cards in (None, 2):
            scenarios.append("pair")
            if target.backend == "cuda":
                scenarios.append("masked")
        if not scenarios:
            parser.error("Resource probes require a one- or two-card lane")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    suites = list(dict.fromkeys(args.suite))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    summary: dict[str, object] = {
        "schema": 1,
        "run_id": run_id,
        "commit": commit,
        "hardware": args.hardware,
        "required_suites": suites,
        "cards": cards,
        "policy_board": os.environ.get("BENCHMARK_BOARD_ID"),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "results": {},
        "status": "running",
    }
    results: dict[str, object] = {}
    summary["results"] = results

    def save() -> None:
        temp = output / "summary.tmp"
        temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temp.replace(output / "summary.json")

    save()
    failed = False
    try:
        for suite in suites:
            directory = output / suite
            directory.mkdir()
            env = dict(
                os.environ,
                VLLM_GR_FUNCTIONAL_GATE="1",
                VLLM_GR_FUNCTIONAL_REPORT_DIR=str(directory / "functional"),
                VLLM_GR_RESOURCE_REPORT_DIR=str(directory / "resource"),
            )
            env.pop("PYTEST_ADDOPTS", None)
            selection = SUITES[suite]
            if suite == "resource":
                selection = [f"{selection[0]}::test_resource_probe[{name}]" for name in scenarios]
            elif cards is not None and suite == "unit":
                marker = (
                    f"cards_{cards}"
                    if cards > 1
                    else "not ("
                    + " or ".join(f"cards_{count}" for count in CARD_COUNTS if count > 1)
                    + ")"
                )
                selection = [*selection[:-1], f"({selection[-1]}) and ({marker})"]
            command = [
                sys.executable,
                "-m",
                "pytest",
                "--run-slow",
                "--strict-test-levels",
                "--test-hardware",
                args.hardware,
                "--timeout=0",
                "-p",
                "tools.ci.strict_gate",
                "--junitxml",
                str(directory / "junit.xml"),
                *selection,
            ]
            results[suite] = {"status": "running"}
            save()
            try:
                code = run_gate_process(
                    command,
                    cwd=REPO_ROOT,
                    log=directory / "pytest.log",
                    timeout=args.timeout,
                    env=env,
                )
                passed = code == 0
                results[suite] = {"status": "passed" if passed else "failed", "exit_code": code}
            except (OSError, subprocess.TimeoutExpired) as exc:
                passed = False
                results[suite] = {"status": "failed", "error": str(exc)}
            failed |= not passed
            save()
        summary["status"] = "failed" if failed else "passed"
    except KeyboardInterrupt:
        summary["status"] = "cancelled"
        return 130
    finally:
        save()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
