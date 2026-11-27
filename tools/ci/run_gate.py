# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run selected pytest suites on the allocated hardware."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

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
    "unit": [
        "tests",
        "--ignore=tests/development/test_resource_probe.py",
        "--ignore=tests/e2e/test_beam_functional_gate.py",
    ],
    "resource": ["tests/development/test_resource_probe.py"],
    "functional": ["tests/e2e/test_beam_functional_gate.py"],
}
for _suite in ("recall", "benchmark", "performance", "performance-ab"):
    SUITES[_suite] = [f"tests/system/test_measurement_gates.py::test_measurement[{_suite}]"]
SUITES["unit"].append("--ignore=tests/system/test_measurement_gates.py")
TARGETS = ["host", *(f"{backend}:{sku}" for backend, skus in SKUS.items() for sku in skus)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", choices=sorted(SUITES), required=True)
    parser.add_argument("--hardware", choices=TARGETS, required=True)
    parser.add_argument("--cards", type=int, choices=CARD_COUNTS)
    parser.add_argument("--levels", default="l1 or l2 or l3")
    parser.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    target = parse_target(args.hardware)
    cuda_suites = {"functional", "recall", "performance", "performance-ab"}
    if cuda_suites.intersection(args.suite) and target.backend != "cuda":
        parser.error("Functional, recall and performance suites require CUDA")
    if "resource" in args.suite and target.backend == "host":
        parser.error("Resource probes require an accelerator")
    if args.cards is not None:
        if target.backend == "host":
            parser.error("--cards requires an accelerator")
        inventory = probe_devices(target.backend)
        validate_inventory(target, inventory)
        if len(inventory.names) != args.cards:
            parser.error("--cards must equal the allocated runtime-visible device count")
    env = dict(os.environ, VLLM_GR_FUNCTIONAL_GATE="1", VLLM_GR_MEASUREMENT_GATE="1")
    env.pop("PYTEST_ADDOPTS", None)
    for suite in dict.fromkeys(args.suite):
        expression = args.levels if suite == "unit" else ""
        if args.cards:
            cards = (
                f"cards_{args.cards}"
                if args.cards > 1
                else ("not (" + " or ".join(f"cards_{n}" for n in CARD_COUNTS if n > 1) + ")")
            )
            expression = f"({expression}) and ({cards})" if expression else cards
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--run-slow",
            "--strict-test-levels",
            "--test-hardware",
            args.hardware,
            "--timeout=0",
            *SUITES[suite],
        ]
        if expression:
            command += ["-m", expression]
        if suite != "unit":
            command.append("--strict-gate")
        print(f"Running {suite} on {args.hardware}", flush=True)
        try:
            status = run_gate_process(command, cwd=REPO_ROOT, timeout=args.timeout, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
