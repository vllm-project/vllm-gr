# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Build optional gate jobs using the repository's Buildkite hardware presets."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any

import yaml

from tests.helpers.hardware import CARD_COUNTS, parse_target
from tests.helpers.paths import REPO_ROOT
from tools.ci.resolve_benchmark_policy import load_policy, resolve_policy

SUITES = {"unit", "resource", "functional", "recall", "benchmark", "performance"}


def plan(registry: dict[str, Any], targets: list[str], suites: list[str]) -> list[dict[str, Any]]:
    if not targets or not suites or set(suites) - SUITES:
        raise ValueError("Select registered targets and supported gate suites")
    policy = load_policy(REPO_ROOT / ".github/benchmark-gate.yaml")
    steps = []
    for target in dict.fromkeys(targets):
        hardware = parse_target(target)
        allocations = registry.get(target)
        if not allocations:
            raise ValueError(f"No provisioned allocations for {target}")
        for suite in dict.fromkeys(suites):
            matched = False
            for allocation in allocations:
                cards = allocation["cards"]
                if (
                    not isinstance(cards, int)
                    or isinstance(cards, bool)
                    or cards not in CARD_COUNTS
                    or hardware.backend == "host"
                ):
                    raise ValueError("Accelerator allocations require a card count from 1 to 8")
                if suite not in allocation["suites"]:
                    continue
                if hardware.backend == "npu" and suite in {"functional", "recall", "performance"}:
                    raise ValueError(f"{suite} requires CUDA")
                if suite == "resource" and allocation.get("isolated") is not True:
                    raise ValueError("Resource probes require isolated allocations")
                device = "gpu" if hardware.backend == "cuda" else "npu"
                env = resolve_policy(policy, device, allocation.get("policy_board", "default"))
                env.update(
                    VLLM_GR_TARGET_DEVICE=hardware.backend,
                    VLLM_GR_ISOLATED_RUNNER="true" if allocation.get("isolated") else "false",
                )
                steps.append(
                    {
                        "label": f"{target} / {cards} cards / {suite}",
                        "key": target.replace(":", "-") + f"-{cards}-{suite}",
                        "mirror_hardwares": allocation["preset"],
                        "timeout_in_minutes": 250,
                        "env": env,
                        "command": f"python -m tools.ci.run_gate --hardware {target} --cards {cards} --suite {suite}",
                    }
                )
                matched = True
            if not matched:
                raise ValueError(f"No allocation supports {target} / {suite}")
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / ".buildkite/gate-runners.json")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--suites", nargs="+", default=["unit"])
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    targets = args.targets or [name for name, allocations in registry.items() if allocations]
    steps = plan(registry, targets, args.suites)
    # Reuse the same preset expansion as ready/merge pipelines.
    uploader = runpy.run_path(str(REPO_ROOT / ".buildkite/common/scripts/upload_pipeline.py"))
    presets = yaml.safe_load((REPO_ROOT / ".buildkite/common/ci_mirror_hardwares.yml").read_text())
    uploader["expand_steps"](steps, presets["mirror_hardwares"])
    print(yaml.safe_dump({"steps": steps}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
