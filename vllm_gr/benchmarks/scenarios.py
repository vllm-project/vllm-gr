# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Scenario registry for ``vllm bench --gr`` workloads in the source checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchScenario:
    name: str
    help: str
    module: str | None = None
    path: Path | None = None
    argv_prefix: tuple[str, ...] = ()


_REPO_ROOT = Path(__file__).resolve().parents[2]

SCENARIOS: dict[str, BenchScenario] = {
    "open-one-rec": BenchScenario(
        "open-one-rec",
        "OneRec evaluation/throughput through vLLM's bench entrypoint",
        module="benchmarks.open_one_rec.one_rec_main",
        argv_prefix=("bench",),
    ),
    "beam-performance": BenchScenario(
        "beam-performance",
        "Beam V1 performance gate (prepare/offline/online)",
        module="benchmarks.beam_search.measure_latency",
    ),
    "beam-performance-ab": BenchScenario(
        "beam-performance-ab",
        "Compare recorded base and PR result directories and write a report",
        module="benchmarks.beam_search.compare_latency_ab",
    ),
    "beam-recall": BenchScenario(
        "beam-recall",
        "Beam V1 recall gate",
        module="benchmarks.beam_search.measure_onerec_recall",
    ),
    "qwen": BenchScenario(
        "qwen",
        "Qwen offline Beam search benchmark",
        module="benchmarks.qwen.benchmark",
    ),
    "ab-offline": BenchScenario(
        "ab-offline",
        "Offline OneRec cold/warm latency benchmark",
        module="benchmarks.open_one_rec.offline_ab",
    ),
}


def scenario_names() -> tuple[str, ...]:
    return tuple(SCENARIOS)


def run_scenario(name: str, argv: list[str]) -> None:
    """Dispatch to the workload's own argument parser and result format."""

    scenario = SCENARIOS[name]
    # Accept the old nested spelling as well as the documented shorter form.
    if scenario.argv_prefix and argv[:1] == list(scenario.argv_prefix):
        argv = argv[1:]
    argv = [*scenario.argv_prefix, *argv]
    relative_path = (
        Path(*scenario.module.split(".")).with_suffix(".py")
        if scenario.module is not None
        else scenario.path
    )
    assert relative_path is not None
    # The benchmark workloads and their data belong to the source checkout.
    # An installed CLI may be invoked from any directory within that checkout.
    cwd = Path.cwd()
    for root in (_REPO_ROOT, cwd, *cwd.parents):
        if (root / "vllm_gr" / "__init__.py").is_file() and (root / relative_path).is_file():
            break
    else:
        raise SystemExit(
            f"Benchmark {name!r} requires a vllm-gr source checkout containing {relative_path}"
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    # Prefer the selected checkout on PYTHONPATH over the caller's directory.
    args = ["-P", "-m", scenario.module] if scenario.module else [str(root / relative_path)]
    result = subprocess.run([sys.executable, *args, *argv], env=env)
    if result.returncode:
        raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
