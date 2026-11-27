# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Prepare measurement inputs without changing legacy policy or repetition logic."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from tests.helpers.artifacts import ensure_derived, snapshot_spec
from tests.helpers.gate_process import run_gate_process
from tests.helpers.onerec_artifacts import ensure_onerec_constraints
from tests.helpers.paths import REPO_ROOT

SCRIPTS = {
    "recall": "run_beam_v1_recall.sh",
    "benchmark": "run_benchmark_and_verify.sh",
    "performance": "run_beam_v1_performance.sh",
    "performance-ab": "run_beam_v1_performance_ab.sh",
}


def video_inputs(dataset: Path) -> Path:
    catalog_cli = REPO_ROOT / "benchmarks/open_one_rec/tools/create_task_catalog.py"
    triples_cli = REPO_ROOT / "tools/constraints/build_constraint_triples.py"
    sources = {
        str(p.relative_to(REPO_ROOT)): p
        for folder in ("benchmarks/open_one_rec", "vllm_gr/constraints")
        for p in (REPO_ROOT / folder).rglob("*.py")
    }
    sources.update(
        video=dataset / "video/video_test.parquet",
        mapping=dataset / "sid2pid.json",
        compiler=triples_cli,
    )

    def build(output: Path) -> None:
        commands = [
            [
                sys.executable,
                str(catalog_cli),
                "-t",
                "video",
                "-d",
                str(dataset),
                "-o",
                str(output / "catalog.json"),
            ],
            [
                sys.executable,
                str(triples_cli),
                "--catalog",
                str(output / "catalog.json"),
                "--output",
                str(output / "triples.json"),
                "--begin",
                "<|sid_begin|>",
                "--end",
                "<|sid_end|>",
            ],
        ]
        for index, command in enumerate(commands):
            log = output / f"prepare-{index}.log"
            code = run_gate_process(command, cwd=REPO_ROOT, log=log, timeout=600)
            if code:
                raise RuntimeError(f"Video input preparation failed ({code}): {log.read_text()}")

    return ensure_derived(
        name="video-gate-inputs",
        inputs=sources,
        parameters={"task": "video", "begin": "<|sid_begin|>", "end": "<|sid_end|>"},
        builder_version="video-gate-v1",
        required_files=("catalog.json", "triples.json"),
        builder=build,
    )


def prepare_environment(suite: str, output: Path, fixture: Callable[[str], Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    spec = snapshot_spec("onerec1b7")
    env.setdefault("VLLM_GR_TEST_MODEL", spec.repo_id)
    revision_key = "V1_RECALL_MODEL_REVISION" if suite == "recall" else "V1_PERF_MODEL_REVISION"
    env.setdefault(revision_key, spec.revision)
    if suite == "recall":
        dataset = (
            Path(env["V1_RECALL_DATASET_PATH"])
            if "V1_RECALL_DATASET_PATH" in env
            else fixture("onerec_dataset")
        )
        for key, mapping in (
            ("V1_RECALL_TRIPLES_PATH", "sid2pid"),
            ("V1_RECALL_TRIPLES_IID_PATH", "sid2iid"),
        ):
            if key not in env:
                env[key] = str(ensure_onerec_constraints(dataset, mapping=mapping) / "triples.json")
        env.update(
            V1_RECALL_DATASET_PATH=str(dataset),
            V1_RECALL_OUTPUT_DIR=str(output),
            VLLM_GR_PREPARE_RECALL_MODEL="1",
        )
    elif suite == "benchmark":
        dataset = Path(env["DATASET_PATH"]) if "DATASET_PATH" in env else fixture("onerec_dataset")
        env.update(
            VLLM_GR_TEST_MODEL=str(fixture("onerec_model")),
            DATASET_PATH=str(dataset),
            VLLM_GR_BENCHMARK_CONFIG_DIR=str(output / "config"),
        )
        if "CATALOG_PATH" not in env and "TRIPLES_PATH" not in env:
            prepared = video_inputs(dataset)
            env.update(
                CATALOG_PATH=str(prepared / "catalog.json"),
                TRIPLES_PATH=str(prepared / "triples.json"),
            )
        for key in ("CATALOG_PATH", "TRIPLES_PATH"):
            if key not in env or not Path(env[key]).is_file():
                raise ValueError(f"Explicit video inputs require a valid {key}")
    else:
        env["V1_PERF_OUTPUT_DIR"] = str(output)
        if "TRIPLES_PATH" not in env:
            prepared = video_inputs(fixture("onerec_dataset"))
            env["TRIPLES_PATH"] = str(prepared / "triples.json")
    return env


def validate_outputs(suite: str, output: Path) -> None:
    required = {
        "recall": ("b4_repeat_summary.json",),
        "performance": ("setup.json", "offline.json", "online.json"),
        "performance-ab": ("context.json", "report.json"),
        "benchmark": (),  # The original gate validates every parsed metric and policy decision.
    }[suite]
    for name in required:
        data = json.loads((output / name).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            raise ValueError(f"Empty/invalid measurement report: {name}")
    if suite == "performance":
        for mode in ("offline", "online"):
            if json.loads((output / f"{mode}.json").read_text()).get("complete") is not True:
                raise ValueError(f"Incomplete {mode} performance result")


def execute(suite: str, output: Path, fixture: Callable[[str], Any]) -> None:
    if os.name != "posix" or not shutil.which("bash"):
        raise RuntimeError("Measurement gates require POSIX Bash")
    output.mkdir(parents=True, exist_ok=False)
    env = prepare_environment(suite, output, fixture)
    script = REPO_ROOT / "tests/system" / SCRIPTS[suite]
    status = run_gate_process(
        ["bash", str(script)], cwd=REPO_ROOT, env=env, log=output / "measurement.log", timeout=10800
    )
    if status:
        raise AssertionError(f"{suite} exited {status}; see {output / 'measurement.log'}")
    validate_outputs(suite, output)
