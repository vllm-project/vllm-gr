# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Cached OneRec catalog/triples preparation using the existing CLI contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.helpers.artifacts import ArtifactError, ensure_derived
from tests.helpers.paths import REPO_ROOT


def ensure_onerec_constraints(
    dataset_dir: Path, *, mapping: str = "sid2pid", cache_dir: Path | None = None
) -> Path:
    """Return a directory containing catalog.json and triples.json.

    dataset_dir is the benchmark_data directory. This full-mapping path is
    separate from the video task catalog. Reuse the same conversion tools as
    the existing recall workflow; do not substitute another triples format.
    """
    if mapping not in ("sid2pid", "sid2iid"):
        raise ValueError("mapping must be sid2pid or sid2iid")
    converter = REPO_ROOT / "benchmarks/open_one_rec/tools/convert_catalog.py"
    triples = REPO_ROOT / "tools/constraints/build_constraint_triples.py"
    parameters = {"mapping": mapping, "begin": "<|sid_begin|>", "end": "<|sid_end|>"}

    def build(output: Path) -> None:
        commands = [
            [
                sys.executable,
                str(converter),
                "--dataset-path",
                str(dataset_dir),
                "--mapping-file",
                f"{mapping}.json",
                "--output-path",
                str(output / "catalog.json"),
            ],
            [
                sys.executable,
                str(triples),
                "--catalog",
                str(output / "catalog.json"),
                "--output",
                str(output / "triples.json"),
                "--begin",
                parameters["begin"],
                "--end",
                parameters["end"],
            ],
        ]
        for command in commands:
            try:
                result = subprocess.run(
                    command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False
                )
            except subprocess.TimeoutExpired as exc:
                raise ArtifactError("OneRec constraint preparation timed out") from exc
            if result.returncode:
                raise ArtifactError(f"OneRec constraint preparation failed:\n{result.stderr}")

    return ensure_derived(
        name="onerec-constraints",
        inputs={
            "mapping": dataset_dir / f"{mapping}.json",
            "converter": converter,
            "triples_cli": triples,
            "json_compiler": REPO_ROOT / "vllm_gr/constraints/json_compiler.py",
        },
        parameters=parameters,
        builder_version="onerec-mapping-v1",
        required_files=("catalog.json", "triples.json"),
        builder=build,
        cache_dir=cache_dir,
    )
