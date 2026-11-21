# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Lazy fixtures: downloads happen on the machine executing a selected test."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.helpers.artifacts import (
    artifact_identity,
    ensure_dataset,
    ensure_model,
    snapshot_spec,
)


def _override(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value).expanduser()
    return None


def _record(name: str, path: Path, source: str, record: Callable[[str, object], None]) -> None:
    record(f"{name}.path", str(path))
    record(f"{name}.source", source)
    record(f"{name}.identity", json.dumps(artifact_identity(path), sort_keys=True))


@pytest.fixture(scope="session")  # type: ignore[misc]
def onerec_model(record_testsuite_property: Callable[[str, object], None]) -> Path:
    spec = snapshot_spec("onerec1b7")
    local = _override("VLLM_GR_TEST_MODEL")
    path = ensure_model(spec, local_override=local)
    _record(
        "onerec_model",
        path,
        "local" if local else f"{spec.repo_id}@{spec.revision}",
        record_testsuite_property,
    )
    return path


@pytest.fixture(scope="session")  # type: ignore[misc]
def onerec_dataset(record_testsuite_property: Callable[[str, object], None]) -> Path:
    spec = snapshot_spec("onerec_benchmark")
    local = _override("VLLM_GR_TEST_DATASET")
    path = ensure_dataset(spec, local_override=local)
    _record(
        "onerec_dataset",
        path,
        "local" if local else f"{spec.repo_id}@{spec.revision}",
        record_testsuite_property,
    )
    return path / "benchmark_data"


@pytest.fixture(scope="session")  # type: ignore[misc]
def onerec_pid_constraints(
    onerec_dataset: Path, record_testsuite_property: Callable[[str, object], None]
) -> Path:
    from tests.helpers.onerec_artifacts import ensure_onerec_constraints

    path = ensure_onerec_constraints(onerec_dataset, mapping="sid2pid")
    _record("onerec_pid_constraints", path, "derived", record_testsuite_property)
    return path


@pytest.fixture(scope="session")  # type: ignore[misc]
def onerec_iid_constraints(
    onerec_dataset: Path, record_testsuite_property: Callable[[str, object], None]
) -> Path:
    from tests.helpers.onerec_artifacts import ensure_onerec_constraints

    path = ensure_onerec_constraints(onerec_dataset, mapping="sid2iid")
    _record("onerec_iid_constraints", path, "derived", record_testsuite_property)
    return path