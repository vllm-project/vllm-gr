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
    ArtifactError,
    artifact_identity,
    ensure_dataset,
    ensure_model,
    snapshot_spec,
)
from tests.helpers.paths import REPO_ROOT


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
def pangu_model(record_testsuite_property: Callable[[str, object], None]) -> Path:
    from tests.st.pangu1b_st_common import require_pangu_st_runtime

    # Keep the old runtime guard (including strict-CI failures), but let the
    # fixture prepare absent weights instead of skipping for a missing model.
    require_pangu_st_runtime(require_model=False)
    spec = snapshot_spec("pangu1b")
    local = _override("PANGU1B_MODEL_PATH", "PANGU_MODEL_PATH")
    path = ensure_model(spec, local_override=local)
    _record(
        "pangu_model",
        path,
        "local" if local else f"{spec.repo_id}@{spec.revision}",
        record_testsuite_property,
    )
    return path


@pytest.fixture(scope="session")  # type: ignore[misc]
def gsm8k_mini(record_testsuite_property: Callable[[str, object], None]) -> Path:
    from tests.helpers.artifacts import _digest, file_identity
    from tests.st.pangu1b_st_common import GSM8K_MINI, load_dataset

    local = _override("PANGU1B_DATASET_PATH")
    path: Path = (local if local is not None else REPO_ROOT / GSM8K_MINI).resolve()
    records = load_dataset(path, num_prompts=None)
    if not records or any(
        not isinstance(row, dict)
        or not isinstance(row.get("question"), str)
        or not isinstance(row.get("answer"), str)
        for row in records
    ):
        raise ArtifactError("GSM8K input requires nonempty question/answer records")
    # Canonical JSON identity is stable across Git's Windows line-ending conversion.
    canonical_sha = _digest(records)
    if local is None and canonical_sha != (
        "81b9e522a592da8d08ac9f463cebb7676def02c176628bf6d7ef350d5b5f65dc"
    ):
        raise ArtifactError("Bundled GSM8K-mini changed; review its accepted input identity")
    record_testsuite_property("gsm8k_mini.path", str(path))
    record_testsuite_property("gsm8k_mini.identity", json.dumps(file_identity(path)))
    record_testsuite_property("gsm8k_mini.canonical_sha256", canonical_sha)
    return path


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
