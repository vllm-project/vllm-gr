# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Session fixtures prepare inputs only for tests that request them."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers.artifacts import ensure_dataset, ensure_model, snapshot_spec
from tests.helpers.onerec_artifacts import ensure_onerec_constraints


def _override(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


@pytest.fixture(scope="session")  # type: ignore[misc]
def onerec_model() -> Path:
    return ensure_model(snapshot_spec("onerec1b7"), local_override=_override("VLLM_GR_TEST_MODEL"))


@pytest.fixture(scope="session")  # type: ignore[misc]
def onerec_dataset() -> Path:
    root = ensure_dataset(
        snapshot_spec("onerec_benchmark"), local_override=_override("VLLM_GR_TEST_DATASET")
    )
    return root / "benchmark_data"


@pytest.fixture(scope="session")  # type: ignore[misc]
def onerec_pid_constraints(onerec_dataset: Path) -> Path:
    return ensure_onerec_constraints(onerec_dataset, mapping="sid2pid")


@pytest.fixture(scope="session")  # type: ignore[misc]
def onerec_iid_constraints(onerec_dataset: Path) -> Path:
    return ensure_onerec_constraints(onerec_dataset, mapping="sid2iid")
