# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Manage established accuracy/performance policies as explicit pytest gates."""

import os
from pathlib import Path

import pytest

from tests.helpers.measurement_gate import SCRIPTS, execute

pytestmark = [pytest.mark.l4, pytest.mark.slow, pytest.mark.model_test]


@pytest.fixture(autouse=True)
def reset_vocab_slice_config() -> None:
    pass


@pytest.mark.parametrize(
    "suite",
    [
        pytest.param(
            name,
            marks=[
                pytest.mark.gpu_test,
                *(
                    [pytest.mark.npu_test]
                    if name in {"benchmark"}
                    else []
                ),
            ],
        )
        for name in SCRIPTS
    ],
)
def test_measurement(suite: str, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    if os.environ.get("VLLM_GR_MEASUREMENT_GATE") != "1":
        pytest.skip("Use run_gate to opt in to policy-controlled measurements")
    root = Path(os.environ.get("VLLM_GR_MEASUREMENT_REPORT_DIR", str(tmp_path)))
    execute(suite, root / suite, request.getfixturevalue)
