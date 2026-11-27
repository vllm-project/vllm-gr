# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Pytest ownership of the existing tiny-model Beam functional drivers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tests.helpers.gate_process import run_gate_process
from tests.helpers.paths import REPO_ROOT

pytestmark = [
    pytest.mark.gpu_test,
    pytest.mark.cards_1,
    pytest.mark.l3,
    pytest.mark.slow,
    pytest.mark.timeout(3720),
]
SCENARIOS = {
    "checkpoint": ("run_checkpoint_reload", ()),
    "checkpoint-graph": ("run_checkpoint_reload", ("--graph",)),
    "trie": ("run_trie_reload", ()),
    "trie-graph": ("run_trie_reload", ("--graph",)),
    "pause-drain": ("run_pause_drain", ()),
    "reload-memory": ("run_reload_memory", ()),
    "reload-memory-graph": ("run_reload_memory", ("--graph",)),
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_beam_functional(scenario: str, tmp_path: Path) -> None:
    if os.environ.get("VLLM_GR_FUNCTIONAL_GATE") != "1":
        pytest.skip("Opt in with VLLM_GR_FUNCTIONAL_GATE=1 on a prepared CUDA runner")
    output = tmp_path / scenario
    output.mkdir()
    driver, options = SCENARIOS[scenario]
    log = output / "driver.log"
    status = run_gate_process(
        [sys.executable, "-m", f"tests.e2e.{driver}", "--output", str(output), *options],
        cwd=REPO_ROOT,
        log=log,
        timeout=3600,
        env={**os.environ, "VLLM_GR_BEAM_GATE_GROUP": "1"},
    )
    assert status == 0, f"Beam driver exited {status}; see {log}"
    result = output / "result.json"
    assert result.is_file(), f"Missing Beam result: {result}"
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report.get("passed") is True, report
