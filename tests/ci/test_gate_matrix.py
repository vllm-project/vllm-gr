# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-def,no-any-return"
"""Buildkite planning retains hardware, allocation size and native suite routing."""

import json

import pytest

from tests.helpers.paths import REPO_ROOT
from tools.ci.gate_matrix import plan

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_model_card_suite_matrix():
    registry = {
        "cuda:H100": [{"cards": n, "preset": f"h100_{n}", "suites": ["unit"]} for n in (1, 2, 4)]
    }
    steps = plan(registry, ["cuda:H100"], ["unit"])
    assert len(steps) == 3
    assert len({step["key"] for step in steps}) == 3
    for n, step in zip((1, 2, 4), steps):
        assert f"--cards {n}" in step["command"]
        assert step["mirror_hardwares"] == f"h100_{n}"
        assert step["env"]["VLLM_GR_TARGET_DEVICE"] == "cuda"


@pytest.mark.parametrize("suite", ["performance-ab", "invented"])
def test_unsupported_suites_rejected(suite):
    with pytest.raises(ValueError):
        plan({}, ["cuda:H100"], [suite])


def test_unknown_allocation_and_npu_cuda_suite_rejected():
    with pytest.raises(ValueError, match="No provisioned"):
        plan({"npu:A3": []}, ["npu:A3"], ["unit"])
    with pytest.raises(ValueError, match="requires CUDA"):
        plan(
            {"npu:A3": [{"cards": 1, "preset": "a3_1", "suites": ["recall"]}]},
            ["npu:A3"],
            ["recall"],
        )


def test_resource_requires_isolation():
    registry = {"cuda:H100": [{"cards": 2, "preset": "h100_2", "suites": ["resource"]}]}
    with pytest.raises(ValueError, match="isolated"):
        plan(registry, ["cuda:H100"], ["resource"])
    registry["cuda:H100"][0]["isolated"] = True
    assert (
        plan(registry, ["cuda:H100"], ["resource"])[0]["env"]["VLLM_GR_ISOLATED_RUNNER"] == "true"
    )


def test_checked_in_matrix_uses_existing_l4_preset():
    registry = json.loads((REPO_ROOT / ".buildkite/gate-runners.json").read_text())
    targets = [target for target, allocations in registry.items() if allocations]
    assert targets == ["cuda:L4"]
    steps = plan(registry, targets, ["unit"])
    assert steps[0]["mirror_hardwares"] == "l4_1"
