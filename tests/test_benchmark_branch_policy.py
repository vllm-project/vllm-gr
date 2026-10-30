# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import unittest
from itertools import product
from pathlib import Path

import yaml

from tools.ci.resolve_benchmark_policy import load_policy, resolve_policy


class TestBenchmarkBranchPolicy(unittest.TestCase):
    def test_workflow_branch_and_device_scope(self) -> None:
        root = Path(__file__).resolve().parents[1]
        policy = load_policy(root / ".github/benchmark-gate.yaml")
        workflow = yaml.safe_load((root / ".github/workflows/device-validation.yml").read_text())
        step = next(
            item
            for item in workflow["jobs"]["validate"]["steps"]
            if item.get("name") == "Resolve base-owned benchmark policy"
        )
        self.assertEqual(
            step["env"]["POLICY_TARGET_BRANCH"],
            "${{ inputs.base_ref }}",
        )
        # Execute the workflow's actual selection logic with stubbed NPU probing.
        # GPU selection must not require nvidia-smi or any model-specific probe.
        selection = step["run"].split("args=(", 1)[0]
        self.assertNotIn("nvidia-smi", selection)
        script = (
            """
command() { [[ "$*" == "-v npu-smi" && "$TEST_DEVICE" == "npu" ]]; }
python() { printf '%s\\n' "$TEST_BOARD"; }
"""
            + selection
            + """
printf '%s\\n' "$device" "$board_id"
"""
        )
        keys = ("PASS_AT_1_THRESHOLD", "PASS_AT_32_THRESHOLD", "RECALL_AT_32_THRESHOLD")
        for branch, (device, hardware_board) in product(
            ("decode_graph", "main", "feature", ""),
            (
                ("gpu", "unknown"),
                ("npu", "Ascend910_9362"),
                ("npu", "OtherNPU"),
                ("npu", "unknown"),
            ),
        ):
            with self.subTest(branch=branch, device=device, hardware_board=hardware_board):
                result = subprocess.run(
                    ["bash", "-eu", "-o", "pipefail", "-c", script],
                    env={
                        **os.environ,
                        "POLICY_TARGET_BRANCH": branch,
                        "TEST_DEVICE": device,
                        "TEST_BOARD": hardware_board,
                    },
                    text=True,
                    capture_output=True,
                    check=True,
                )
                actual_device, board = result.stdout.splitlines()
                self.assertEqual(actual_device, device)
                enabled = branch == "decode_graph"
                expected_board = hardware_board
                if enabled:
                    expected_board = f"decode_graph_{device}"
                    if device == "npu" and hardware_board == "Ascend910_9362":
                        expected_board = "decode_graph_Ascend910_9362"
                self.assertEqual(board, expected_board)
                values = resolve_policy(policy, actual_device, board)
                expected = (0.05, 0.21, 0.03) if enabled else (0.09, 0.29, 0.04)
                self.assertEqual(tuple(float(values[key]) for key in keys), expected)
                self.assertEqual(values["ACCURACY_FAILURE_MODE"], "blocking")
                # Preserve all existing performance and Pangu settings.
                baseline = resolve_policy(policy, device, hardware_board)
                for key in values:
                    if key not in (*keys, "BENCHMARK_BOARD_ID"):
                        self.assertEqual(values[key], baseline[key], key)
