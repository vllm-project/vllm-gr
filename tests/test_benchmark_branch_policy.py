# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tools.ci.resolve_benchmark_policy import load_policy, resolve_policy

BRANCH_ENV_NAMES = {
    "CHANGE_TARGET",
    "CHANGE_BRANCH",
    "CHANGE_ID",
    "BRANCH_NAME",
    "GITHUB_BASE_REF",
    "GITHUB_HEAD_REF",
    "GITHUB_EVENT_NAME",
    "GITHUB_REF_TYPE",
    "GITHUB_REF_NAME",
    "GITHUB_REF",
    "POLICY_TARGET_BRANCH",
    "TARGET_BRANCH",
    "BRANCH",
    "GIT_BRANCH",
}


class TestBenchmarkBranchPolicy(unittest.TestCase):
    def setUp(self) -> None:
        clean_env = {key: value for key, value in os.environ.items() if key not in BRANCH_ENV_NAMES}
        environment = patch.dict(os.environ, clean_env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_workflow_branch_and_device_scope(self) -> None:
        root = Path(__file__).resolve().parents[1]
        policy = load_policy(root / ".github/benchmark-gate.yaml")
        workflow = yaml.safe_load((root / ".github/workflows/pre-commit.yml").read_text())
        step = next(
            item
            for item in workflow["jobs"]["benchmark-policy"]["steps"]
            if item.get("name") == "Resolve benchmark policy"
        )
        self.assertEqual(
            step["env"]["POLICY_TARGET_BRANCH"],
            "${{ github.event.pull_request.base.ref || github.base_ref || github.ref_name }}",
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
                ("npu", "Ascend_9362"),
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
                enabled = branch in {"decode_graph", "main"}
                expected_board = hardware_board
                if enabled:
                    expected_board = f"decode_graph_{device}"
                    if device == "npu" and hardware_board == "Ascend910_9362":
                        expected_board = "decode_graph_Ascend910_9362"
                self.assertEqual(board, expected_board)
                with patch.dict(os.environ, {"CHANGE_TARGET": branch}):
                    values = resolve_policy(policy, actual_device, board)
                    baseline = resolve_policy(policy, device, hardware_board)
                expected = (0.05, 0.21, 0.03) if device == "npu" or enabled else (0.09, 0.29, 0.04)
                self.assertEqual(tuple(float(values[key]) for key in keys), expected)
                self.assertEqual(values["ACCURACY_FAILURE_MODE"], "blocking")
                # Preserve all existing performance and Pangu settings.
                for key in values:
                    if key not in (*keys, "BENCHMARK_BOARD_ID"):
                        self.assertEqual(values[key], baseline[key], key)

    def test_all_npu_boards_resolve_unified_floors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        policy = load_policy(root / ".github/benchmark-gate.yaml")
        keys = ("PASS_AT_1_THRESHOLD", "PASS_AT_32_THRESHOLD", "RECALL_AT_32_THRESHOLD")
        physical = resolve_policy(policy, "npu", "Ascend910_9362")
        self.assertEqual(tuple(float(physical[key]) for key in keys), (0.05, 0.21, 0.03))
        for board in (
            "Ascend910_9362",
            "Ascend_9362",
            "Ascend910B1",
            "decode_graph_Ascend910_9362",
            "decode_graph_npu",
            "OtherNPU",
            "unknown",
        ):
            for branch in ("main", "decode_graph", "feature", ""):
                with self.subTest(board=board, branch=branch):
                    with patch.dict(os.environ, {"CHANGE_TARGET": branch}):
                        values = resolve_policy(policy, "npu", board)
                    expected = (0.05, 0.21, 0.03)
                    self.assertEqual(tuple(float(values[key]) for key in keys), expected)
                    self.assertEqual(values["ACCURACY_FAILURE_MODE"], "blocking")
                    self.assertEqual(values["BENCHMARK_BOARD_ID"], board)
                    self.assertEqual(
                        float(values["P99_TTFT_THRESHOLD"]),
                        280 if board in {"Ascend910_9362", "decode_graph_Ascend910_9362"} else 300,
                    )
                    self.assertEqual(
                        values["PANGU1B_OFFLINE_ACCURACY_THRESHOLD"],
                        physical["PANGU1B_OFFLINE_ACCURACY_THRESHOLD"],
                    )
                    self.assertEqual(
                        values["PANGU1B_ONLINE_ACCURACY_THRESHOLD"],
                        physical["PANGU1B_ONLINE_ACCURACY_THRESHOLD"],
                    )
            with patch.dict(os.environ, {"CHANGE_TARGET": "main", "GITHUB_BASE_REF": "feature"}):
                values = resolve_policy(policy, "npu", board)
                self.assertEqual(tuple(float(values[key]) for key in keys), (0.05, 0.21, 0.03))

    def test_all_npu_profiles_preserve_custom_performance_and_pangu(self) -> None:
        root = Path(__file__).resolve().parents[1]
        policy = load_policy(root / ".github/benchmark-gate.yaml")
        policy["device_overrides"]["npu"]["boards"]["CustomNPU"] = {
            "thresholds": {
                "accuracy": {"pass_at_1_min": 0.8},
                "performance": {"p99_ttft_max": 123, "offline_elapsed_seconds_max": 42},
            }
        }
        for branch in ("main", "decode_graph", "feature", ""):
            with self.subTest(branch=branch), patch.dict(os.environ, {"CHANGE_TARGET": branch}):
                values = resolve_policy(policy, "npu", "CustomNPU")
                self.assertEqual(
                    tuple(
                        float(values[key])
                        for key in (
                            "PASS_AT_1_THRESHOLD",
                            "PASS_AT_32_THRESHOLD",
                            "RECALL_AT_32_THRESHOLD",
                        )
                    ),
                    (0.05, 0.21, 0.03),
                )
                self.assertEqual(float(values["P99_TTFT_THRESHOLD"]), 123)
                self.assertEqual(float(values["ELAPSED_TIME_OFFLINE_THRESHOLD"]), 42)
                self.assertEqual(float(values["PANGU1B_OFFLINE_ACCURACY_THRESHOLD"]), 0.875)
                self.assertEqual(values["PANGU1B_OFFLINE_ACCURACY_FAILURE_MODE"], "blocking")
                gpu = resolve_policy(policy, "gpu", "unknown")
                self.assertEqual(float(gpu["PASS_AT_1_THRESHOLD"]), 0.09)

    def test_resolver_cli_unified_npu_and_strict_fallback(self) -> None:
        root = Path(__file__).resolve().parents[1]
        policy_path = root / ".github/benchmark-gate.yaml"
        policy = load_policy(policy_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
            del raw["device_overrides"]["npu"]["boards"]["decode_graph_Ascend910_9362"]
            no_profile = temp / "no-profile.yaml"
            no_profile.write_text(yaml.safe_dump(raw), encoding="utf-8")
            raw["device_overrides"]["npu"]["boards"]["decode_graph_npu"]["thresholds"] = {
                "performance": {"p99_ttft_max": 111}
            }
            no_accuracy = temp / "no-accuracy.yaml"
            no_accuracy.write_text(yaml.safe_dump(raw), encoding="utf-8")
            del raw["device_overrides"]["npu"]["boards"]["decode_graph_npu"]
            no_generic_profile = temp / "no-generic-profile.yaml"
            no_generic_profile.write_text(yaml.safe_dump(raw), encoding="utf-8")

            strict = (0.09, 0.29, 0.04)
            scoped = (0.05, 0.21, 0.03)
            policy_file = ["--policy-file", str(policy_path)]
            # Exercise branch variables independently of the developer's environment.
            clean_env = dict(os.environ)
            cases: tuple[
                tuple[str, list[str], dict[str, str], str, str, tuple[float, float, float]], ...
            ] = (
                ("missing", policy_file, {}, "npu", "Ascend910_9362", scoped),
                (
                    "jenkins-main",
                    policy_file,
                    {"CHANGE_TARGET": "main"},
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "jenkins-decode",
                    policy_file,
                    {"CHANGE_TARGET": "decode_graph"},
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "jenkins-feature",
                    policy_file,
                    {"CHANGE_TARGET": "feature", "BRANCH_NAME": "main"},
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "jenkins-pr-source",
                    policy_file,
                    {"CHANGE_ID": "123", "CHANGE_BRANCH": "main", "BRANCH_NAME": "main"},
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "github-pr",
                    policy_file,
                    {
                        "GITHUB_EVENT_NAME": "pull_request",
                        "GITHUB_BASE_REF": "main",
                        "GITHUB_HEAD_REF": "feature",
                        "GITHUB_REF_NAME": "123/merge",
                        "GITHUB_REF_TYPE": "branch",
                    },
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "github-review",
                    policy_file,
                    {
                        "GITHUB_EVENT_NAME": "pull_request_review",
                        "GITHUB_BASE_REF": "decode_graph",
                        "GITHUB_REF_NAME": "123/merge",
                        "GITHUB_REF_TYPE": "branch",
                    },
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "github-pr-feature",
                    policy_file,
                    {
                        "GITHUB_EVENT_NAME": "pull_request",
                        "GITHUB_BASE_REF": "feature",
                        "GITHUB_HEAD_REF": "main",
                    },
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "github-pr-missing-base",
                    policy_file,
                    {
                        "GITHUB_EVENT_NAME": "pull_request",
                        "GITHUB_HEAD_REF": "main",
                        "GITHUB_REF_TYPE": "branch",
                        "GITHUB_REF_NAME": "main",
                    },
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "github-push",
                    policy_file,
                    {
                        "GITHUB_EVENT_NAME": "push",
                        "GITHUB_REF_TYPE": "branch",
                        "GITHUB_REF_NAME": "main",
                    },
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "github-manual",
                    policy_file,
                    {
                        "GITHUB_EVENT_NAME": "workflow_dispatch",
                        "GITHUB_REF_TYPE": "branch",
                        "GITHUB_REF_NAME": "decode_graph",
                    },
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "github-tag",
                    policy_file,
                    {
                        "GITHUB_EVENT_NAME": "push",
                        "GITHUB_REF_TYPE": "tag",
                        "GITHUB_REF_NAME": "main",
                    },
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "conflicting-targets",
                    policy_file,
                    {"CHANGE_TARGET": "main", "GITHUB_BASE_REF": "feature"},
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "agreeing-targets",
                    policy_file,
                    {"CHANGE_TARGET": "main", "GITHUB_BASE_REF": "main"},
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                ("other-npu", policy_file, {"CHANGE_TARGET": "main"}, "npu", "OtherNPU", scoped),
                ("gpu", policy_file, {"CHANGE_TARGET": "main"}, "gpu", "unknown", strict),
                (
                    "bootstrap",
                    ["--bootstrap-strict"],
                    {"CHANGE_TARGET": "main"},
                    "npu",
                    "Ascend910_9362",
                    strict,
                ),
                (
                    "bootstrap-unknown",
                    ["--bootstrap-strict"],
                    {"CHANGE_TARGET": "feature"},
                    "npu",
                    "unlisted-npu",
                    strict,
                ),
                (
                    "missing-profile",
                    ["--policy-file", str(no_profile)],
                    {"CHANGE_TARGET": "main"},
                    "npu",
                    "Ascend910_9362",
                    scoped,
                ),
                (
                    "missing-alias-profile",
                    ["--policy-file", str(no_generic_profile)],
                    {"CHANGE_TARGET": "main"},
                    "npu",
                    "Ascend_9362",
                    scoped,
                ),
                (
                    "missing-npu-profile-unknown-board",
                    ["--policy-file", str(no_generic_profile)],
                    {"CHANGE_TARGET": "main"},
                    "npu",
                    "unlisted-npu",
                    strict,
                ),
                (
                    "missing-npu-accuracy-unknown-board",
                    ["--policy-file", str(no_accuracy)],
                    {},
                    "npu",
                    "unlisted-npu",
                    strict,
                ),
            )
            for board, branch in product(
                (
                    "Ascend910_9362",
                    "Ascend_9362",
                    "Ascend910B1",
                    "decode_graph_Ascend910_9362",
                    "decode_graph_npu",
                    "unknown",
                    "unlisted-npu",
                ),
                ("main", "decode_graph", "feature", ""),
            ):
                cases += (
                    (
                        f"{board}-{branch or 'missing'}",
                        policy_file,
                        {"CHANGE_TARGET": branch},
                        "npu",
                        board,
                        scoped,
                    ),
                )
            for board in (
                "Ascend910_9362",
                "Ascend_9362",
                "decode_graph_Ascend910_9362",
                "unknown",
            ):
                cases += (
                    (
                        f"{board}-conflicting-targets",
                        policy_file,
                        {"CHANGE_TARGET": "main", "GITHUB_BASE_REF": "feature"},
                        "npu",
                        board,
                        scoped,
                    ),
                )
            keys = ("PASS_AT_1_THRESHOLD", "PASS_AT_32_THRESHOLD", "RECALL_AT_32_THRESHOLD")
            for index, (name, source, branch_env, device, board, expected) in enumerate(cases):
                with self.subTest(case=name):
                    env_path = temp / f"env-{index}"
                    summary_path = temp / f"summary-{index}"
                    subprocess.run(
                        [
                            sys.executable,
                            str(root / "tools/ci/resolve_benchmark_policy.py"),
                            *source,
                            "--device",
                            device,
                            "--board-id",
                            board,
                            "--source-ref",
                            "test",
                            "--github-env",
                            str(env_path),
                            "--github-step-summary",
                            str(summary_path),
                        ],
                        env={**clean_env, **branch_env},
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    values = dict(line.split("=", 1) for line in env_path.read_text().splitlines())
                    self.assertEqual(tuple(float(values[key]) for key in keys), expected)
                    self.assertEqual(values["ACCURACY_FAILURE_MODE"], "blocking")
                    self.assertEqual(values["BENCHMARK_BOARD_ID"], board)
                    baseline = resolve_policy(policy, device, board)
                    for key in baseline:
                        if key not in (*keys, "BENCHMARK_POLICY_NAME", "PANGU1B_POLICY_SOURCE"):
                            self.assertEqual(values[key], baseline[key], key)


pytestmark = pytest.mark.cpu_test
