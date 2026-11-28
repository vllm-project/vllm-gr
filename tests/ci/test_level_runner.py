# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Hardware-free checks of the level runner's subprocess and exit handling."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from _pytest.mark.expression import Expression

from tests.helpers.paths import REPO_ROOT
from tools.ci.inproc_test_selection import pytest_collection_modifyitems

pytestmark = pytest.mark.cpu_test

ROOT = REPO_ROOT
RUNNER = ROOT / "tools/ci/run_test_level.sh"
INPROC = "tests/kernels/test_gpu_dynamic_beam_model.py"


@pytest.mark.l1
class TestLevelRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        fake_python = self.directory / "python3"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            'printf \'multiproc=%s\\n\' "${VLLM_ENABLE_V1_MULTIPROCESSING-unset}" >> "$CALL_LOG"\n'
            'printf \'selection=%s\\n\' "${VLLM_GR_INPROC_TEST_SELECTION-unset}" >> "$CALL_LOG"\n'
            'printf \'arg=%s\\n\' "$@" >> "$CALL_LOG"\n'
            "printf '%s\\n' END >> \"$CALL_LOG\"\n"
            'if [[ ${VLLM_GR_INPROC_TEST_SELECTION-unset} == include ]]; then exit "$INPROC_EXIT"; fi\n'
            'exit "$MAIN_EXIT"\n',
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

    def run_level(
        self,
        level: str,
        *,
        smoke: bool,
        main: int = 0,
        inproc: int = 0,
        device: str | None = None,
        exclude_migrated_gpu: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        bash = os.environ.get("TEST_BASH") or shutil.which("bash")
        self.assertIsNotNone(bash)
        assert bash is not None
        log = self.directory / "calls.log"
        env = {
            **os.environ,
            "PATH": f"{self.directory}{os.pathsep}{os.environ['PATH']}",
            "CALL_LOG": str(log),
            "MAIN_EXIT": str(main),
            "INPROC_EXIT": str(inproc),
        }
        env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        env.pop("VLLM_GR_TARGET_DEVICE", None)
        env.pop("VLLM_GR_EXCLUDE_MIGRATED_GPU_TESTS", None)
        if exclude_migrated_gpu:
            env["VLLM_GR_EXCLUDE_MIGRATED_GPU_TESTS"] = "1"
        if device is not None:
            env["VLLM_GR_TARGET_DEVICE"] = device
        if smoke:
            env["VLLM_GR_RUN_MODEL_SMOKE"] = "1"
        else:
            env.pop("VLLM_GR_RUN_MODEL_SMOKE", None)
        result = subprocess.run(
            [bash, str(RUNNER), level, "-k", "chosen"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        calls = log.read_text(encoding="utf-8").strip().split("END\n") if log.exists() else []
        return result, calls

    def test_smoke_splits_in_process_tests_and_preserves_options(self) -> None:
        result, calls = self.run_level("l4", smoke=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertIn("multiproc=unset", calls[0])
        self.assertIn("selection=exclude", calls[0])
        self.assertIn("multiproc=0", calls[1])
        self.assertIn("selection=include", calls[1])
        for call in calls:
            self.assertIn("arg=l4", call)
            self.assertIn("arg=-k\narg=chosen", call)
            self.assertIn("arg=tools.ci.inproc_test_selection", call)
            self.assertIn("arg=tests", call)

    def test_without_smoke_runs_once_in_original_environment(self) -> None:
        result, calls = self.run_level("pr", smoke=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn("multiproc=unset", calls[0])
        self.assertIn("arg=l1 or l2", calls[0])

    def test_buildkite_covers_all_ready_and_merge_levels(self) -> None:
        # Exercise the configured runner, rather than duplicating its marker logic.
        # CPU+accelerator marks must stay in the accelerator lane.
        platforms = [
            {"cpu_test"},
            {"gpu_test"},
            {"npu_test"},
            {"gpu_test", "npu_test"},
            {"cpu_test", "gpu_test"},
            {"cpu_test", "npu_test"},
        ]
        for phase in ("ready", "merge"):
            expressions = []
            for device in ("cuda", "npu"):
                pipeline = yaml.safe_load(
                    (ROOT / f".buildkite/test-{device}-{phase}.yml").read_text()
                )
                for step in pipeline["steps"]:
                    args = shlex.split(step["command"])
                    if args[0] == "pytest":
                        self.assertEqual(args[-1], "tests")
                        self.assertIn("--run-slow", args)
                        self.assertIn("--strict-test-levels", args)
                        expressions.append(args[args.index("-m") + 1])
                        continue
                    index = args.index("tools/ci/run_test_level.sh")
                    level = args[index + 1]
                    self.assertEqual(level, "pr" if phase == "ready" else "l3")
                    self.assertFalse(any(arg.startswith("tests/") for arg in args))
                    self.assertFalse(any(arg.startswith("--ignore") for arg in args))
                    (self.directory / "calls.log").unlink(missing_ok=True)
                    env = step["env"]
                    migration_flag = (
                        "VLLM_GR_EXCLUDE_MIGRATED_GPU_TESTS"
                        if device == "cuda"
                        else "VLLM_GR_EXCLUDE_MIGRATED_NPU_TESTS"
                    )
                    self.assertEqual(env.get(migration_flag, "0"), "0")
                    with patch.dict(os.environ, env):
                        result, calls = self.run_level(
                            level,
                            smoke=env.get("VLLM_GR_RUN_MODEL_SMOKE") == "1",
                            device=env["VLLM_GR_TARGET_DEVICE"],
                        )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if device == "cuda":
                        self.assertEqual(len(calls), 2)
                        self.assertIn("selection=exclude", calls[0])
                        self.assertIn("selection=include", calls[1])
                    for call in calls:
                        lines = call.splitlines()
                        self.assertIn("arg=tests", lines)
                        self.assertIn("arg=--run-slow", lines)
                        self.assertIn("arg=--strict-test-levels", lines)
                        marker_index = max(i for i, line in enumerate(lines) if line == "arg=-m")
                        expressions.append(lines[marker_index + 1][4:])
            for test_level in range(1, 6):
                for platform in platforms:
                    marks = {f"l{test_level}", *platform}
                    selected = any(
                        Expression.compile(expr).evaluate(marks.__contains__)
                        for expr in expressions
                    )
                    cpu_only = platform == {"cpu_test"}
                    expected = (
                        test_level in (1, 2)
                        if phase == "ready"
                        else (test_level == 3 or (test_level == 1 and cpu_only))
                    )
                    with self.subTest(phase=phase, marks=marks):
                        self.assertEqual(selected, expected)

    def test_platform_selection_combines_levels_and_device_markers(self) -> None:
        sample = self.directory / "test_matrix.py"
        sample.write_text(
            "import pytest\n"
            "@pytest.mark.l1\n@pytest.mark.cpu_test\ndef test_cpu(): pass\n"
            "@pytest.mark.l1\n@pytest.mark.gpu_test\ndef test_gpu(): pass\n"
            "@pytest.mark.l1\n@pytest.mark.npu_test\ndef test_npu(): pass\n"
            "@pytest.mark.l1\n@pytest.mark.gpu_test\n@pytest.mark.npu_test\ndef test_both(): pass\n"
            "@pytest.mark.l1\n@pytest.mark.cpu_test\n@pytest.mark.gpu_test\ndef test_cpu_gpu(): pass\n"
            "@pytest.mark.l2\n@pytest.mark.cpu_test\ndef test_l2_cpu(): pass\n"
            "@pytest.mark.l2\n@pytest.mark.gpu_test\ndef test_l2_gpu(): pass\n"
            "@pytest.mark.l2\n@pytest.mark.npu_test\ndef test_l2_npu(): pass\n"
            "@pytest.mark.l3\n@pytest.mark.cpu_test\ndef test_l3(): pass\n"
            "@pytest.mark.l3\n@pytest.mark.gpu_test\ndef test_l3_gpu(): pass\n"
            "@pytest.mark.l3\n@pytest.mark.gpu_test\n@pytest.mark.npu_test\ndef test_l3_both(): pass\n"
            "@pytest.mark.l4\n@pytest.mark.gpu_test\ndef test_l4_gpu(): pass\n",
            encoding="utf-8",
        )
        expected = {
            ("l1", "cpu", False): {"test_cpu"},
            ("l1", "cuda", False): {"test_cpu", "test_gpu", "test_both", "test_cpu_gpu"},
            ("l1", "npu", False): {"test_cpu", "test_npu", "test_both"},
            ("pr-device", "cpu", False): {"test_l2_cpu"},
            ("pr-device", "cuda", False): {
                "test_gpu",
                "test_both",
                "test_cpu_gpu",
                "test_l2_cpu",
                "test_l2_gpu",
            },
            ("pr-device", "npu", False): {"test_npu", "test_both", "test_l2_cpu", "test_l2_npu"},
            ("pr-device", "cuda", True): {"test_gpu", "test_both", "test_cpu_gpu", "test_l2_cpu"},
            ("pr-device", "npu", True): {"test_npu", "test_both", "test_l2_cpu", "test_l2_npu"},
            ("l3", "cuda", True): {"test_l3"},
            ("l3", "npu", True): {"test_l3", "test_l3_both"},
            ("l4", "cuda", True): {"test_l4_gpu"},
        }
        for (level, device, exclude_gpu), names in expected.items():
            with self.subTest(level=level, device=device, exclude_gpu=exclude_gpu):
                (self.directory / "calls.log").unlink(missing_ok=True)
                result, calls = self.run_level(
                    level, smoke=False, device=device, exclude_migrated_gpu=exclude_gpu
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                args = calls[0].splitlines()
                marker_index = max(i for i, arg in enumerate(args) if arg == "arg=-m")
                expression = args[marker_index + 1].removeprefix("arg=")
                env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
                env.pop("PYTEST_ADDOPTS", None)
                collected = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "--collect-only",
                        "-q",
                        "-m",
                        expression,
                        str(sample),
                    ],
                    cwd=self.directory,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(collected.returncode, 0, collected.stdout + collected.stderr)
                selected = {
                    line.rsplit("::", 1)[-1]
                    for line in collected.stdout.splitlines()
                    if "::test_" in line
                }
                self.assertEqual(selected, names)

        (self.directory / "calls.log").unlink(missing_ok=True)
        result, calls = self.run_level("l1", smoke=False, device="unknown")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])

    def test_failure_and_empty_selection_handling(self) -> None:
        for main, isolated, expected, call_count in [
            (1, 0, 1, 1),
            (0, 1, 1, 2),
            (5, 0, 0, 2),
            (0, 5, 0, 2),
            (5, 5, 5, 2),
        ]:
            with self.subTest(main=main, isolated=isolated):
                log = self.directory / "calls.log"
                log.unlink(missing_ok=True)
                result, calls = self.run_level("l4", smoke=True, main=main, inproc=isolated)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(len(calls), call_count)

    def test_empty_l5_remains_a_warning(self) -> None:
        result, calls = self.run_level("l5", smoke=True, main=5, inproc=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No L5 tests are currently collected", result.stdout)
        self.assertEqual(len(calls), 2)

    def test_collection_partitions_all_parameters_without_duplicates(self) -> None:
        source = [
            SimpleNamespace(nodeid=f"{INPROC}::test_example[one]"),
            SimpleNamespace(nodeid=f"{INPROC}::test_example[two]"),
            SimpleNamespace(nodeid="tests/test_other.py::test_example"),
        ]
        outputs = []
        for selection in ("exclude", "include"):
            items = source.copy()
            removed: list[SimpleNamespace] = []
            config = SimpleNamespace(
                hook=SimpleNamespace(
                    pytest_deselected=lambda *, items, removed=removed: removed.extend(items)
                )
            )
            with patch.dict(os.environ, VLLM_GR_INPROC_TEST_SELECTION=selection):
                pytest_collection_modifyitems(config, items)
            self.assertEqual(len(items) + len(removed), len(source))
            outputs.append(items)
        self.assertEqual(outputs[0], source[2:])
        self.assertEqual(outputs[1], source[:2])


if __name__ == "__main__":
    unittest.main()
