# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_tests.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestRunTestsScript(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = _load_module("run_tests_script_test_module", MODULE_PATH)

    def _make_args(self, **overrides) -> argparse.Namespace:
        values = {
            "all": False,
            "auto_platform": False,
            "examples_only": False,
            "suite": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_all_with_auto_platform_prefers_npu_lane(self):
        args = self._make_args(all=True, auto_platform=True)

        with mock.patch.object(
            self.module,
            "detect_runtime_backend",
            return_value="npu",
        ):
            suites, backend = self.module.determine_requested_suites(args)

        self.assertEqual(backend, "npu")
        self.assertEqual(suites, ["unit", "npu-smoke"])

    def test_all_with_auto_platform_falls_back_to_gpu_lane(self):
        args = self._make_args(all=True, auto_platform=True)

        with mock.patch.object(
            self.module,
            "detect_runtime_backend",
            return_value="gpu",
        ):
            suites, backend = self.module.determine_requested_suites(args)

        self.assertEqual(backend, "gpu")
        self.assertEqual(suites, ["unit", "gpu-smoke"])

    def test_all_with_auto_platform_skips_platform_smoke_on_cpu(self):
        args = self._make_args(all=True, auto_platform=True)

        with mock.patch.object(
            self.module,
            "detect_runtime_backend",
            return_value="cpu",
        ):
            suites, backend = self.module.determine_requested_suites(args)

        self.assertEqual(backend, "cpu")
        self.assertEqual(suites, ["unit"])

    def test_explicit_smoke_suite_is_not_overridden_by_auto_platform(self):
        args = self._make_args(
            suite=["gpu-smoke"],
            auto_platform=True,
        )

        with mock.patch.object(
            self.module,
            "detect_runtime_backend",
            return_value="npu",
        ):
            suites, backend = self.module.determine_requested_suites(args)

        self.assertEqual(backend, "npu")
        self.assertEqual(suites, ["gpu-smoke"])

    def test_run_examples_propagates_expected_device(self):
        recorded_calls: list[tuple[str, list[str] | None]] = []

        def _fake_run_example(
            script: str,
            extra_args: list[str] | None = None,
        ) -> bool:
            recorded_calls.append((script, extra_args))
            return True

        with mock.patch.object(
            self.module,
            "run_example",
            side_effect=_fake_run_example,
        ):
            self.assertTrue(
                self.module.run_examples(
                    datasets=["debug", "streaming"],
                    expected_device="npu",
                )
            )

        self.assertEqual(
            recorded_calls,
            [
                ("simple_inference.py", ["--expect-device", "npu"]),
                (
                    "dataset_inference.py",
                    ["--expect-device", "npu", "debug"],
                ),
                (
                    "dataset_inference.py",
                    ["--expect-device", "npu", "streaming"],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
