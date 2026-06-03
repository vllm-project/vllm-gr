# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_tests.py"


def _load_module(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestRunTestsScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

    def test_all_with_auto_platform_prefers_npu_lane(self) -> None:
        args = self._make_args(all=True, auto_platform=True)

        with mock.patch.object(
            self.module,
            "detect_runtime_backend",
            return_value="npu",
        ):
            suites, backend = self.module.determine_requested_suites(args)

        self.assertEqual(backend, "npu")
        self.assertEqual(suites, ["unit", "npu-smoke"])

    def test_all_with_auto_platform_falls_back_to_gpu_lane(self) -> None:
        args = self._make_args(all=True, auto_platform=True)

        with mock.patch.object(
            self.module,
            "detect_runtime_backend",
            return_value="gpu",
        ):
            suites, backend = self.module.determine_requested_suites(args)

        self.assertEqual(backend, "gpu")
        self.assertEqual(suites, ["unit", "gpu-smoke"])

    def test_all_with_auto_platform_skips_platform_smoke_on_cpu(self) -> None:
        args = self._make_args(all=True, auto_platform=True)

        with mock.patch.object(
            self.module,
            "detect_runtime_backend",
            return_value="cpu",
        ):
            suites, backend = self.module.determine_requested_suites(args)

        self.assertEqual(backend, "cpu")
        self.assertEqual(suites, ["unit"])

    def test_explicit_smoke_suite_is_not_overridden_by_auto_platform(self) -> None:
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

    def test_run_examples_propagates_expected_device(self) -> None:
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
                ("fuxi_simple_inference.py", ["--expect-device", "npu"]),
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
