# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise GPU test collection without importing an accelerator runtime."""

import builtins
import os
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

GPU_TEST = Path(__file__).parent / "kernels/attention/test_cascade_beam_attn_gpu.py"


class _ModuleSkipped(Exception):
    def __init__(self, reason: str, *, allow_module_level: bool) -> None:
        super().__init__(reason)
        self.allow_module_level = allow_module_level


class TestCascadeBeamAttentionGPUCollection(unittest.TestCase):
    def setUp(self) -> None:
        self.dependency_imports: list[str] = []

    def _execute_module(self, *, cuda_available: bool, target_device: str | None) -> None:
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "vllm" or name.startswith(("vllm.", "vllm_gr.")):
                self.dependency_imports.append(name)
                raise ImportError("sentinel CUDA dependency failure")
            return original_import(name, *args, **kwargs)

        def skip_module(reason: str, *, allow_module_level: bool) -> None:
            raise _ModuleSkipped(reason, allow_module_level=allow_module_level)

        torch_stub = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda_available),
        )
        pytest_stub = SimpleNamespace(skip=skip_module)

        # Execute the actual test module. The import sentinel must never be
        # reached on CPU/NPU, but must remain a real failure on a CUDA runner.
        with (
            patch.dict(os.environ),
            patch.dict(sys.modules, {"pytest": pytest_stub, "torch": torch_stub}),
            patch("builtins.__import__", side_effect=guarded_import),
        ):
            if target_device is None:
                os.environ.pop("VLLM_GR_TARGET_DEVICE", None)
            else:
                os.environ["VLLM_GR_TARGET_DEVICE"] = target_device
            runpy.run_path(str(GPU_TEST), run_name="__gpu_collection_regression__")

    def test_cpu_skips_before_vllm_imports(self) -> None:
        with self.assertRaises(_ModuleSkipped) as skipped:
            self._execute_module(cuda_available=False, target_device=None)
        self.assertTrue(skipped.exception.allow_module_level)
        self.assertEqual(self.dependency_imports, [])

    def test_explicit_npu_target_skips_even_when_cuda_is_available(self) -> None:
        with self.assertRaises(_ModuleSkipped) as skipped:
            self._execute_module(cuda_available=True, target_device="npu")
        self.assertTrue(skipped.exception.allow_module_level)
        self.assertEqual(self.dependency_imports, [])

    def test_cuda_dependency_import_error_is_not_skipped(self) -> None:
        with self.assertRaisesRegex(ImportError, "sentinel CUDA dependency failure"):
            self._execute_module(cuda_available=True, target_device="cuda")
        self.assertEqual(self.dependency_imports, ["vllm.config"])


if __name__ == "__main__":
    unittest.main()


pytestmark = pytest.mark.cpu_test
