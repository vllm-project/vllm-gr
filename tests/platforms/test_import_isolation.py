# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess checks for backend infrastructure import isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_platform_package_and_builtins_do_not_load_hardware_modules() -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = """
import sys

import vllm_gr.platforms

for module_name in (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "vllm_gr.v1.worker.gpu_model_runner_patch",
    "vllm_gr.v1.worker.npu_model_runner_patch",
    "vllm_gr.v1.attention.backends.beam_attn_gpu",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
    "vllm_gr.v1.ascend_compat",
):
    assert module_name not in sys.modules, module_name

from vllm_gr.platforms.builtins import register_builtin_backends

register_builtin_backends()

for module_name in (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "vllm_gr.v1.worker.gpu_model_runner_patch",
    "vllm_gr.v1.worker.npu_model_runner_patch",
    "vllm_gr.v1.attention.backends.beam_attn_gpu",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
    "vllm_gr.v1.ascend_compat",
):
    assert module_name not in sys.modules, module_name
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cuda_runtime_orchestration_does_not_load_npu_modules() -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = """
import sys

from vllm_gr.platforms import activate_backend
from vllm_gr.platforms.builtins import register_builtin_backends

register_builtin_backends()
activate_backend("cuda")

import vllm_gr.patch
import vllm_gr.engine.arg_utils_gr
import vllm_gr.entrypoints.cli.check_env
import vllm_gr.entrypoints.gr
import vllm_gr.v1.worker.model_runner_patch
from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

assert BeamAttentionBackend.get_builder_cls().__name__ == "CUDABeamAttentionMetadataBuilder"

for module_name in (
    "torch_npu",
    "vllm_ascend",
    "vllm_gr.v1.ascend_compat",
    "vllm_gr.v1.worker.npu_model_runner_patch",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
):
    assert module_name not in sys.modules, module_name
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_backend_activation_is_process_local() -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = """
import subprocess
import sys

from vllm_gr.platforms import activate_backend, get_active_backend
from vllm_gr.platforms.builtins import register_builtin_backends

register_builtin_backends()
activate_backend("cuda")

child_code = '''
import sys
from types import ModuleType, SimpleNamespace

from vllm_gr.platforms import (
    BackendAlreadyActiveError,
    BackendNotActiveError,
    UnsupportedBackendError,
    activate_backend,
    activate_current_backend,
    get_active_backend,
)

try:
    get_active_backend()
except BackendNotActiveError:
    pass
else:
    raise AssertionError("parent backend leaked into child")

vllm_module = ModuleType("vllm")
platforms_module = ModuleType("vllm.platforms")
platforms_module.current_platform = SimpleNamespace(device_type="cuda")
vllm_module.platforms = platforms_module
sys.modules["vllm"] = vllm_module
sys.modules["vllm.platforms"] = platforms_module

import vllm_gr.platforms.loader as backend_loader

backend_loader._PLUGINS_LOADED = True
assert activate_current_backend().device_type == "cuda"
assert get_active_backend().device_type == "cuda"

try:
    activate_backend("npu")
except UnsupportedBackendError:
    pass
else:
    raise AssertionError("missing NPU Provider fell back or activated")
'''

subprocess.run(
    [sys.executable, "-c", child_code],
    check=True,
    cwd=".",
)
assert get_active_backend().device_type == "cuda"
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
