# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess checks for backend infrastructure import isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_platform_package_does_not_load_provider_or_hardware_modules() -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = """
import sys

baseline_modules = set(sys.modules)

import vllm_gr.platforms

for module_name in (
    "torch",
    "vllm",
    "vllm_gr.platforms.cuda",
    "vllm_gr.v1.worker.gpu_model_runner_patch",
    "vllm_gr.v1.attention.backends.beam_attn_gpu",
):
    assert module_name not in set(sys.modules) - baseline_modules, module_name
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cuda_runtime_orchestration_uses_cuda_provider() -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = """
import torch
import vllm.logger

from vllm_gr.platforms import activate_backend, get_active_backend, register_backend
from vllm_gr.platforms.cuda import CUDA_BACKEND

register_backend(CUDA_BACKEND, source="test:cuda")
activate_backend("cuda")

import vllm_gr.patch
import vllm_gr.v1.worker.model_runner_patch
assert get_active_backend().device_type == "cuda"
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

from vllm_gr.platforms import activate_backend, get_active_backend, register_backend
from vllm_gr.platforms.cuda import CUDA_BACKEND

register_backend(CUDA_BACKEND, source="test:cuda")
activate_backend("cuda")

child_code = '''
import sys
from types import ModuleType, SimpleNamespace

from vllm_gr.platforms import (
    BackendAlreadyActiveError,
    BackendNotActiveError,
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

class CUDAEntryPoint:
    name = "cuda"
    value = "vllm_gr.platforms.cuda:CUDA_BACKEND"
    dist = None

    @staticmethod
    def load():
        from vllm_gr.platforms.cuda import CUDA_BACKEND
        return CUDA_BACKEND

backend_loader.entry_points = lambda *, group: (CUDAEntryPoint(),)
assert activate_current_backend().device_type == "cuda"
assert get_active_backend().device_type == "cuda"

from vllm_gr.platforms import register_backend

other_backend = SimpleNamespace(
    api_version=1,
    device_type="other",
    configure_engine=lambda config: None,
    check_environment=lambda: {},
    release_accelerator_memory=lambda: None,
    apply_runtime_patches=lambda *, force=False: 0,
    get_model_runner_hooks=lambda: object(),
    get_attention_hooks=lambda: object(),
)
register_backend(other_backend, source="test:other")
try:
    activate_backend("other")
except BackendAlreadyActiveError:
    pass
else:
    raise AssertionError("child switched active backend")
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
