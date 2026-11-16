# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Subprocess checks for backend infrastructure import isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


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


@pytest.mark.l1  # type: ignore[misc]
def test_cuda_plugin_initialization_does_not_import_ascend() -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = """
import sys

# On Ascend images, importing torch autoloads the torch_npu device backend
# (torch >= 2.10 entry-point auto-loading) and importing vllm activates the
# vendor ascend platform plugin during platform detection. Those vendor
# imports are properties of the machine image, not of vllm-gr's backend
# selection, so they cannot be forbidden here. The enforceable isolation
# contract is that vllm-gr's own Ascend integration stays unloaded while the
# CUDA provider initializes.
blocked = ("vllm_gr.v1.ascend_compat", "vllm_gr.npu_spawn_patch")

def reject_ascend(event, args):
    # Metadata probes (find_spec) are allowed; importing the GR Ascend
    # integration is not.
    if event == "import" and args[0] in blocked:
        raise AssertionError(f"unselected backend imported: {args[0]}")

sys.addaudithook(reject_ascend)
import vllm.platforms
from vllm.platforms.interface import Platform

# Synthetic device identity: exercise host initialization, not CUDA execution.
platform = Platform()
platform.device_type = "cuda"
vllm.platforms.current_platform = platform
from vllm_gr.platforms import loader
loader.entry_points = lambda **kwargs: ()

from vllm_gr.plugin import initialize_runtime
first = initialize_runtime()
assert first["patch_count"] > 0, first
assert set(first["patches"].values()) == {"applied"}, first
second = initialize_runtime()
assert second["patch_count"] == 0, second
assert set(second["patches"].values()) == {"already_applied"}, second
assert not any(name in sys.modules for name in blocked)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


pytestmark = pytest.mark.cpu_test
