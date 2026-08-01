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
