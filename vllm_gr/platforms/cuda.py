# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin CUDA provider with lazy hardware-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import cast

from .protocol import BeamAttentionExecution, ModelRunnerPatchHooks


@dataclass(frozen=True)
class CUDABackendProvider:
    device_type: str = "cuda"

    def check_environment(self) -> dict[str, str]:
        from vllm.platforms import current_platform

        from vllm_gr._env_check import check_runtime_versions

        if str(current_platform.device_type) != "cuda":
            raise RuntimeError("CUDA provider requires the CUDA platform")

        report = check_runtime_versions()
        report["platform"] = "cuda"
        return report

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(
            ModelRunnerPatchHooks,
            import_module("vllm_gr.v1.worker.gpu_model_runner_patch"),
        )

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_gpu"),
        )

CUDA_BACKEND = CUDABackendProvider()
