# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin CUDA provider with lazy hardware-specific imports."""

from dataclasses import dataclass
from importlib import import_module
from typing import cast

from .protocol import BACKEND_API_VERSION, BeamAttentionHooks, ModelRunnerPatchHooks


@dataclass(frozen=True)
class CUDABackendProvider:
    api_version: int = BACKEND_API_VERSION
    device_type: str = "cuda"

    def configure_engine(self, config: object) -> None:
        del config

    def check_environment(self) -> dict[str, str]:
        from vllm.platforms import current_platform

        from vllm_gr._env_check import check_runtime_versions

        if str(current_platform.device_type) != "cuda":
            raise RuntimeError("CUDA provider requires the CUDA platform")

        report = check_runtime_versions()
        report["platform"] = "cuda"
        return report

    def release_accelerator_memory(self) -> None:
        import torch

        from .memory import cleanup_torch_backend

        cleanup_torch_backend(torch.cuda, collect_ipc=True)

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(
            ModelRunnerPatchHooks,
            import_module("vllm_gr.v1.worker.gpu_model_runner_patch"),
        )

    def get_attention_hooks(self) -> BeamAttentionHooks:
        return cast(
            BeamAttentionHooks,
            import_module("vllm_gr.v1.attention.backends.beam_attn_gpu"),
        )


CUDA_BACKEND = CUDABackendProvider()
