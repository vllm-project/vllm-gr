# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin CUDA provider with lazy hardware-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from ..identity import PlatformIdentity, platform_identity
from ..protocol import (
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    ModelRunnerPatchHooks,
)

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import ConstrainedTopKExecution


@dataclass(frozen=True)
class CUDABackendProvider:
    backend_id: str = "cuda"
    #: The version-1 identity preserved for diagnostics of legacy providers.
    device_type: str = "cuda"
    # Semantic ownership includes hooks still installed by shared dispatch.
    patch_ids: ClassVar[tuple[str, ...]] = ("GR-PATCH-011",)
    api_version: ClassVar[int] = PROVIDER_API_VERSION
    gr_version_range: ClassVar[str | None] = ">=0.30.0,<0.31.0"
    vllm_version_range: ClassVar[str | None] = "==0.30.0"
    vendor_version_range: ClassVar[None] = None
    vendor_distribution: ClassVar[None] = None

    def matches(self, identity: PlatformIdentity) -> bool:
        """Match the native NVIDIA CUDA platform, not every ``cuda`` placement.

        AMD ROCm also reports ``device_type == "cuda"``; it is a distinct
        upstream enum value and needs its own provider.
        """
        return identity.platform_enum == "CUDA" and identity.device_type == "cuda"

    matches_legacy_identity = matches

    def check_environment(self) -> dict[str, str]:
        from vllm.platforms import current_platform

        from vllm_gr._env_check import check_runtime_versions

        if not self.matches(platform_identity(current_platform)):
            raise RuntimeError("CUDA provider requires the native CUDA platform")

        report = check_runtime_versions()
        report["platform"] = "cuda"
        return report

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(
            ModelRunnerPatchHooks,
            import_module("vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch"),
        )

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_gpu"),
        )

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "ConstrainedTopKExecution | None":
        """Create the production CUDA Constraint Table execution adapter."""
        from vllm_gr.constraints.execution import CUDAConstrainedTopKExecution

        return CUDAConstrainedTopKExecution(device=device)


CUDA_BACKEND = CUDABackendProvider()
