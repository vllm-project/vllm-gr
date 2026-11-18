# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin CUDA provider with lazy hardware-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from .identity import PlatformIdentity, platform_identity
from .protocol import (
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    ModelRunnerPatchHooks,
)

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import ConstrainedTopKExecution

    from .services import (
        BackendConfigService,
        BeamAttentionService,
        BeamMetadataProfile,
        StageExecutionService,
    )


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
            import_module("vllm_gr.v1.worker.gpu_model_runner_patch"),
        )

    def select_runner_patch_module_name(self, *, prefer_v2: bool = False) -> str:
        """Return the CUDA runner patch module, honoring an explicit V2 opt-in.

        CUDA is the only backend that implements ModelRunner V2, so the V2 flag
        selects its V2 module here rather than in shared dispatch.
        """
        if prefer_v2:
            return "vllm_gr.v1.worker.gpu_model_runner_v2_patch"
        return "vllm_gr.v1.worker.gpu_model_runner_patch"

    def supports_v2_model_runner(self) -> bool:
        """CUDA ships a ModelRunner V2 patch module in this release."""
        return True

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_gpu"),
        )

    def create_beam_attention_service(self) -> BeamAttentionService:
        """Return the CUDA Beam attention service.

        The service defers its own imports, so building it loads no CUDA module.
        """
        from .cuda_services import CudaBeamAttentionService

        return CudaBeamAttentionService()

    def create_config_service(self) -> BackendConfigService:
        """Return CUDA's configuration-time policy."""
        from .cuda_services import CudaConfigService

        return CudaConfigService()

    def create_beam_metadata_profile(self) -> BeamMetadataProfile:
        """Return CUDA's answers for shared Beam metadata construction."""
        from .cuda_services import CudaBeamMetadataProfile

        return CudaBeamMetadataProfile()

    def create_stage_execution_service(self) -> StageExecutionService:
        """Return the CUDA stage execution service.

        The service defers its own imports, so building it loads no CUDA module
        and this provider keeps its import-light contract.
        """
        from .cuda_services import CudaStageExecutionService

        return CudaStageExecutionService()

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "ConstrainedTopKExecution | None":
        """Create the production CUDA Constraint Table execution adapter."""
        from vllm_gr.constraints.execution import CUDAConstrainedTopKExecution

        return CUDAConstrainedTopKExecution(device=device)


CUDA_BACKEND = CUDABackendProvider()
