# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin CUDA provider with lazy hardware-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from ..identity import PlatformIdentity, platform_identity
from ..interface import (
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    ModelRunnerInstaller,
)
from ..plan import BackendPlan, BackendRequest, PlanData
from ..services import BuiltinPlanSupport, require_provider_plan
from .runner import runner_id

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import ConstrainedTopKExecution


@dataclass(frozen=True)
class CUDABackendProvider(BuiltinPlanSupport):
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

    def resolve(self, request: BackendRequest) -> BackendPlan:
        """Admit the existing CUDA paths without importing a device runtime."""
        if request.options.to_dict():
            raise ValueError("CUDA currently accepts no additional backend options")
        gr = request.config.to_dict()
        native = request.native.to_dict()
        if gr["npu_decode_execution"] == "xllm":
            raise ValueError("CUDA does not support npu_decode_execution='xllm'")
        table = gr["constraint_table"] or {}
        constraint = (
            "canonical"
            if table.get("enabled")
            else ("legacy" if gr["constraint_backend"] == "constraint_table" else "none")
        )
        return BackendPlan(
            self.backend_id,
            self.api_version,
            request,
            PlanData(
                {
                    "runner": runner_id(native["use_v2_model_runner"]),
                    "attention": "vllm_gr.v1.attention.backends.beam_attn_gpu",
                    "prefill_attention": "cuda.cascade",
                    "decode_attention": "cuda.cascade",
                    "prefill_flash_attn_version": gr["prefill_flash_attn_version"],
                    "requires_attention": gr["attention_backend"] == "CUSTOM",
                    "constraint": constraint,
                    "constraint_implementation": "cuda" if constraint != "none" else None,
                    "stage": "cuda.v2" if native["use_v2_model_runner"] else "cuda.v1",
                    "graph": "capture-with-eager-fallback" if gr["beam_graph_enabled"] else "eager",
                }
            ),
        )

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

    def get_model_runner_hooks(self) -> ModelRunnerInstaller:
        from . import runner

        return runner

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_gpu"),
        )

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
        plan: BackendPlan | None = None,
    ) -> "ConstrainedTopKExecution | None":
        """Create the production CUDA Constraint Table execution adapter."""
        from vllm_gr.constraints.execution import CUDAConstrainedTopKExecution

        if plan is not None:
            require_provider_plan(self, plan)
        return CUDAConstrainedTopKExecution(device=device)


def install_worker_utilities() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_gr.v1.worker.model_runner_patch import install_worker_utility_hooks

    install_worker_utility_hooks(Worker)


def install_worker_hooks() -> None:
    """Both CUDA runner generations use the same Worker and NumPy staging."""
    from vllm_gr.v1.worker.cuda.numpy_staging import install_numpy_staging

    install_worker_utilities()
    install_numpy_staging()


CUDA_BACKEND = CUDABackendProvider()
