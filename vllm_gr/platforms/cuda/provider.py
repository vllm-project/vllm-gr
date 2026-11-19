# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin CUDA provider with lazy hardware-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..constraint_plan import constraint_plan_choices
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
        from vllm_gr.v1.attention.backends.beam_dense_prefill import (
            admitted_dense_prefill_mode,
        )

        custom_attention = gr["attention_backend"] == "CUSTOM"
        dense_prefill = admitted_dense_prefill_mode(
            native.get("dense_prefill_request") if custom_attention else None,
            batch_invariant=bool(native.get("dense_prefill_batch_invariant")),
            kv_transfer=bool(native.get("has_kv_transfer")),
            prefix_caching=bool(native.get("cache_config", {}).get("enable_prefix_caching")),
            chunked_prefill=bool(native.get("scheduler_config", {}).get("enable_chunked_prefill")),
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
                    "dense_prefill_mode": dense_prefill,
                    "decode_attention": "cuda.cascade",
                    "prefill_flash_attn_version": gr["prefill_flash_attn_version"],
                    "kv_layout": native.get("kv_cache_layout"),
                    "requires_attention": custom_attention,
                    **constraint_plan_choices(gr, legacy_id="cuda"),
                    "operator_policy": "cuda.onerec",
                    "stage": "cuda.v2" if native["use_v2_model_runner"] else "cuda.v1",
                    "graph": "capture-with-eager-fallback" if gr["beam_graph_enabled"] else "eager",
                }
            ),
        )

    def native_attention(self, config: Any) -> Any:
        from os import environ

        from vllm_gr.config import GRConfig

        from .attention import CudaAttention

        mode = GRConfig.from_vllm_config(config).npu_decode_execution
        if mode is not None and mode.value == "xllm":
            raise ValueError("CUDA does not support npu_decode_execution='xllm'")
        return CudaAttention(kv_layout=environ.get("VLLM_KV_CACHE_LAYOUT"))

    def attention_admission(self, *, plan: BackendPlan) -> Any:
        return self._attention_for_plan(plan)

    def _attention_for_plan(self, plan: BackendPlan) -> Any:
        from .attention import CudaAttention

        require_provider_plan(self, plan)
        return CudaAttention(plan=plan)

    def create_attention_executor(self, *, plan: BackendPlan, device: object) -> Any:
        executor = self._attention_for_plan(plan)
        if not executor.execution_module().is_available():
            raise RuntimeError("Admitted Cuda attention implementation is unavailable")
        return executor

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

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None:
        super().validate_worker(plan=plan, native_runner=native_runner)
        if (
            plan.choices.to_dict().get("constraint") == "canonical"
            and plan.request.config.to_dict()["beam_execution_mode"] == "v1"
        ):
            from vllm_gr.constraints.backend import validate_device_stage_backend

            validate_device_stage_backend(
                native_runner.constraint_backend,
                native_runner.device,
                native_runner.model_config.get_vocab_size(),
            )

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
    ) -> ConstrainedTopKExecution | None:
        """Create the production CUDA Constraint Table execution adapter."""
        from vllm_gr.constraints.cuda_execution import CUDAConstrainedTopKExecution

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
