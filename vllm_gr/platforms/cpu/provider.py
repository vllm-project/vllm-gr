# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Experimental eager CPU provider.

Reuses the eager Ascend V1 stage runner (plain PyTorch) on vLLM's
CPUModelRunner, with a reference PyTorch Beam Attention. Temporary: to be
moved onto the #851 platform interfaces once they land.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, cast

from vllm_gr.config.backend_plan import BackendPlan, BackendRequest, PlanData

from ..compat import SUPPORTED_VLLM_VERSION
from ..config import BackendConfigService
from ..identity import PlatformIdentity
from ..interface import PROVIDER_API_VERSION, BeamAttentionExecution, ModelRunnerInstaller
from ..services import BuiltinPlanSupport, require_provider_plan

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import ConstrainedTopKExecution

_RUNNER = "vllm_gr.v1.worker.cpu.cpu_model_runner_patch"
_ATTENTION = "vllm_gr.v1.attention.backends.beam_attn_cpu"


class CPUConfigService(BackendConfigService):
    default_worker_decision = True


def _keep_requested_async_scheduling() -> int:
    """vLLM's CPU platform turns async scheduling off as "not required".

    Beam V1 Batch Sessions need it and the CPU executors support it, so an
    explicit request survives every (parent and EngineCore) config update.
    """
    from vllm.platforms.cpu import CpuPlatform

    original = CpuPlatform.check_and_update_config
    if getattr(original, "_vllm_gr_async", False):
        return 0

    def check_and_update_config(cls: Any, vllm_config: Any) -> None:
        requested = vllm_config.scheduler_config.async_scheduling
        original.__func__(cls, vllm_config)
        if requested:
            vllm_config.scheduler_config.async_scheduling = True

    check_and_update_config._vllm_gr_async = True  # type: ignore[attr-defined]
    CpuPlatform.check_and_update_config = classmethod(check_and_update_config)  # type: ignore[assignment]
    return 1


@dataclass(frozen=True)
class CPUBackendProvider(BuiltinPlanSupport):
    backend_id: str = "cpu"
    device_type: str = "cpu"
    api_version: ClassVar[int] = PROVIDER_API_VERSION
    gr_version_range: ClassVar[str | None] = ">=0.30.0,<0.31.0"
    vllm_version_range: ClassVar[str | None] = f"=={SUPPORTED_VLLM_VERSION}"
    vendor_version_range: ClassVar[None] = None
    vendor_distribution: ClassVar[None] = None

    def matches(self, identity: PlatformIdentity) -> bool:
        return identity.platform_enum == "CPU" and identity.device_type == "cpu"

    matches_legacy_identity = matches

    def resolve(self, request: BackendRequest) -> BackendPlan:
        if request.options.to_dict():
            raise ValueError("CPU currently accepts no additional backend options")
        gr = request.config.to_dict()
        constraint = "canonical" if request.has_constraint_table else "none"
        return BackendPlan(
            self.backend_id,
            self.api_version,
            request,
            PlanData(
                {
                    "runner": _RUNNER,
                    "attention": _ATTENTION,
                    "prefill_attention": "cpu.torch",
                    "decode_attention": "cpu.torch",
                    "requires_attention": gr["attention_backend"] == "CUSTOM",
                    "constraint": constraint,
                    "constraint_implementation": "torch" if constraint != "none" else None,
                    "stage": "cpu.v1",
                    "graph": "eager",
                }
            ),
        )

    def create_config_service(self) -> BackendConfigService:
        return CPUConfigService()

    def check_environment(self) -> dict[str, str]:
        from ..compat import check_vllm_runtime

        report = check_vllm_runtime()
        report["platform"] = "cpu"
        return report

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return _keep_requested_async_scheduling()

    def get_model_runner_hooks(self) -> ModelRunnerInstaller:
        return cast(ModelRunnerInstaller, import_module(_RUNNER))

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(BeamAttentionExecution, import_module(_ATTENTION))

    def create_attention_executor(
        self, *, plan: BackendPlan, device: object
    ) -> BeamAttentionExecution:
        require_provider_plan(self, plan)
        return self.get_attention_execution()

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
        plan: BackendPlan | None = None,
    ) -> ConstrainedTopKExecution:
        from vllm_gr.constraints.execution import TorchConstrainedTopKExecution

        if plan is not None:
            require_provider_plan(self, plan)
        return TorchConstrainedTopKExecution(device=device)


CPU_BACKEND = CPUBackendProvider()
