# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Intel XPU provider for fixed-width, eager, persistent Beam execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from vllm_gr.config.backend_plan import BackendPlan, BackendRequest, PlanData
from vllm_gr.platforms.compat import SUPPORTED_VLLM_VERSION
from vllm_gr.platforms.config import BackendConfigService
from vllm_gr.platforms.identity import PlatformIdentity, current_platform_identity
from vllm_gr.platforms.interface import PROVIDER_API_VERSION
from vllm_gr.platforms.services import BuiltinPlanSupport, require_provider_plan

RUNNER_ID = "vllm_gr.v1.worker.xpu.runner"


class XPUConfigService(BackendConfigService):
    default_worker_decision = True

    def validate_engine(self, config: Any, gr: Any) -> None:
        if gr.beam_max_width is None:
            return
        if (
            not config.model_config.enforce_eager
            or gr.beam_graph_enabled
            or gr.deployment.prefill_graph_enabled
        ):
            raise ValueError("XPU Beam requires eager execution with both GR graph modes disabled")
        if gr.attention_backend is None or gr.attention_backend.value != "CUSTOM":
            raise ValueError("XPU Beam requires attention_backend='CUSTOM'")
        if gr.constraint is not None and gr.constraint.mode.value != "none":
            raise ValueError("XPU Beam does not yet support canonical constraint tables")
        if gr.beam_worker_decision is not True:
            raise ValueError("XPU Beam requires gr.worker_decision=True")
        for widths in (gr.deployment.beam_width, gr.deployment.candidate_width):
            if widths is not None and len(set(widths)) != 1:
                raise ValueError("XPU Beam currently requires fixed width schedules")
        if config.cache_config.cache_dtype not in ("auto", "bfloat16", "float16"):
            raise ValueError("XPU Beam requires unquantized KV cache")
        from vllm_gr.config.execution import validate_gr_execution_runtime

        validate_gr_execution_runtime(config)
        from vllm import envs

        if not envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("XPU Beam requires VLLM_USE_V2_MODEL_RUNNER=1")

    @staticmethod
    def validate_request(request: Any) -> None:
        if len(request.items) != 1:
            raise ValueError("XPU Beam currently supports one prompt per session")
        params = request.beam_params
        if params.dynamic_beam_widths is not None or params.per_parent_beam_widths is not None:
            raise ValueError("XPU Beam currently requires a fixed Beam width")


@dataclass(frozen=True)
class XPUBackendProvider(BuiltinPlanSupport):
    backend_id: str = "xpu"
    device_type: str = "xpu"
    api_version: ClassVar[int] = PROVIDER_API_VERSION
    patch_ids: ClassVar[tuple[str, ...]] = ("GR-PATCH-023",)
    gr_version_range: ClassVar[str] = ">=0.30.0.dev0,<0.31.0"
    vllm_version_range: ClassVar[str] = f"=={SUPPORTED_VLLM_VERSION}"
    vendor_version_range: ClassVar[None] = None
    vendor_distribution: ClassVar[None] = None
    adapter_revision = "xpu-native-v2-eager-v1"

    def matches(self, identity: PlatformIdentity) -> bool:
        return identity.platform_enum == "XPU" and identity.device_type == "xpu"

    def create_config_service(self) -> BackendConfigService:
        return XPUConfigService()

    def check_environment(self) -> dict[str, str]:
        import torch

        from vllm_gr.platforms.compat import require_runtime_version

        if not self.matches(current_platform_identity()) or not torch.xpu.is_available():
            raise RuntimeError("XPU provider requires an available native Intel XPU runtime")
        return {
            "vllm": require_runtime_version("vllm", SUPPORTED_VLLM_VERSION),
            "torch": torch.__version__,
            "platform": "xpu",
        }

    def resolve(self, request: BackendRequest) -> BackendPlan:
        if request.options.to_dict():
            raise ValueError("XPU accepts no additional backend options")
        if request.has_constraint_table:
            raise ValueError("XPU canonical constraint tables are not implemented")
        return BackendPlan(
            self.backend_id,
            self.api_version,
            request,
            PlanData(
                {
                    "runner": RUNNER_ID,
                    "attention": "vllm_gr.v1.attention.backends.beam_attn_xpu",
                    "requires_attention": request.uses_custom_attention,
                    "constraint": "none",
                    "stage": "xpu.v2",
                    "graph": "eager",
                }
            ),
        )

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> Any:
        from vllm_gr.v1.worker.xpu import runner

        return runner

    def get_attention_backend(self) -> type:
        from vllm_gr.v1.attention.backends.beam_attn_xpu import XPUBeamAttentionBackend

        return XPUBeamAttentionBackend

    def get_attention_execution(self) -> Any:
        from vllm_gr.v1.attention.backends import beam_attn_xpu

        return beam_attn_xpu

    def create_attention_executor(self, *, plan: BackendPlan, device: object) -> Any:
        require_provider_plan(self, plan)
        return self.get_attention_execution()

    def create_constrained_topk_execution(self, *, device: Any = None, plan: Any = None) -> None:
        raise NotImplementedError("XPU canonical constrained TopK is not implemented")


XPU_BACKEND = XPUBackendProvider()
