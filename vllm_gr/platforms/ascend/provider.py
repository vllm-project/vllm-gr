# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin Ascend provider with lazy NPU-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from vllm_gr.config.backend_plan import BackendPlan, BackendRequest, PlanData

from ..compat import SUPPORTED_VLLM_ASCEND_VERSION, SUPPORTED_VLLM_VERSION
from ..identity import PlatformIdentity
from ..interface import (
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    ModelRunnerInstaller,
)
from ..services import BuiltinPlanSupport, require_provider_plan

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

    from ..config import BackendConfigService

#: Qualified name of the vendor platform class this provider supports. The
#: check is a lineage lookup on the already-selected platform instance, so it
#: imports no vendor code and core keeps no platform table.
_ASCEND_PLATFORM_CLASS = "vllm_ascend.platform.NPUPlatform"


@dataclass(frozen=True)
class NPUBackendProvider(BuiltinPlanSupport):
    backend_id: str = "ascend"
    #: Tensor placement of the platform this provider implements.
    device_type: str = "npu"
    # Semantic ownership includes deferred hooks and existing shared dispatch.
    patch_ids: ClassVar[tuple[str, ...]] = (
        "GR-PATCH-002",
        "GR-PATCH-012",
        "GR-PATCH-016",
        "GR-PATCH-018",
        "GR-PATCH-019",
        "GR-PATCH-021",
    )
    api_version: ClassVar[int] = PROVIDER_API_VERSION
    gr_version_range: ClassVar[str | None] = ">=0.31.0,<0.32.0"
    vllm_version_range: ClassVar[str | None] = f"=={SUPPORTED_VLLM_VERSION}"
    vendor_version_range: ClassVar[str | None] = f"=={SUPPORTED_VLLM_ASCEND_VERSION}"
    vendor_distribution: ClassVar[str | None] = "vllm-ascend"

    def matches(self, identity: PlatformIdentity) -> bool:
        """Match the supported Ascend platform identity and ``npu`` placement.

        The vendor platform enum is ``OOT``, which a future out-of-tree provider
        could share, so the qualified class lineage is required as well.
        """
        return identity.device_type == "npu" and identity.has_lineage(_ASCEND_PLATFORM_CLASS)

    def resolve(self, request: BackendRequest) -> BackendPlan:
        """Keep Ascend's V1 runner and its existing auto-decode policy."""
        if request.options.to_dict():
            raise ValueError("Ascend currently accepts no additional backend options")
        gr = request.config.to_dict()
        constraint = "canonical" if request.has_constraint_table else "none"
        return BackendPlan(
            self.backend_id,
            self.api_version,
            request,
            PlanData(
                {
                    "runner": "vllm_gr.v1.worker.npu.npu_model_runner_patch",
                    "attention": "vllm_gr.v1.attention.backends.beam_attn_npu",
                    "prefill_attention": "ascend.cascade",
                    "decode_attention": gr["npu_decode_execution"] or "auto",
                    "requires_attention": gr["attention_backend"] == "CUSTOM",
                    "constraint": constraint,
                    "constraint_implementation": "npu" if constraint != "none" else None,
                    "stage": "ascend.v1",
                    "graph": "capture-with-eager-fallback" if gr["beam_graph_enabled"] else "eager",
                }
            ),
        )

    def create_attention_executor(
        self, *, plan: BackendPlan, device: object
    ) -> BeamAttentionExecution:
        require_provider_plan(self, plan)
        from vllm_gr.v1.attention.backends import beam_attn

        # Preserve the existing capability-sensitive default until #713 moves
        # its selection into the executor. Report the actual module to the
        # engine so workers cannot silently disagree on the auto choice.
        actual = beam_attn._get_npu_decode_execution()
        requested = plan.choices.to_dict()["decode_attention"]
        if requested != "auto" and actual != requested:
            raise RuntimeError(
                "Ascend attention mode is already fixed by another engine; use a separate process"
            )
        module = self.get_attention_execution()
        if not module.is_available():
            raise RuntimeError("Ascend attention implementation is unavailable")
        return module

    def matches_legacy_identity(self, identity: PlatformIdentity) -> bool:
        """API v1 selected Ascend by its original tensor placement."""
        return identity.device_type == self.device_type

    def create_config_service(self) -> BackendConfigService:
        from .config import AscendConfigService

        return AscendConfigService()

    def check_environment(self) -> dict[str, str]:
        from .environment import check_ascend_environment

        return check_ascend_environment(require_npu_platform=True)

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force  # Installers inspect their current targets on every call.
        from .spawn_patch import apply_npu_spawn_patch

        # Preserve spawn protection before native worker imports or process
        # creation. NPUWorker initializes Ascend ops before its ModelRunner;
        # keep that delayed import before the common Beam installer runs.
        spawn_count = int(apply_npu_spawn_patch())

        from vllm_gr.v1.worker.npu.startup_patch import patch_npu_worker_init_device

        worker_count = patch_npu_worker_init_device()
        return spawn_count + worker_count

    def get_model_runner_hooks(self) -> ModelRunnerInstaller:
        return cast(
            ModelRunnerInstaller,
            import_module("vllm_gr.v1.worker.npu.npu_model_runner_patch"),
        )

    def get_attention_execution(self) -> BeamAttentionExecution:
        from vllm_gr.v1.attention.backends.beam_attn import _get_npu_decode_execution

        path = "vllm_gr.v1.attention.backends.beam_attn_npu"
        if _get_npu_decode_execution() == "xllm":
            path += "_xllm"
        return cast(BeamAttentionExecution, import_module(path))

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
        plan: BackendPlan | None = None,
    ) -> NPUConstrainedTopKExecution:
        """Create a fresh NPU constrained top-k execution adapter.

        The adapter is stateful (holds loaded tables + digest), so a new
        instance is created per call; the ModelRunner owns it for one
        lifetime and creates another after a reset.
        """
        from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

        if plan is not None:
            require_provider_plan(self, plan)
        return NPUConstrainedTopKExecution(device=device)


NPU_BACKEND = NPUBackendProvider()
