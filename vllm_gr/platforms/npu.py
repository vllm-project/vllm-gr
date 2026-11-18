# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin Ascend provider with lazy NPU-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from .identity import PlatformIdentity
from .protocol import (
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    ModelRunnerPatchHooks,
)

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

    from .services import BackendConfigService, BeamAttentionService, StageExecutionService

#: Qualified name of the vendor platform class this provider supports. The
#: check is a lineage lookup on the already-selected platform instance, so it
#: imports no vendor code and core keeps no platform table.
_ASCEND_PLATFORM_CLASS = "vllm_ascend.platform.NPUPlatform"


@dataclass(frozen=True)
class NPUBackendProvider:
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
    gr_version_range: ClassVar[str | None] = ">=0.30.0,<0.31.0"
    vllm_version_range: ClassVar[str | None] = "==0.30.0"
    vendor_version_range: ClassVar[str | None] = "==0.30.0rc2"
    vendor_distribution: ClassVar[str | None] = "vllm-ascend"

    def matches(self, identity: PlatformIdentity) -> bool:
        """Match the supported Ascend platform identity and ``npu`` placement.

        The vendor platform enum is ``OOT``, which a future out-of-tree provider
        could share, so the qualified class lineage is required as well.
        """
        return identity.device_type == "npu" and identity.has_lineage(_ASCEND_PLATFORM_CLASS)

    def check_environment(self) -> dict[str, str]:
        from vllm_gr._env_check import check_ascend_environment

        return check_ascend_environment(require_npu_platform=True)

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force  # Installers inspect their current targets on every call.
        from vllm_gr import npu_spawn_patch
        from vllm_gr.v1 import ascend_compat

        # Preserve spawn protection before native worker imports or process
        # creation. NPUWorker initializes Ascend ops before its ModelRunner;
        # keep that delayed import before the common Beam installer runs.
        spawn_count = int(npu_spawn_patch.apply_npu_spawn_patch())
        worker_count = ascend_compat.patch_npu_worker_init_device()
        return spawn_count + worker_count

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(
            ModelRunnerPatchHooks,
            import_module("vllm_gr.v1.worker.npu_model_runner_patch"),
        )

    def select_runner_patch_module_name(self, *, prefer_v2: bool = False) -> str:
        """Return the Ascend runner patch module.

        Ascend has no ModelRunner V2 implementation in this release, so the V2
        opt-in does not apply here and the flag is ignored rather than silently
        falling back to another backend's module.
        """
        del prefer_v2
        return "vllm_gr.v1.worker.npu_model_runner_patch"

    def supports_v2_model_runner(self) -> bool:
        """Ascend has no ModelRunner V2 patch module in this release."""
        return False

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_npu"),
        )

    def create_beam_attention_service(self) -> BeamAttentionService:
        """Return the Ascend Beam attention service.

        The service defers its own imports, so building it loads no vendor
        runtime and this provider keeps its import-light contract.
        """
        from .ascend_services import AscendBeamAttentionService

        return AscendBeamAttentionService()

    def create_config_service(self) -> BackendConfigService:
        """Return Ascend's configuration-time policy."""
        from .ascend_services import AscendConfigService

        return AscendConfigService()

    def create_stage_execution_service(self) -> StageExecutionService:
        """Return the Ascend stage execution service.

        The service defers its own imports, so building it loads no vendor
        runtime and this provider keeps its import-light contract.
        """
        from .ascend_services import AscendStageExecutionService

        return AscendStageExecutionService()

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "NPUConstrainedTopKExecution":
        """Create a fresh NPU constrained top-k execution adapter.

        The adapter is stateful (holds loaded tables + digest), so a new
        instance is created per call; the ModelRunner owns it for one
        lifetime and creates another after a reset.
        """
        from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

        return NPUConstrainedTopKExecution(device=device)


NPU_BACKEND = NPUBackendProvider()
