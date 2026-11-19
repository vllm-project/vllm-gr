# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin Ascend provider with lazy NPU-specific imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..constraint_plan import constraint_plan_choices
from ..graph import bind_graph, prefill_enabled
from ..identity import PlatformIdentity
from ..interface import (
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    ModelRunnerInstaller,
)
from ..plan import BackendPlan, BackendRequest, PlanData
from ..services import BuiltinPlanSupport, require_provider_plan
from .config import AscendConfigService
from .graph import AscendGraph

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.npu_execution import NPUConstrainedTopKExecution

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
    deferred_patch_ids: ClassVar[tuple[str, ...]] = (
        "GR-PATCH-002",
        "GR-PATCH-012",
        "GR-PATCH-018",
        "GR-PATCH-019",
        "GR-PATCH-021",
    )
    gr_version_range: ClassVar[str | None] = ">=0.30.0,<0.31.0"
    vllm_version_range: ClassVar[str | None] = "==0.30.0"
    vendor_version_range: ClassVar[str | None] = "==0.30.0rc2"
    vendor_distribution: ClassVar[str | None] = "vllm-ascend"

    def create_config_service(self) -> AscendConfigService:
        return AscendConfigService()

    @property
    def runtime_patch_checks(self) -> dict[str, Callable[[], bool | None]]:
        from vllm_gr.npu_spawn_patch import is_npu_spawn_patched
        from vllm_gr.v1.ascend_compat import is_npu_worker_patched

        return {"GR-PATCH-002": is_npu_worker_patched, "GR-PATCH-016": is_npu_spawn_patched}

    def matches(self, identity: PlatformIdentity) -> bool:
        """Match the supported Ascend platform identity and ``npu`` placement.

        The vendor platform enum is ``OOT``, which a future out-of-tree provider
        could share, so the qualified class lineage is required as well.
        """
        return identity.device_type == self.device_type and identity.has_lineage(
            _ASCEND_PLATFORM_CLASS
        )

    def matches_legacy_identity(self, identity: PlatformIdentity) -> bool:
        """API v1 selected Ascend by its original tensor placement."""
        return identity.device_type == self.device_type

    native_graph_factory = AscendGraph

    def create_graph_executor(self, *, plan: BackendPlan, runner: Any) -> Any:
        require_provider_plan(self, plan)
        return bind_graph(plan, runner, AscendGraph)

    def resolve(self, request: BackendRequest) -> BackendPlan:
        """Keep Ascend's V1 runner and its existing auto-decode policy."""
        if request.options.to_dict():
            raise ValueError("Ascend currently accepts no additional backend options")
        gr = request.config.to_dict()
        native = request.native.to_dict()
        eager = (native.get("model_config") or {}).get("enforce_eager", False)
        options = native.get("graph_options", {})
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
                    "kv_layout": request.native.to_dict().get("kv_cache_layout"),
                    "requires_attention": gr["attention_backend"] == "CUSTOM",
                    **constraint_plan_choices(gr, legacy_id=self.device_type),
                    "operator_policy": "ascend.xllm-with-reference-fallback",
                    "stage": "ascend.v1",
                    "graph_implementation": AscendGraph.implementation_id,
                    "prefill_graph": not eager and prefill_enabled(options),
                    "graph_options": options,
                    "graph": "capture-with-eager-fallback"
                    if gr["beam_graph_enabled"] and not eager
                    else "eager",
                }
            ),
        )

    def native_attention(self, config: Any) -> Any:
        from os import environ

        from vllm_gr.config import GRConfig

        from .attention import AscendAttention

        configured = GRConfig.from_vllm_config(config).npu_decode_execution

        return AscendAttention(
            (configured.value if configured else "auto") if config is not None else "cascade",
            kv_layout=environ.get("VLLM_KV_CACHE_LAYOUT"),
        )

    def attention_admission(self, *, plan: BackendPlan) -> Any:
        return self._attention_for_plan(plan)

    def _attention_for_plan(self, plan: BackendPlan) -> Any:
        from .attention import AscendAttention

        require_provider_plan(self, plan)
        return AscendAttention(
            plan.choices.to_dict()["decode_attention"],
            kv_layout=plan.choices.to_dict().get("kv_layout"),
        )

    def create_attention_executor(self, *, plan: BackendPlan, device: object) -> Any:
        executor = self._attention_for_plan(plan)
        if not executor.execution_module().is_available():
            raise RuntimeError("Admitted Ascend attention implementation is unavailable")
        return executor

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

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None:
        super().validate_worker(plan=plan, native_runner=native_runner)
        if (
            plan.choices.to_dict().get("constraint") == "canonical"
            and plan.request.config.to_dict()["beam_execution_mode"] == "v1"
        ):
            backend = native_runner.constraint_backend
            if backend.info.table_metadata.trie_count > 1 and not callable(
                getattr(backend, "select_for_rows", None)
            ):
                raise RuntimeError("Ascend multi-trie selection requires host-row routing")

    def get_model_runner_hooks(self) -> ModelRunnerInstaller:
        return cast(
            ModelRunnerInstaller,
            import_module("vllm_gr.v1.worker.npu.npu_model_runner_patch"),
        )

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_npu"),
        )

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
        from vllm_gr.constraints.npu_execution import NPUConstrainedTopKExecution

        if plan is not None:
            require_provider_plan(self, plan)
        return NPUConstrainedTopKExecution(device=device)


NPU_BACKEND = NPUBackendProvider()
