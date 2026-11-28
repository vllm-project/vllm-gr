# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend platform extensions for GR."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from typing import TYPE_CHECKING, cast

from vllm_ascend.platform import NPUPlatform

from ..interface import BeamAttentionExecution, GRPlatform

if TYPE_CHECKING:
    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner


class AscendGRPlatform(GRPlatform, NPUPlatform):
    native_platform_type = NPUPlatform
    backend_id = "ascend"
    vendor_distribution = "vllm-ascend"
    beam_attention_layout = "ascend"
    canonical_constraint_backend = "npu"
    default_use_v2_model_runner = False
    default_worker_decision = True
    supports_unconstrained = True
    supports_dynamic_width = True
    supports_sid_stage_vocab = True

    def get_gr_worker_cls(self) -> str:
        return "vllm_gr.v1.worker.npu.worker.AscendGRWorker"

    def check_environment(self) -> dict[str, str]:
        from .environment import check_ascend_environment

        return check_ascend_environment(require_npu_platform=True)

    def prepare_gr_engine(self) -> int:
        from .spawn_patch import apply_npu_spawn_patch

        spawn_count = int(apply_npu_spawn_patch())
        from vllm.v1.engine.core import EngineCoreProc

        from .config_patch import apply_engine_core_config_sanitizer

        apply_engine_core_config_sanitizer(EngineCoreProc)
        return spawn_count

    def check_and_update_gr_config(self, config) -> None:
        cache, model = config.cache_config, config.model_config
        # EngineArgs paths can carry configs whose model_config is not yet
        # constructed (and it is legitimately None in unit-test doubles);
        # the retired AscendConfigService tolerated that via getattr, and
        # config-finalization order must not depend on model presence here.
        if not getattr(cache, "user_specified_block_size", False) and not getattr(
            model, "is_hybrid", False
        ):
            cache.block_size = 128
        from vllm_gr.config import GRAttentionBackend, GRConfig

        gr = GRConfig.from_vllm_config(config)
        if gr.beam_max_width is not None or gr.attention_backend is GRAttentionBackend.CUSTOM:
            if getattr(config, "use_v2_model_runner", False):
                raise ValueError("NPU GR requires ModelRunner V1; set VLLM_USE_V2_MODEL_RUNNER=0")
            self.validate_gr_request(config, None)
        # Native plugin initialization can replace the process entrypoint again.
        from vllm.v1.engine.core import EngineCoreProc

        from .config_patch import apply_engine_core_config_sanitizer

        apply_engine_core_config_sanitizer(EngineCoreProc)

    def get_beam_limits(self, config):
        from vllm_gr.config import GRConfig

        from ..admission import local_kv_geometry

        heads, size = None, None
        if GRConfig.from_vllm_config(config).beam_max_decode_steps > 3:
            heads, size = local_kv_geometry(config)
        return replace(
            super().get_beam_limits(config),
            max_decode_rows=getattr(
                getattr(config, "scheduler_config", None), "max_num_batched_tokens", None
            ),
            retain_candidate_widths=True,
            local_kv_heads=heads,
            head_size=size,
        )

    def validate_gr_request(self, config, request) -> None:
        from vllm_gr.config import GRConfig

        from ..admission import beam_limits
        from .admission import validate_npu_device_config

        validate_npu_device_config(beam_limits(config), GRConfig.from_vllm_config(config), request)

    def get_attention_execution(self) -> BeamAttentionExecution:
        from vllm.config import get_current_vllm_config_or_none

        from .decode import engine_mode

        return self._attention_module(engine_mode(get_current_vllm_config_or_none()))

    @staticmethod
    def _attention_module(mode: str) -> BeamAttentionExecution:
        path = "vllm_gr.v1.attention.backends.beam_attn_npu"
        if mode == "xllm":
            path += "_xllm"
        return cast(BeamAttentionExecution, import_module(path))

    def get_prefill_graph_runner_class(self) -> type[BasePrefillGraphRunner]:
        from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

        return NPUPrefillGraphRunner
