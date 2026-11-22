# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vllm_gr.config.models import ConstraintMode, configured_constraint_mode

from ..config import BackendConfigService

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig


class AscendConfigService(BackendConfigService):
    default_worker_decision = True

    def apply_native_defaults(self, config: Any) -> None:
        cache = getattr(config, "cache_config", None)
        model = getattr(config, "model_config", None)
        if cache is None or getattr(cache, "user_specified_block_size", False):
            return
        if getattr(model, "is_hybrid", False):
            return
        if cache.block_size != 128:
            logging.getLogger(__name__).info(
                "Using Ascend default KV cache block size 128 (native default %s)", cache.block_size
            )
            cache.block_size = 128

    def validate_engine(self, config: Any, gr: GRConfig) -> None:
        if configured_constraint_mode(gr) is ConstraintMode.NONE and gr.beam_max_width is not None:
            from vllm_gr.platforms.ascend.admission import (
                validate_npu_unconstrained_runner,
            )

            validate_npu_unconstrained_runner(config)

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        # Wrap after all native plugins and before EngineCore is spawned.
        from vllm.v1.engine.core import EngineCoreProc

        from .config_patch import apply_engine_core_config_sanitizer

        apply_engine_core_config_sanitizer(EngineCoreProc)
