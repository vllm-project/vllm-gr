# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..config import BackendConfigService

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig


class AscendConfigService(BackendConfigService):
    supports_sid_stage_vocab = True
    infer_matching_candidate_widths = True
    default_worker_decision = True
    promote_catalog_tables = True

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
        if gr.constrained_generation is False and gr.beam_max_width is not None:
            from vllm_gr.v1.engine.beam_search_v1 import (
                validate_npu_unconstrained_runner,
            )

            validate_npu_unconstrained_runner(config)

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        # Wrap after all native plugins and before EngineCore is spawned.
        from vllm.v1.engine.core import EngineCoreProc

        from vllm_gr.v1.ascend_compat import apply_engine_core_config_sanitizer

        apply_engine_core_config_sanitizer(EngineCoreProc)

    def validate_device(self, config: Any, gr: GRConfig, request: Any) -> None:
        from vllm_gr.v1.engine.npu_admission import validate_npu_device_config

        validate_npu_device_config(config, gr, request)

    def decode_batch_decline_reason(
        self, config: Any, batch_size: int, beam_width: int
    ) -> str | None:
        capacity = config.scheduler_config.max_num_batched_tokens // beam_width
        if batch_size > capacity:
            return (
                f"NPU Decode batch size {batch_size} exceeds the physical input "
                f"capacity of {capacity} prompts at the requested beam width"
            )
        return None

    def validate_request_budget(self, config: Any, request: Any) -> None:
        from vllm_gr.v1.engine.npu_admission import validate_decode_token_budget

        validate_decode_token_budget(request, config)
