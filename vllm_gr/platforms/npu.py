# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin NPU provider with lazy Ascend-specific imports."""

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import cast

from .protocol import BACKEND_API_VERSION, BeamAttentionHooks, ModelRunnerPatchHooks

logger = logging.getLogger("vllm_gr.platforms.npu")


@dataclass(frozen=True)
class NPUBackendProvider:
    api_version: int = BACKEND_API_VERSION
    device_type: str = "npu"

    def configure_engine(self, config: object) -> None:
        cache_config = getattr(config, "cache_config", None)
        model_config = getattr(config, "model_config", None)
        if cache_config is None:
            return
        if getattr(cache_config, "user_specified_block_size", False):
            return
        if model_config is not None and getattr(model_config, "is_hybrid", False):
            return
        if cache_config.block_size != 128:
            logger.info(
                "Using Ascend-compatible default KV cache block size 128 "
                "(vLLM resolved the unspecified value to %s).",
                cache_config.block_size,
            )
            cache_config.block_size = 128

    def check_environment(self) -> dict[str, str]:
        from vllm_gr._env_check import check_ascend_environment

        return cast(
            dict[str, str],
            check_ascend_environment(require_npu_platform=True),
        )

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        from vllm_gr.v1 import ascend_compat

        attention_count = ascend_compat.apply_ascend_attention_backend_patch(
            force=force,
        )
        worker_count = ascend_compat.patch_npu_worker_init_device()
        return int(attention_count + worker_count)

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(
            ModelRunnerPatchHooks,
            import_module("vllm_gr.v1.worker.npu_model_runner_patch"),
        )

    def get_attention_hooks(self) -> BeamAttentionHooks:
        return cast(
            BeamAttentionHooks,
            import_module("vllm_gr.v1.attention.backends.beam_attn_npu"),
        )

    def release_accelerator_memory(self) -> None:
        import torch

        from vllm_gr.utils.accelerator_memory import cleanup_torch_backend

        cleanup_torch_backend(getattr(torch, "npu", None), collect_ipc=False)


NPU_BACKEND = NPUBackendProvider()
