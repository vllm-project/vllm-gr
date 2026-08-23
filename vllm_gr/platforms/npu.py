# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin NPU provider with lazy Ascend-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import cast

from .protocol import BeamAttentionExecution, ModelRunnerPatchHooks


@dataclass(frozen=True)
class NPUBackendProvider:
    device_type: str = "npu"

    def check_environment(self) -> dict[str, str]:
        from vllm_gr._env_check import check_ascend_environment

        return check_ascend_environment(require_npu_platform=True)

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        from vllm_gr.v1 import ascend_compat

        attention_count = ascend_compat.apply_ascend_attention_backend_patch(
            force=force,
        )
        worker_count = ascend_compat.patch_npu_worker_init_device()
        return attention_count + worker_count

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(
            ModelRunnerPatchHooks,
            import_module("vllm_gr.v1.worker.npu_model_runner_patch"),
        )

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_npu"),
        )

NPU_BACKEND = NPUBackendProvider()
