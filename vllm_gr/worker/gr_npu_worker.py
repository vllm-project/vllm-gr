# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend NPU worker for GR/HSTU execution."""

import gc

from vllm_ascend.worker.worker import NPUWorker

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.logger import init_logger
from vllm_gr.utils.npu_adapter import release_accelerator_memory

logger = init_logger(__name__)


class GRNPUWorker(NPUWorker):
    """NPU worker that replaces the default runner with GRNPUModelRunner."""

    def init_device(self):
        super().init_device()

        from vllm_gr.patch import re_apply_patches

        re_apply_patches()

        from vllm_gr.worker.gr_npu_model_runner import GRNPUModelRunner

        gr_config = getattr(self.vllm_config, "gr_config", None) or GRConfig()
        self.model_runner = GRNPUModelRunner(self.vllm_config, self.device, gr_config=gr_config)
        logger.info(
            "GRNPUWorker: using GRNPUModelRunner on device %s",
            self.device,
        )

    def shutdown(self) -> None:
        super().shutdown()

        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        static_forward_context = getattr(
            compilation_config,
            "static_forward_context",
            None,
        )
        if isinstance(static_forward_context, dict):
            static_forward_context.clear()

        if hasattr(self, "model_runner"):
            runner = self.model_runner
            if runner is not None:
                model = getattr(runner, "model", None)
                if model is not None and hasattr(model, "_hstu_kv_cache_layers"):
                    model._hstu_kv_cache_layers = None
            self.model_runner = None

        gc.collect()
        release_accelerator_memory()
