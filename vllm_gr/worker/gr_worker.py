# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRWorker — extends GPUWorker to use GRModelRunner.

Injected via worker_cls in GREngineArgs so vLLM's EngineCore
instantiates this worker instead of the default GPUWorker.

The vllm-GR plugin (registered via entry_points) applies monkey-patches
in every process via load_general_plugins().  The re_apply_patches()
call in init_device() is a defense-in-depth measure that catches any
vLLM modules loaded after the initial plugin execution.
"""

import gc

from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.logger import init_logger
from vllm_gr.utils.npu_adapter import release_accelerator_memory

logger = init_logger(__name__)


class GRWorker(GPUWorker):
    """GPU worker that replaces the default model runner with GRModelRunner."""

    def init_device(self):
        super().init_device()

        # Defense-in-depth: re-apply patches to cover any vLLM modules
        # loaded after the plugin's initial patch in load_general_plugins().
        from vllm_gr.patch import re_apply_patches

        re_apply_patches()

        from vllm_gr.worker.gr_model_runner import GRModelRunner

        gr_config = getattr(self.vllm_config, "gr_config", None) or GRConfig()
        self.model_runner = GRModelRunner(self.vllm_config, self.device, gr_config=gr_config)

        logger.info("GRWorker: using GRModelRunner on device %s", self.device)

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
