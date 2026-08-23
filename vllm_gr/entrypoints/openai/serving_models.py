# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from vllm_gr.config import GRConfig
from vllm_gr.constraints import BeamConstraintCoordinator

_original_init = OpenAIServingModels.__init__


def init_gr(
    self,
    engine_client: EngineClient,
    base_model_paths: list[BaseModelPath],
    *,
    lora_modules: list[LoRAModulePath] | None = None,
):
    _original_init(self, engine_client, base_model_paths, lora_modules=lora_modules)
    vllm_config = getattr(engine_client, "vllm_config", None)
    gr_config = GRConfig.from_vllm_config(vllm_config)
    self.constraints_enabled = gr_config.constraint_table.enabled
    self.constraint_coordinator = BeamConstraintCoordinator(self.constraints_enabled)


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
