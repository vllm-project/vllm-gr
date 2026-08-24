# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from vllm_gr.config import GRConfig

_original_init = OpenAIServingModels.__init__


def _validate_constraint_protocol_available(
    engine_client: EngineClient,
    gr_config: GRConfig,
) -> None:
    if gr_config.constraint_table.enabled and not callable(
        getattr(engine_client, "beam_request_step_update", None)
    ):
        raise RuntimeError(
            "constraint_table requires an engine client implementing "
            "beam_request_step_update (EngineCore beam sessions); the "
            "fallback cannot carry step/prefix metadata."
        )


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
    self._gr_config = gr_config
    self.constraints_enabled = gr_config.constraint_table.enabled
    _validate_constraint_protocol_available(engine_client, gr_config)


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
