# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from vllm_gr.config import GRConfig
from vllm_gr.utils.catalog import Catalog

_original_init = OpenAIServingModels.__init__


def init_gr(
    self,
    engine_client: EngineClient,
    base_model_paths: list[BaseModelPath],
    *,
    lora_modules: list[LoRAModulePath] | None = None,
):
    _original_init(self, engine_client, base_model_paths, lora_modules=lora_modules)
    self.catalog = None
    tokenizer = self.input_processor.tokenizer
    vllm_config = getattr(engine_client, "vllm_config", None)
    catalog_path = GRConfig.from_vllm_config(vllm_config).catalog_path
    if catalog_path:
        self.catalog = Catalog(tokenizer)
        self.catalog.load(catalog_path)


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
