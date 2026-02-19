# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any
from vllm.tokenizers import TokenizerLike
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.serving_models import BaseModelPath, LoRAModulePath, OpenAIServingModels



class Catalog:

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer
        self.trie: dict[int, Any] = {}

    def load(self, path: str):
        try:
            with open(path) as f:
                catalog_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Catalog file not found: {path}")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from catalog file: {path}")

        for seq in catalog_data:
            token_ids = self.tokenizer.convert_tokens_to_ids(seq)
            node = self.trie
            for token_id in token_ids:
                if token_id not in node:
                    node[token_id] = {}
                node = node[token_id]

    def valid(self, token_ids: list[int]) -> set[int]:
        node = self.trie
        for tid in token_ids:
            if tid in node:
                node = node[tid]
            else:
                return []
        return set(list(node.keys()))
    
def initGr(self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules: list[LoRAModulePath] | None = None):
    original_init = OpenAIServingModels.__init__
    original_init(self, engine_client, base_model_paths, lora_modules=lora_modules)
    tokenizer = self.input_processor.tokenizer
    if self.model_config.catalog_path:
        self.catalog = Catalog(tokenizer)
        self.catalog.load(self.model_config.catalog_path)


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = initGr