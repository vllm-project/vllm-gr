# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)
from vllm.tokenizers import TokenizerLike

from vllm_gr.logger import init_logger

logger = init_logger(__name__)

_original_init = OpenAIServingModels.__init__


def compute_catalog_slice_config(catalog_path: str, tokenizer_or_model: Any):
    """Calculate (vocab_offset, vocab_slice_size) from a JSON catalog file."""
    import json
    try:
        if isinstance(tokenizer_or_model, str):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_model, trust_remote_code=True)
        else:
            tokenizer = tokenizer_or_model

        with open(catalog_path) as f:
            catalog_data = json.load(f)

        min_tid = float("inf")
        max_tid = -float("inf")

        for seq in catalog_data:
            token_ids = tokenizer.convert_tokens_to_ids(seq)
            for token_str, token_id in zip(seq, token_ids):
                if token_id is not None and token_id != -1:
                    if not (isinstance(token_str, str) and token_str.startswith("<|sid_")):
                        if token_id < min_tid:
                            min_tid = token_id
                        if token_id > max_tid:
                            max_tid = token_id

        if min_tid != float("inf") and max_tid != -float("inf"):
            return int(min_tid), int(max_tid - min_tid + 1)
    except Exception as e:
        logger.warning("vllm_gr: Failed to compute catalog slice config: %s", e)
    return None, None


class Catalog:
    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer
        self.trie: dict[int, Any] = {}
        self.vocab_offset: int | None = None
        self.vocab_slice_size: int | None = None

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
            for i, (token_str, token_id) in enumerate(zip(seq, token_ids)):
                if token_id is None or token_id == -1:
                    raise ValueError(f"Invalid token ID: {seq[i]} at index {i}")
                if token_id not in node:
                    node[token_id] = {}
                node = node[token_id]

        offset, slice_size = compute_catalog_slice_config(path, self.tokenizer)
        if offset is not None and slice_size is not None:
            self.vocab_offset = offset
            self.vocab_slice_size = slice_size
            from vllm_gr.ops.sliced_logits import set_vocab_slice_config

            set_vocab_slice_config(self.vocab_offset, self.vocab_slice_size)
            logger.info(
                "vllm_gr: Catalog vocabulary slice configured: offset=%d, slice_size=%d (min_token_id=%d, max_token_id=%d)",
                self.vocab_offset,
                self.vocab_slice_size,
                self.vocab_offset,
                self.vocab_offset + self.vocab_slice_size - 1,
            )

    def valid(self, token_ids: list[int]) -> set[int]:
        node = self.trie
        for tid in token_ids:
            if tid in node:
                node = node[tid]
            else:
                return set()
        return set(node.keys())


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
    additional_config = getattr(vllm_config, "additional_config", None)
    catalog_path = (
        additional_config.get("vllm_gr_catalog_path")
        if isinstance(additional_config, dict)
        else None
    )
    if catalog_path:
        self.catalog = Catalog(tokenizer)
        self.catalog.load(catalog_path)
        if isinstance(additional_config, dict) and self.catalog.vocab_offset is not None:
            additional_config["vllm_gr_vocab_offset"] = self.catalog.vocab_offset
            additional_config["vllm_gr_vocab_slice_size"] = self.catalog.vocab_slice_size


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
