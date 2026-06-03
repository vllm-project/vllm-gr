# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM/HuggingFace registration helpers for FuXi config integration."""

from __future__ import annotations

import logging

from transformers import AutoConfig

from vllm_gr.config.fuxi_config import FuXiConfig, get_fuxi_max_model_len

logger = logging.getLogger(__name__)

_FUXI_CONFIG_REGISTERED = False


def _register_fuxi_config() -> bool:
    global _FUXI_CONFIG_REGISTERED
    if _FUXI_CONFIG_REGISTERED:
        return True

    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    class FuXiModelArchConfigConvertor(ModelArchConfigConvertorBase):
        """Maps FuXiConfig fields to vLLM's ModelArchitectureConfig interface."""

        def get_hidden_size(self) -> int:
            return self.hf_text_config.item_embedding_dim

        def get_num_hidden_layers(self) -> int:
            return self.hf_text_config.num_blocks

        def get_total_num_attention_heads(self) -> int:
            return self.hf_text_config.num_heads

        def get_head_size(self) -> int:
            return self.hf_text_config.dqk

        def get_vocab_size(self) -> int:
            return self.hf_text_config.vocab_size

        def get_total_num_kv_heads(self) -> int:
            if getattr(self.hf_text_config, "enable_kv_cache", False):
                return self.hf_text_config.num_heads
            return 0

        def derive_max_model_len_and_key(self) -> tuple:
            return float(get_fuxi_max_model_len(self.hf_text_config)), "fuxi_max_seq_len"

    try:
        AutoConfig.register("fuxi", FuXiConfig)
        _FUXI_CONFIG_REGISTERED = True
    except ValueError as exc:
        if "already used" in str(exc).lower():
            _FUXI_CONFIG_REGISTERED = True
        else:
            logger.error("Failed to register FuXiConfig: %s", exc)
            return False

    if "fuxi" not in MODEL_ARCH_CONFIG_CONVERTORS:
        MODEL_ARCH_CONFIG_CONVERTORS["fuxi"] = FuXiModelArchConfigConvertor

    return True


def ensure_fuxi_config_registered() -> bool:
    return _register_fuxi_config()


def is_fuxi_config_registered() -> bool:
    return _FUXI_CONFIG_REGISTERED


__all__ = [
    "ensure_fuxi_config_registered",
    "is_fuxi_config_registered",
]
