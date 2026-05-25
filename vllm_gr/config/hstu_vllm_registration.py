# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM/HuggingFace registration helpers for HSTU config integration."""

from __future__ import annotations

import logging

from transformers import AutoConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

from vllm_gr.config.hstu_config import HSTUConfig, get_hstu_max_model_len

logger = logging.getLogger(__name__)


class HSTUModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Maps HSTUConfig fields to vLLM's ModelArchitectureConfig interface."""

    def get_hidden_size(self) -> int:
        return self.hf_text_config.hstu_transducer_embedding_dim

    def get_num_hidden_layers(self) -> int:
        return self.hf_text_config.hstu_attn_num_layers

    def get_total_num_attention_heads(self) -> int:
        return self.hf_text_config.hstu_num_heads

    def get_head_size(self) -> int:
        return self.hf_text_config.hstu_attn_qk_dim

    def get_vocab_size(self) -> int:
        return self.hf_text_config.hash_size

    def get_total_num_kv_heads(self) -> int:
        if getattr(self.hf_text_config, "enable_kv_cache", False):
            return self.hf_text_config.hstu_num_heads
        return 0

    def derive_max_model_len_and_key(self) -> tuple:
        return float(get_hstu_max_model_len(self.hf_text_config)), (
            "hstu_full_visible_seq_len"
        )


_HSTU_CONFIG_REGISTERED = False


def _register_hstu_config() -> bool:
    global _HSTU_CONFIG_REGISTERED
    if _HSTU_CONFIG_REGISTERED:
        return True

    try:
        AutoConfig.register("hstu", HSTUConfig)
        _HSTU_CONFIG_REGISTERED = True
    except ValueError as exc:
        if "already used" in str(exc).lower():
            _HSTU_CONFIG_REGISTERED = True
        else:
            logger.error("Failed to register HSTUConfig: %s", exc)
            return False

    if "hstu" not in MODEL_ARCH_CONFIG_CONVERTORS:
        MODEL_ARCH_CONFIG_CONVERTORS["hstu"] = HSTUModelArchConfigConvertor

    return True


def ensure_hstu_config_registered() -> bool:
    return _register_hstu_config()


def is_hstu_config_registered() -> bool:
    return _HSTU_CONFIG_REGISTERED


__all__ = [
    "HSTUModelArchConfigConvertor",
    "ensure_hstu_config_registered",
    "is_hstu_config_registered",
]
