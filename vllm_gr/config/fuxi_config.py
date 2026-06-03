# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FuXi-alpha configuration definitions."""

from __future__ import annotations

from typing import Any, List, Optional

from transformers import PretrainedConfig


class FuXiConfig(PretrainedConfig):
    """HuggingFace-compatible config for FuXi-alpha inference (RAB-off)."""

    model_type = "fuxi"
    architectures = ["FuXiEmbeddingModel"]
    is_encoder_decoder = False
    is_text_encoder = True
    is_attention_free = False

    def __init__(
        self,
        max_seq_len: int = 200,
        max_num_candidates_inference: int = 100,
        item_embedding_dim: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dqk: int = 50,
        dv: int = 50,
        ffn_multiply: float = 1.0,
        ffn_single_stage: bool = False,
        linear_activation: str = "silu",
        linear_dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
        max_output_len: int = 0,
        layer_norm_eps: float = 1e-6,
        enable_relative_attention_bias: bool = False,
        enable_kv_cache: bool = True,
        input_preprocessor_type: str = "posi",
        positional_embedding_max_len: Optional[int] = None,
        rating_embedding_dim: int = 0,
        num_ratings: int = 0,
        output_postprocessor_type: str = "l2",
        item_l2_norm: bool = True,
        temperature: float = 0.05,
        item_id_feature_name: str = "item_id",
        rating_feature_name: str = "rating",
        timestamp_feature_name: str = "timestamp",
        use_rating_embedding: bool = False,
        use_timestamp_payload: bool = False,
        task_names: Optional[List[str]] = None,
        vocab_size: int = 4_000_000,
        hash_size: int = 4_000_000,
        is_inference: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if enable_relative_attention_bias:
            raise ValueError(
                "FuXi-RAB-off adapter only supports enable_relative_attention_bias=False"
            )

        self.max_seq_len = max_seq_len
        self.max_num_candidates_inference = max_num_candidates_inference
        self.item_embedding_dim = item_embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dqk = dqk
        self.dv = dv
        self.ffn_multiply = ffn_multiply
        self.ffn_single_stage = ffn_single_stage
        self.linear_activation = linear_activation
        self.linear_dropout_rate = linear_dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.max_output_len = max_output_len
        self.layer_norm_eps = layer_norm_eps
        self.enable_relative_attention_bias = False
        self.enable_kv_cache = bool(enable_kv_cache)
        self.input_preprocessor_type = input_preprocessor_type
        self.positional_embedding_max_len = positional_embedding_max_len
        self.rating_embedding_dim = rating_embedding_dim
        self.num_ratings = num_ratings
        self.output_postprocessor_type = output_postprocessor_type
        self.item_l2_norm = item_l2_norm
        self.temperature = temperature
        self.item_id_feature_name = item_id_feature_name
        self.rating_feature_name = rating_feature_name
        self.timestamp_feature_name = timestamp_feature_name
        self.use_rating_embedding = use_rating_embedding
        self.use_timestamp_payload = use_timestamp_payload
        self.task_names = list(task_names) if task_names else ["score"]
        self.vocab_size = vocab_size
        self.hash_size = hash_size
        self.is_inference = is_inference


def get_fuxi_max_model_len(config: FuXiConfig | dict[str, Any]) -> int:
    """Return scheduler-visible sequence length for FuXi (history only)."""
    if isinstance(config, dict):
        return int(config.get("max_seq_len", 0))
    return int(getattr(config, "max_seq_len", 0))
