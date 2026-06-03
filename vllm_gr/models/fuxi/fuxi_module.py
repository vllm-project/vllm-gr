# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FuXi-alpha RAB-off inference model.

Structure and naming follow ``FuXi-alpha/.../fuxi.py`` for weight alignment and diffing.
Necessary vllm-GR deviations are marked inline (KV-cache wiring, RAB-off attention).
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_gr.model_executor.layers.hstu_attention_layer import HSTUKVCacheAttentionLayer
from vllm_gr.models.fuxi.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor,
)
from vllm_gr.models.fuxi.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
    OutputPostprocessorModule,
)
from vllm_gr.models.fuxi.utils import get_current_embeddings
from vllm_gr.utils import ReliableOps

TIMESTAMPS_KEY = "timestamps"


def _latent_attention_jagged(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x_offsets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
) -> torch.Tensor:
    """RAB-off single-channel path aligned with FuXi-alpha latent AMS channel."""
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    padded_q = ReliableOps.jagged_to_padded_dense(
        values=q,
        offsets=[x_offsets],
        max_lengths=[n],
        padding_value=0.0,
    )
    padded_k = ReliableOps.jagged_to_padded_dense(
        values=k,
        offsets=[x_offsets],
        max_lengths=[n],
        padding_value=0.0,
    )
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q.view(B, n, num_heads, attention_dim),
        padded_k.view(B, n, num_heads, attention_dim),
    )
    qk_attn = F.silu(qk_attn) / n
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    padded_v = ReliableOps.jagged_to_padded_dense(
        values=v,
        offsets=[x_offsets],
        max_lengths=[n],
        padding_value=0.0,
    ).reshape(B, n, num_heads, linear_dim)
    output_latent = torch.einsum("bhnm,bmhd->bnhd", qk_attn, padded_v)
    combined_output = output_latent.reshape(B, n, num_heads * linear_dim)
    attn_output = ReliableOps.dense_to_jagged(combined_output, [x_offsets])
    return attn_output


class MultistageFeedforwardNeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        ams_output_size,
        input_size,
        hidden_size,
        output_size,
        dropout_ratio: float,
        bias: bool = False,
        single_stage: bool = False,
        epsilon: float = 1e-6,
    ):
        super(MultistageFeedforwardNeuralNetwork, self).__init__()
        self.lin0 = torch.nn.Linear(ams_output_size, input_size)
        self.is_single_stage = single_stage
        self.dropout_ratio = dropout_ratio
        self.input_size = input_size
        self.eps = epsilon
        if not single_stage:
            self.lin1 = torch.nn.Linear(input_size, hidden_size, bias=bias)
            self.lin2 = torch.nn.Linear(hidden_size, output_size, bias=bias)
            self.lin3 = torch.nn.Linear(input_size, hidden_size, bias=bias)

    def forward(self, X, X0):
        X = (
            self.lin0(
                F.dropout(
                    X,
                    p=self.dropout_ratio,
                    training=self.training,
                )
            )
            + X0
        )
        if not self.is_single_stage:
            normed_X = F.rms_norm(X, normalized_shape=[self.input_size], eps=self.eps)
            normed_X = F.dropout(
                normed_X,
                p=self.dropout_ratio,
                training=self.training,
            )
            X1 = F.silu(self.lin1(normed_X)) * self.lin3(normed_X)
            X = self.lin2(X1) + X
        return X

    def init(self):
        torch.nn.init.xavier_uniform_(self.lin0.weight)
        if not self.is_single_stage:
            torch.nn.init.xavier_uniform_(self.lin1.weight)
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            torch.nn.init.xavier_uniform_(self.lin3.weight)


class FuXiBlockJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[Any],
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        epsilon: float = 1e-6,
        ffn_multiply: float = 1,
        ffn_single_stage: bool = False,
        max_length: Optional[int] = None,
        *,
        vllm_config=None,
        prefix: str = "",
        layer_idx: int = 0,
        enable_kv_cache: bool = True,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias = relative_attention_bias_module
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        self._max_sequence_length: int = max_length or 0
        if self._linear_config == "uvqk":
            # RAB-off: u/v/q/k widths are H·dv + H·dv + H·dqk + H·dqk.
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * num_heads * 2 + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation

        self._mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size=linear_hidden_dim * num_heads,
            input_size=embedding_dim,
            hidden_size=int(embedding_dim * ffn_multiply),
            output_size=embedding_dim,
            dropout_ratio=dropout_ratio,
            single_stage=ffn_single_stage,
            epsilon=epsilon,
        )
        self._mffn.init()

        self._eps: float = epsilon

        # vllm-GR: optional paged-KV wiring (Phase 1 uses full mode only).
        torch_dtype = torch.bfloat16
        self.attn_layer = HSTUKVCacheAttentionLayer(
            layer_name=f"{prefix}.attn",
            num_heads=self._num_heads,
            head_size=self._attention_dim,
            head_size_v=self._linear_dim,
            dtype=torch_dtype,
            vllm_config=vllm_config,
            enable_kv_cache=bool(enable_kv_cache),
        )
        self.attn_layer.impl.scale = 1.0
        self.attn_layer.scale = 1.0

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x,
            normalized_shape=[self._linear_dim * self._num_heads],
            eps=self._eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        hstu_attn_metadata=None,
    ) -> torch.Tensor:
        del all_timestamps
        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        num_heads = self._num_heads
        dqk = self._attention_dim
        dv = self._linear_dim
        q3 = q.view(-1, num_heads, dqk)
        k3 = k.view(-1, num_heads, dqk)
        v3 = v.view(-1, num_heads, dv)

        if self._normalization == "rel_bias":
            kv_cache = self.attn_layer.get_runtime_kv_cache()
            if (
                kv_cache is not None
                and hstu_attn_metadata is not None
                and self._rel_attn_bias is None
            ):
                attn = self.attn_layer(
                    query=q3,
                    key=k3,
                    value=v3,
                    kv_cache=kv_cache,
                    attn_metadata=hstu_attn_metadata,
                    attn_mode="full",
                )
                attn_output = attn.reshape(-1, num_heads * dv)
            elif self._rel_attn_bias is not None:
                raise NotImplementedError(
                    "RAB-on FuXi attention is not supported in vllm-GR inference"
                )
            else:
                attn_output = _latent_attention_jagged(
                    num_heads=num_heads,
                    attention_dim=dqk,
                    linear_dim=dv,
                    q=q,
                    k=k,
                    v=v,
                    x_offsets=x_offsets,
                    invalid_attn_mask=invalid_attn_mask,
                )
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        ams_output = u * self._norm_attn_output(attn_output)
        return self._mffn(ams_output, x)


class FuXiJagged(torch.nn.Module):
    def __init__(
        self,
        modules: List[FuXiBlockJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=modules)
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        hstu_attn_metadata=None,
    ) -> torch.Tensor:
        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for layer in self._attention_layers:
                x = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    hstu_attn_metadata=hstu_attn_metadata,
                )
        return x

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        hstu_attn_metadata=None,
    ) -> torch.Tensor:
        if len(x.size()) == 3:
            x = ReliableOps.dense_to_jagged(x, [x_offsets])

        jagged_x = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            hstu_attn_metadata=hstu_attn_metadata,
        )
        return ReliableOps.jagged_to_padded_dense(
            values=jagged_x,
            offsets=[x_offsets],
            max_lengths=[invalid_attn_mask.size(1)],
            padding_value=0.0,
        )


def build_input_features_preprocessor(cfg: Any) -> InputFeaturesPreprocessorModule:
    positional_max_len = int(getattr(cfg, "positional_embedding_max_len", None) or cfg.max_seq_len)
    dropout_rate = float(getattr(cfg, "linear_dropout_rate", 0.0))
    preprocessor_type = getattr(cfg, "input_preprocessor_type", "posi")
    if cfg.use_rating_embedding or preprocessor_type in {"posir", "rated"}:
        return LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
            max_sequence_len=positional_max_len,
            item_embedding_dim=cfg.item_embedding_dim,
            dropout_rate=dropout_rate,
            rating_embedding_dim=cfg.rating_embedding_dim,
            num_ratings=cfg.num_ratings,
        )
    return LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=positional_max_len,
        embedding_dim=cfg.item_embedding_dim,
        dropout_rate=dropout_rate,
    )


def build_output_postprocessor(cfg: Any) -> OutputPostprocessorModule:
    emb_dim = cfg.item_embedding_dim + (cfg.rating_embedding_dim if cfg.use_rating_embedding else 0)
    postprocessor_type = getattr(cfg, "output_postprocessor_type", "l2")
    if postprocessor_type == "l2":
        return L2NormEmbeddingPostprocessor(embedding_dim=emb_dim)
    if postprocessor_type == "ln":
        return LayerNormEmbeddingPostprocessor(embedding_dim=emb_dim)
    return nn.Identity()


class FuXiAlphaInferenceModel(nn.Module):
    """Inference counterpart of FuXi-alpha ``FuXi`` (RAB-off, history-only encoder)."""

    def __init__(self, hf_config, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self._cfg = hf_config
        self._embedding_dim: int = hf_config.item_embedding_dim + (
            hf_config.rating_embedding_dim if hf_config.use_rating_embedding else 0
        )
        self._item_embedding_dim: int = hf_config.item_embedding_dim
        self._max_sequence_length: int = hf_config.max_seq_len
        self._num_blocks: int = hf_config.num_blocks
        self._num_heads: int = hf_config.num_heads
        self._dqk: int = hf_config.dqk
        self._dv: int = hf_config.dv
        self._linear_activation: str = getattr(hf_config, "linear_activation", "silu")
        self._linear_dropout_rate: float = float(getattr(hf_config, "linear_dropout_rate", 0.0))
        self._attn_dropout_rate: float = float(getattr(hf_config, "attn_dropout_rate", 0.0))
        max_output_len = int(getattr(hf_config, "max_output_len", 0))

        self._embedding_module = nn.Module()
        self._embedding_module._item_emb = nn.Embedding(
            num_embeddings=hf_config.vocab_size,
            embedding_dim=hf_config.item_embedding_dim,
            padding_idx=0,
        )

        self._input_features_preproc = build_input_features_preprocessor(hf_config)
        self._output_postproc = build_output_postprocessor(hf_config)

        block_prefix = f"{prefix}._fuxi._attention_layers" if prefix else "_fuxi._attention_layers"
        self._fuxi = FuXiJagged(
            modules=[
                FuXiBlockJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=self._dv,
                    attention_dim=self._dqk,
                    normalization="rel_bias",
                    linear_config="uvqk",
                    linear_activation=self._linear_activation,
                    num_heads=self._num_heads,
                    relative_attention_bias_module=None,
                    dropout_ratio=self._linear_dropout_rate,
                    attn_dropout_ratio=self._attn_dropout_rate,
                    ffn_multiply=hf_config.ffn_multiply,
                    ffn_single_stage=hf_config.ffn_single_stage,
                    epsilon=hf_config.layer_norm_eps,
                    max_length=self._max_sequence_length,
                    vllm_config=vllm_config,
                    prefix=f"{block_prefix}.{layer_idx}",
                    layer_idx=layer_idx,
                    enable_kv_cache=bool(hf_config.enable_kv_cache),
                )
                for layer_idx in range(self._num_blocks)
            ],
            autocast_dtype=None,
        )

        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

    @property
    def kv_cache_layers(self) -> list[nn.Module]:
        return [layer.attn_layer for layer in self._fuxi._attention_layers]

    def debug_str(self) -> str:
        debug_str = (
            f"FuXi-alpha-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            f"-ad{self._attn_dropout_rate}-norab"
        )
        return debug_str

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module._item_emb(item_ids)

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: dict,
        hstu_attn_metadata=None,
    ) -> torch.Tensor:
        """
        [B, N] -> [B, N, D].
        """
        float_dtype = past_embeddings.dtype

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        float_dtype = user_embeddings.dtype
        max_len = past_ids.size(1)
        attn_mask = self._attn_mask[:max_len, :max_len]
        x_offsets = ReliableOps.asynchronous_complete_cumsum(past_lengths)
        user_embeddings = self._fuxi(
            x=user_embeddings,
            x_offsets=x_offsets,
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY] if TIMESTAMPS_KEY in past_payloads else None
            ),
            invalid_attn_mask=1.0 - attn_mask.to(float_dtype),
            hstu_attn_metadata=hstu_attn_metadata,
        )
        return self._output_postproc(user_embeddings)

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: dict,
        hstu_attn_metadata=None,
    ) -> torch.Tensor:
        encoded_seq_embeddings = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            hstu_attn_metadata=hstu_attn_metadata,
        )
        return get_current_embeddings(
            lengths=past_lengths,
            encoded_embeddings=encoded_seq_embeddings,
        )

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: dict,
        hstu_attn_metadata=None,
    ) -> torch.Tensor:
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            hstu_attn_metadata=hstu_attn_metadata,
        )

    def _score_candidates(
        self,
        *,
        current_embeddings: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = current_embeddings.shape[0]
        score_chunks: list[torch.Tensor] = []
        temperature = max(float(getattr(self._cfg, "temperature", 1.0)), 1e-6)

        for batch_idx in range(batch_size):
            num_candidates = int(candidate_lengths[batch_idx].item())
            if num_candidates == 0:
                continue

            user_emb = current_embeddings[batch_idx]
            cand_ids = candidate_ids[batch_idx, :num_candidates]
            cand_emb = self.get_item_embeddings(cand_ids)
            if hasattr(self._input_features_preproc, "_rating_emb"):
                cand_emb = torch.cat(
                    [
                        cand_emb,
                        self._input_features_preproc._rating_emb(
                            torch.zeros_like(cand_ids),
                        ),
                    ],
                    dim=-1,
                )
            cand_emb = self._output_postproc(cand_emb.unsqueeze(0)).squeeze(0)
            scores = (cand_emb * user_emb.unsqueeze(0)).sum(dim=-1) / temperature
            score_chunks.append(scores)

        if not score_chunks:
            return torch.zeros(1, 0, device=current_embeddings.device)

        return torch.cat(score_chunks, dim=0).unsqueeze(0)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_payloads: Optional[dict] = None,
        candidate_ids: Optional[torch.Tensor] = None,
        candidate_lengths: Optional[torch.Tensor] = None,
        hstu_attn_metadata=None,
    ) -> torch.Tensor:
        past_payloads = past_payloads or {}
        past_embeddings = self.get_item_embeddings(past_ids)

        if candidate_ids is None or candidate_lengths is None:
            return self.generate_user_embeddings(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_embeddings=past_embeddings,
                past_payloads=past_payloads,
                hstu_attn_metadata=hstu_attn_metadata,
            )

        current_embeddings = self.encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            hstu_attn_metadata=hstu_attn_metadata,
        )
        return self._score_candidates(
            current_embeddings=current_embeddings,
            candidate_ids=candidate_ids,
            candidate_lengths=candidate_lengths,
        )
