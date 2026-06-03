# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-time TorchRec compatibility helpers for HSTU embedding lookup."""

from __future__ import annotations

from typing import Any, Dict

import torch


def _resolve_embedding_index_upper_bound(embedding_module: Any) -> int:
    """Return the per-table maximum valid id for a TorchRec embedding module."""
    weight = getattr(embedding_module, "weight", None)
    if hasattr(weight, "shape") and len(weight.shape) >= 1:
        return max(int(weight.shape[0]) - 1, 0)

    num_embeddings = getattr(embedding_module, "num_embeddings", None)
    if num_embeddings is not None:
        return max(int(num_embeddings) - 1, 0)

    raise AttributeError("Embedding module does not expose weight.shape[0] or num_embeddings")


def _validate_embedding_indices(
    *,
    feature_name: str,
    embedding_name: str,
    indices: torch.Tensor,
    upper_bound: int,
) -> torch.Tensor:
    """Fail fast on out-of-range ids to preserve inference precision."""
    if indices.numel() == 0:
        return indices

    min_id = int(indices.min().item())
    max_id = int(indices.max().item())
    if min_id < 0 or max_id > upper_bound:
        raise ValueError(
            "Embedding lookup received out-of-range ids for "
            f"feature={feature_name!r}, embedding={embedding_name!r}: "
            f"expected ids in [0, {upper_bound}], got min={min_id}, max={max_id}"
        )
    return indices


def ec_patched_forward_wo_embedding_copy(
    ec_module: Any,
    features: Any,
) -> Dict[str, Any]:
    """Mirror golden sparse inference lookup behavior for EmbeddingCollection.

    This patch keeps the inference path close to the upstream `model_family`
    implementation while failing fast on invalid ids for the inference-only
    adapter.
    """
    from torchrec.sparse.jagged_tensor import JaggedTensor
    from torchrec.sparse.tensor_dict import maybe_td_to_kjt

    features = maybe_td_to_kjt(features, None)
    feature_embeddings: Dict[str, Any] = {}
    jt_dict: Dict[str, Any] = features.to_dict()

    for i, emb_module in enumerate(ec_module.embeddings.values()):
        feature_names = ec_module._feature_names[i]
        embedding_names = ec_module._embedding_names_by_table[i]
        upper_bound = _resolve_embedding_index_upper_bound(emb_module)

        for j, embedding_name in enumerate(embedding_names):
            feature_name = feature_names[j]
            feature = jt_dict[feature_name]
            indices = _validate_embedding_indices(
                feature_name=feature_name,
                embedding_name=embedding_name,
                indices=feature.values(),
                upper_bound=upper_bound,
            )
            lookup = emb_module(input=indices)
            feature_embeddings[embedding_name] = JaggedTensor(
                values=lookup,
                lengths=feature.lengths(),
                weights=feature.values() if ec_module._need_indices else None,
            )

    return feature_embeddings
