# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HSTU KJT (KeyedJaggedTensor) builder.

Migrated from ``gr_runner_mixin.py`` — assembles KJT tensors for UIH and
candidate features from a batch of decoded feature dicts.
"""

from __future__ import annotations

from typing import Any

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _CompatKeyedJaggedTensor:
    """Fallback KJT shim for lightweight test stubs."""

    def __init__(
        self,
        *,
        keys: list[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        self._keys = list(keys)
        self._values = values
        self._lengths = lengths

    def keys(self) -> list[str]:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def lengths(self) -> torch.Tensor:
        return self._lengths


def _concat_1d_int_tensors(
    tensor_parts: list[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not tensor_parts:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(tensor_parts, dim=0)


def _build_kjt(
    *,
    keys: list[str],
    values: torch.Tensor,
    lengths: torch.Tensor,
):
    factory = getattr(KeyedJaggedTensor, "from_lengths_sync", None)
    if callable(factory):
        return factory(keys=keys, values=values, lengths=lengths)

    try:
        return KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
    except Exception:
        return _CompatKeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
        )


def build_hstu_model_kwargs(
    batch_features: list[tuple[str, dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    """Build ``{uih_features: KJT, candidates_features: KJT}`` for HSTU.

    Parameters
    ----------
    batch_features:
        List of ``(req_id, decoded_features)`` tuples.  All entries must
        share the same schema fingerprint.
    device:
        Target device for the output tensors (typically ``cpu``).
    """
    if not batch_features:
        return {}

    _req_id_0, features_0 = batch_features[0]
    uih_keys: list[str] = features_0["uih_keys"]
    cand_keys: list[str] = features_0["cand_keys"]
    ctx_keys: list[str] = features_0["ctx_keys"]
    full_uih_keys: list[str] = features_0.get("full_uih_keys", uih_keys + ctx_keys)

    num_uih_keys = len(uih_keys)
    num_cand_keys = len(cand_keys)
    num_ctx_keys = len(ctx_keys)

    uih_values_by_key: list[list[torch.Tensor]] = [[] for _ in range(num_uih_keys)]
    uih_lengths_by_key: list[list[int]] = [[] for _ in range(num_uih_keys)]
    cand_values_by_key: list[list[torch.Tensor]] = [[] for _ in range(num_cand_keys)]
    cand_lengths_by_key: list[list[int]] = [[] for _ in range(num_cand_keys)]
    ctx_values_by_key: list[list[torch.Tensor]] = [[] for _ in range(num_ctx_keys)]

    for _req_id, features in batch_features:
        uih_lengths = features["uih_lengths_per_key"]
        cand_lengths = features["cand_lengths_per_key"]

        for key_idx in range(num_uih_keys):
            values = features["uih_values_per_key"][key_idx].to(device)
            uih_values_by_key[key_idx].append(values)
            uih_lengths_by_key[key_idx].append(int(uih_lengths[key_idx].item()))

        for key_idx in range(num_cand_keys):
            values = features["cand_values_per_key"][key_idx].to(device)
            cand_values_by_key[key_idx].append(values)
            cand_lengths_by_key[key_idx].append(int(cand_lengths[key_idx].item()))

        for key_idx in range(num_ctx_keys):
            values = features["ctx_values_per_key"][key_idx].reshape(-1).to(device)
            ctx_values_by_key[key_idx].append(values)

    uih_key_to_idx = {key: idx for idx, key in enumerate(uih_keys)}
    ctx_key_to_idx = {key: idx for idx, key in enumerate(ctx_keys)}
    all_uih_value_parts: list[torch.Tensor] = []
    all_uih_lengths: list[int] = []
    batch_size = len(batch_features)

    for key in full_uih_keys:
        if key in uih_key_to_idx:
            key_idx = uih_key_to_idx[key]
            values_per_request = uih_values_by_key[key_idx]
            if values_per_request:
                all_uih_value_parts.append(torch.cat(values_per_request, dim=0))
            all_uih_lengths.extend(uih_lengths_by_key[key_idx])
            continue

        if key in ctx_key_to_idx:
            key_idx = ctx_key_to_idx[key]
            values_per_request = ctx_values_by_key[key_idx]
            if values_per_request:
                all_uih_value_parts.append(torch.cat(values_per_request, dim=0))
            all_uih_lengths.extend([1] * batch_size)
            continue

        raise ValueError(f"Full UIH schema contains unknown key not present in GR payload: {key!r}")

    uih_values = _concat_1d_int_tensors(all_uih_value_parts, device=device)
    uih_kjt = _build_kjt(
        keys=full_uih_keys,
        values=uih_values,
        lengths=torch.tensor(all_uih_lengths, dtype=torch.int64, device=device),
    )

    cand_values = _concat_1d_int_tensors(
        [
            _concat_1d_int_tensors(values_per_key, device=device)
            for values_per_key in cand_values_by_key
        ],
        device=device,
    )
    cand_lengths = torch.tensor(
        [length for lengths_per_key in cand_lengths_by_key for length in lengths_per_key],
        dtype=torch.int64,
        device=device,
    )
    cand_kjt = _build_kjt(
        keys=cand_keys,
        values=cand_values,
        lengths=cand_lengths,
    )

    return {
        "uih_features": uih_kjt,
        "candidates_features": cand_kjt,
    }
