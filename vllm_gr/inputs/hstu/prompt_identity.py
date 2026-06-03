# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HSTU prompt identity helpers for vLLM native prefix caching.

This module lives near the input pipeline on purpose: it builds
request-side synthetic ``prompt_token_ids`` and ``cache_salt`` so HSTU
requests can participate in vLLM's native prefix-cache hashing without
introducing a separate HSTU-owned cache subsystem.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

_PROMPT_IDENTITY_VERSION = "hstu-prefix-v1"
# vLLM runtime paths materialize prompt token ids into int32-backed
# tensors/arrays, so HSTU synthetic prompt identity tokens must stay within
# the positive int32 range.
_TOKEN_MASK = (1 << 31) - 1


def _stable_digest(payload: Any, *, digest_size: int = 16) -> bytes:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(serialized, digest_size=digest_size).digest()


def _stable_token_id(payload: Any) -> int:
    token_id = int.from_bytes(_stable_digest(payload, digest_size=8), "big")
    token_id &= _TOKEN_MASK
    return token_id or 1


def _values_by_key(
    keys: Sequence[str],
    values_per_key: Sequence[Sequence[int]],
) -> dict[str, list[int]]:
    return {str(key): [int(value) for value in values] for key, values in zip(keys, values_per_key)}


def _segment_tokens_from_feature_rows(
    *,
    segment: str,
    feature_keys: Sequence[str],
    values_per_key: Sequence[Sequence[int]],
    num_rows: int,
) -> list[int]:
    if num_rows <= 0:
        return []

    if not feature_keys:
        return [
            _stable_token_id({"segment": segment, "kind": "empty", "row": row_idx})
            for row_idx in range(num_rows)
        ]

    values_by_key = _values_by_key(feature_keys, values_per_key)
    tokens: list[int] = []
    for row_idx in range(num_rows):
        features: list[list[int | str]] = []
        for key in feature_keys:
            series = values_by_key.get(str(key), [])
            value = int(series[row_idx]) if row_idx < len(series) else 0
            features.append([str(key), value])
        tokens.append(
            _stable_token_id(
                {
                    "segment": segment,
                    "features": features,
                }
            )
        )
    return tokens


def _build_contextual_tokens(
    *,
    ctx_keys: Sequence[str],
    ctx_values_per_key: Sequence[Sequence[int]],
    history_len: int,
    contextual_feature_to_max_length: Mapping[str, int],
    contextual_feature_to_min_uih_length: Mapping[str, int],
) -> list[int]:
    if not contextual_feature_to_max_length:
        return []

    values_by_key = _values_by_key(ctx_keys, ctx_values_per_key)
    tokens: list[int] = []
    for feature_name, max_len in contextual_feature_to_max_length.items():
        min_uih_len = int(contextual_feature_to_min_uih_length.get(feature_name, 0))
        is_active = history_len >= min_uih_len
        feature_values = values_by_key.get(str(feature_name), [])
        for slot_idx in range(int(max_len)):
            has_value = is_active and slot_idx < len(feature_values)
            if has_value:
                slot_state = "value"
                slot_value = int(feature_values[slot_idx])
            elif is_active:
                slot_state = "pad"
                slot_value = 0
            else:
                slot_state = "inactive"
                slot_value = 0
            tokens.append(
                _stable_token_id(
                    {
                        "segment": "contextual",
                        "feature": str(feature_name),
                        "slot": slot_idx,
                        "state": slot_state,
                        "value": slot_value,
                    }
                )
            )
    return tokens


def build_hstu_prompt_token_ids(
    *,
    uih_keys: Sequence[str],
    uih_values_per_key: Sequence[Sequence[int]],
    cand_keys: Sequence[str],
    cand_values_per_key: Sequence[Sequence[int]],
    ctx_keys: Sequence[str],
    ctx_values_per_key: Sequence[Sequence[int]],
    history_len: int,
    num_candidates: int,
    contextual_feature_to_max_length: Mapping[str, int] | None = None,
    contextual_feature_to_min_uih_length: Mapping[str, int] | None = None,
) -> list[int]:
    contextual_feature_to_max_length = contextual_feature_to_max_length or {}
    contextual_feature_to_min_uih_length = contextual_feature_to_min_uih_length or {}

    context_tokens = _build_contextual_tokens(
        ctx_keys=ctx_keys,
        ctx_values_per_key=ctx_values_per_key,
        history_len=history_len,
        contextual_feature_to_max_length=contextual_feature_to_max_length,
        contextual_feature_to_min_uih_length=contextual_feature_to_min_uih_length,
    )
    history_tokens = _segment_tokens_from_feature_rows(
        segment="history",
        feature_keys=uih_keys,
        values_per_key=uih_values_per_key,
        num_rows=history_len,
    )
    candidate_tokens = _segment_tokens_from_feature_rows(
        segment="candidate",
        feature_keys=cand_keys,
        values_per_key=cand_values_per_key,
        num_rows=num_candidates,
    )

    prompt_token_ids = context_tokens + history_tokens + candidate_tokens
    return prompt_token_ids or [0]


def build_hstu_cache_salt(
    *,
    schema_fingerprint: str,
    contextual_feature_to_max_length: Mapping[str, int] | None = None,
    contextual_feature_to_min_uih_length: Mapping[str, int] | None = None,
    primary_id_feature_name: str,
    model_type: str = "hstu",
) -> str:
    payload = {
        "version": _PROMPT_IDENTITY_VERSION,
        "model_type": str(model_type),
        "schema_fingerprint": str(schema_fingerprint),
        "contextual_feature_to_max_length": [
            [str(key), int(value)]
            for key, value in (contextual_feature_to_max_length or {}).items()
        ],
        "contextual_feature_to_min_uih_length": [
            [str(key), int(value)]
            for key, value in (contextual_feature_to_min_uih_length or {}).items()
        ],
        "primary_id_feature_name": str(primary_id_feature_name),
    }
    digest = _stable_digest(payload, digest_size=12).hex()
    return f"{_PROMPT_IDENTITY_VERSION}:{digest}"


__all__ = [
    "build_hstu_cache_salt",
    "build_hstu_prompt_token_ids",
]
