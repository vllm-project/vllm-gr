# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Canonical digest of tokenizer state used by constraint-table artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

_TOKENIZER_DIGEST_FORMAT = "tokenizer_digest_v1"
_SPECIAL_TOKEN_ID_ATTRIBUTES = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
    "sep_token_id",
    "cls_token_id",
    "mask_token_id",
)


def _normalize_vocab(value: object, name: str) -> list[tuple[str, int]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"tokenizer {name} must be a mapping")
    normalized: list[tuple[str, int]] = []
    for token, token_id in value.items():
        if (
            not isinstance(token, str)
            or isinstance(token_id, bool)
            or not isinstance(token_id, Integral)
        ):
            raise ValueError(f"tokenizer {name} must map strings to integer token IDs")
        normalized.append((token, int(token_id)))
    normalized.sort(key=lambda item: item[0])
    return normalized


def _normalize_optional_id(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"tokenizer {name} must be an integer or None")
    return int(value)


def _normalize_id_list(value: object, name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"tokenizer {name} must be a sequence of integer IDs")
    normalized: list[int] = []
    for item in value:
        normalized_id = _normalize_optional_id(item, name)
        if normalized_id is None:
            raise ValueError(f"tokenizer {name} must contain only integer IDs")
        normalized.append(normalized_id)
    return normalized


def _get_added_vocab(tokenizer: Any) -> object:
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        return get_added_vocab()
    added_tokens_encoder = getattr(tokenizer, "added_tokens_encoder", None)
    return {} if added_tokens_encoder is None else added_tokens_encoder


def compute_tokenizer_digest(tokenizer: Any) -> str:
    """Hash vocab, added vocab, and special-token role assignments."""

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("tokenizer must expose get_vocab() to build a constraint artifact")
    vocab = _normalize_vocab(get_vocab(), "vocab")
    if not vocab:
        raise ValueError("tokenizer vocab must be non-empty")
    added_vocab = _normalize_vocab(_get_added_vocab(tokenizer), "added vocab")

    special_token_roles = {
        attribute: _normalize_optional_id(getattr(tokenizer, attribute, None), attribute)
        for attribute in _SPECIAL_TOKEN_ID_ATTRIBUTES
    }
    payload = {
        "digest_format": _TOKENIZER_DIGEST_FORMAT,
        "vocab": vocab,
        "added_vocab": added_vocab,
        "special_token_roles": special_token_roles,
        "additional_special_token_ids": _normalize_id_list(
            getattr(tokenizer, "additional_special_tokens_ids", None),
            "additional_special_tokens_ids",
        ),
        "all_special_ids": _normalize_id_list(
            getattr(tokenizer, "all_special_ids", None),
            "all_special_ids",
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
