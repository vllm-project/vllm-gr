# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable digest of the tokenizer vocabulary used by the offline compiler."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def compute_tokenizer_digest(tokenizer: Any) -> str:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("tokenizer must expose get_vocab() to build a constraint Artifact")
    vocab = get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ValueError("tokenizer.get_vocab() must return a non-empty mapping")

    normalized: list[tuple[str, int]] = []
    for token, token_id in vocab.items():
        if not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("tokenizer vocabulary must map strings to integer token IDs")
        normalized.append((token, token_id))
    normalized.sort(key=lambda item: (item[1], item[0]))
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
