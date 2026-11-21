# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Resolve explicit boundary tokens without accepting an UNK fallback."""

from typing import Any


def boundary_token_id(tokenizer: Any, token: str | None, name: str) -> int | None:
    if token is None:
        return None
    message = f"{name} must be a valid tokenizer token: {token!r}"
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except (KeyError, ValueError) as exc:
        raise ValueError(message) from exc
    if token_id is None or isinstance(token_id, bool) or not isinstance(token_id, int):
        raise ValueError(message)
    if token_id < 0:
        raise ValueError(message)
    if token_id == getattr(tokenizer, "unk_token_id", None):
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            valid = get_vocab().get(token) == token_id
        else:
            reverse = getattr(tokenizer, "convert_ids_to_tokens", None)
            valid = (
                reverse(token_id) == token
                if callable(reverse)
                else token == getattr(tokenizer, "unk_token", None)
            )
        if not valid:
            raise ValueError(message)
    return token_id
