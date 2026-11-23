# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Validate complete five-token OneRec SID outputs at benchmark boundaries."""

import re

SID_PATTERN = re.compile(r"<\|sid_begin\|><s_a_[0-9]+><s_b_[0-9]+><s_c_[0-9]+><\|sid_end\|>")


def validate_sid_text(text: str, context: str) -> None:
    if not isinstance(text, str) or SID_PATTERN.fullmatch(text) is None:
        raise RuntimeError(f"{context}: incomplete or malformed SID output: {text!r}")


def validate_sid_tokens(tokenizer, tokens: list[int], context: str) -> None:
    if len(tokens) != 5:
        raise RuntimeError(f"{context}: expected five SID tokens, got {len(tokens)}")
    pieces = tokenizer.convert_ids_to_tokens(tokens)
    if not isinstance(pieces, list) or len(pieces) != 5:
        raise RuntimeError(f"{context}: could not decode SID token IDs")
    validate_sid_text("".join(pieces), context)
