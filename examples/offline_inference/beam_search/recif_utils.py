# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Application-side RECIF SID/token conversion helpers."""

VOCAB = 8192


def sid_to_levels(value: int) -> tuple[int, int, int]:
    value = int(value)
    return value & 0x1FFF, (value >> 14) & 0x1FFF, (value >> 28) & 0x1FFF


def levels_to_sid(first: int, second: int, third: int) -> int:
    return int(first) | (int(second) << 14) | (int(third) << 28)


def build_prefix(history_sids: list[int]) -> list[int]:
    tokens: list[int] = []
    for sid in history_sids:
        first, second, third = sid_to_levels(sid)
        tokens.extend([0, first, second + VOCAB, third + 2 * VOCAB])
    tokens.append(0)
    return tokens


def generated_tokens_to_sid(tokens: list[int]) -> int:
    if len(tokens) != 3:
        raise ValueError("RECIF output must contain exactly three tokens")
    return levels_to_sid(tokens[0], tokens[1] - VOCAB, tokens[2] - 2 * VOCAB)
