# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HSTU model feature payload for cross-process transport."""

from typing import Any

import msgspec


class HSTUFeaturesPayload(msgspec.Struct):
    """HSTU model payload — UIH + candidates + context feature groups.

    Field order and types are preserved for msgspec serialization
    compatibility.
    """

    uih_keys: list[str]
    uih_values_per_key: list[list[int]]
    uih_lengths_per_key: list[int]

    cand_keys: list[str]
    cand_values_per_key: list[list[int]]
    cand_lengths_per_key: list[int]

    ctx_keys: list[str] = []
    ctx_values_per_key: list[list[int]] = []
    full_uih_keys: list[str] = []

    user_id: str = ""
    original_candidates: list[dict[str, Any]] = []
