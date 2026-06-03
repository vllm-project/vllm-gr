# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FuXi-alpha model feature payload for cross-process transport."""

from typing import Any

import msgspec


class FuXiFeaturesPayload(msgspec.Struct):
    """FuXi-alpha payload — compact list fields (no KJT / context)."""

    past_ids: list[int]
    past_ratings: list[int]
    past_timestamps: list[int]
    history_len: int

    candidate_ids: list[int]
    num_candidates: int

    user_id: str = ""
    original_candidates: list[dict[str, Any]] = []
