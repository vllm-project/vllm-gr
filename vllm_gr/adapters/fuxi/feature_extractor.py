# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Feature extraction helpers for FuXi-alpha prompts."""

from __future__ import annotations

from typing import Any

FUXI_SCHEMA_FINGERPRINT = "fuxi:v1"


def extract_fuxi_history_features(
    user_history: list[dict[str, Any]],
    *,
    item_id_key: str,
    rating_key: str,
    ts_key: str,
    use_rating: bool,
    use_timestamp: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Flatten ``user_history`` into parallel id/rating/timestamp lists."""
    past_ids: list[int] = []
    past_ratings: list[int] = []
    past_timestamps: list[int] = []

    for record in user_history:
        if item_id_key not in record:
            raise ValueError(
                f"Each user_history record must contain {item_id_key!r}; "
                f"got keys={list(record.keys())!r}"
            )
        past_ids.append(int(record[item_id_key]))
        past_ratings.append(int(record.get(rating_key, 0)) if use_rating else 0)
        past_timestamps.append(int(record.get(ts_key, 0)) if use_timestamp else 0)

    return past_ids, past_ratings, past_timestamps


def extract_fuxi_candidate_ids(
    candidates: list[dict[str, Any]],
    *,
    item_id_key: str,
) -> list[int]:
    """Extract candidate item ids from prompt candidate records."""
    candidate_ids: list[int] = []
    for record in candidates:
        if item_id_key not in record:
            raise ValueError(
                f"Each candidate record must contain {item_id_key!r}; "
                f"got keys={list(record.keys())!r}"
            )
        candidate_ids.append(int(record[item_id_key]))
    return candidate_ids
