# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HSTU feature extraction logic.

Migrated from ``gr_schema.py`` and ``gr_input_processor.py`` — provides
the ``extract_features`` and ``extract_context_features`` helpers used by
``HSTUModelAdapter.parse_prompt()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_gr.gr_schema import (
    attach_schema_fingerprint,
)
from vllm_gr.gr_schema import extract_context_features as _extract_context_features
from vllm_gr.gr_schema import extract_features as _extract_features
from vllm_gr.gr_schema import (
    validate_context_group,
    validate_feature_group,
)


def extract_uih_features(
    records: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[list[str], list[list[int]], list[int]]:
    """Extract UIH features from user history records."""
    return _extract_features(
        records=records,
        feature_names=feature_names,
        group_name="uih",
    )


def extract_candidate_features(
    records: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[list[str], list[list[int]], list[int]]:
    """Extract candidate features from candidate records."""
    return _extract_features(
        records=records,
        feature_names=feature_names,
        group_name="cand",
    )


def extract_context_features(
    context: Mapping[str, Any],
    feature_names: list[str] | None = None,
) -> tuple[list[str], list[list[int]]]:
    """Extract context features from the context dict."""
    return _extract_context_features(context, feature_names=feature_names)


def validate_parsed_features(
    *,
    uih_keys: list[str],
    uih_values_per_key: list[list[int]],
    uih_lengths_per_key: list[int],
    cand_keys: list[str],
    cand_values_per_key: list[list[int]],
    cand_lengths_per_key: list[int],
    ctx_keys: list[str],
    ctx_values_per_key: list[list[int]],
) -> None:
    """Validate all feature groups after extraction."""
    validate_feature_group("uih", uih_keys, uih_values_per_key, uih_lengths_per_key)
    validate_feature_group("cand", cand_keys, cand_values_per_key, cand_lengths_per_key)
    validate_context_group(ctx_keys, ctx_values_per_key)


def compute_schema_fingerprint(
    *,
    uih_keys: list[str],
    cand_keys: list[str],
    ctx_keys: list[str],
) -> tuple[list[str], list[str], list[str], str]:
    """Attach a schema fingerprint to feature keys."""
    return attach_schema_fingerprint(
        uih_keys=uih_keys,
        cand_keys=cand_keys,
        ctx_keys=ctx_keys,
    )
