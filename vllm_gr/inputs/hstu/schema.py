# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Schema validation helpers for GR payloads.

These helpers stay independent from the vLLM runtime so they can be reused by
both the request-side input processor and worker-side payload decoding, and be
tested without the full engine installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def attach_schema_fingerprint(
    *,
    uih_keys: list[str],
    cand_keys: list[str],
    ctx_keys: list[str],
) -> tuple[list[str], list[str], list[str], str]:
    fingerprint = build_schema_fingerprint(
        uih_keys=uih_keys,
        cand_keys=cand_keys,
        ctx_keys=ctx_keys,
    )
    return uih_keys, cand_keys, ctx_keys, fingerprint


def build_schema_fingerprint(
    *,
    uih_keys: list[str],
    cand_keys: list[str],
    ctx_keys: list[str],
) -> str:
    return (
        f"uih:{','.join(uih_keys)}|"
        f"cand:{','.join(cand_keys)}|"
        f"ctx:{','.join(ctx_keys)}"
    )


def validate_feature_group(
    group_name: str,
    keys: list[str],
    values_per_key: list[list[int]],
    lengths_per_key: list[int],
) -> None:
    if len(keys) != len(values_per_key) or len(keys) != len(lengths_per_key):
        raise ValueError(
            f"{group_name} schema is inconsistent: "
            f"len(keys)={len(keys)}, "
            f"len(values_per_key)={len(values_per_key)}, "
            f"len(lengths_per_key)={len(lengths_per_key)}"
        )

    for key, values, length in zip(keys, values_per_key, lengths_per_key):
        if int(length) != len(values):
            raise ValueError(
                f"{group_name} feature {key!r} has length metadata {length} "
                f"but {len(values)} values"
            )

    if values_per_key:
        expected_size = len(values_per_key[0])
        mismatched_keys = [
            key
            for key, values in zip(keys, values_per_key)
            if len(values) != expected_size
        ]
        if mismatched_keys:
            raise ValueError(
                f"{group_name} features must all contain the same number of records; "
                f"expected {expected_size}, mismatched keys={mismatched_keys}"
            )


def validate_context_group(
    ctx_keys: list[str],
    ctx_values_per_key: list[list[int]],
) -> None:
    if len(ctx_keys) != len(ctx_values_per_key):
        raise ValueError(
            f"context schema is inconsistent: len(ctx_keys)={len(ctx_keys)}, "
            f"len(ctx_values_per_key)={len(ctx_values_per_key)}"
        )

    for key, values in zip(ctx_keys, ctx_values_per_key):
        if len(values) != 1:
            raise ValueError(
                f"context feature {key!r} must contain exactly one value, "
                f"got {len(values)}"
            )


def extract_context_features(
    context: Mapping[str, Any],
    feature_names: list[str] | None = None,
) -> tuple[list[str], list[list[int]]]:
    if feature_names is None:
        ctx_keys = sorted(context.keys())
    else:
        ctx_keys = list(feature_names)
        missing_keys = [key for key in ctx_keys if key not in context]
        if missing_keys:
            raise ValueError(
                "context feature records do not match configured schema: "
                + "; ".join(f"{key} missing in context" for key in missing_keys)
            )
        unexpected_keys = sorted(set(context.keys()) - set(ctx_keys))
        if unexpected_keys:
            raise ValueError(
                "context contains unexpected keys for configured schema: "
                f"{unexpected_keys!r}"
            )
    try:
        ctx_values_per_key = [[int(context[k])] for k in ctx_keys]
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Context values must be convertible to int. "
            f"Got context={context!r}: {exc}"
        ) from exc
    return ctx_keys, ctx_values_per_key


def extract_features(
    records: list[dict[str, Any]],
    feature_names: list[str],
    *,
    group_name: str,
) -> tuple[list[str], list[list[int]], list[int]]:
    """Extract feature lists from records with strict schema validation."""
    if not feature_names:
        if not records:
            return [], [], []
        raise ValueError(
            f"{group_name} feature_names must be configured when records are present"
        )

    if not records:
        return (
            list(feature_names),
            [[] for _ in feature_names],
            [0 for _ in feature_names],
        )

    missing_features: list[str] = []
    for key in feature_names:
        missing_record_indexes = [
            str(idx)
            for idx, record in enumerate(records)
            if key not in record
        ]
        if missing_record_indexes:
            missing_features.append(
                f"{key} missing in records [{', '.join(missing_record_indexes)}]"
            )

    if missing_features:
        raise ValueError(
            f"{group_name} feature records do not match configured schema: "
            + "; ".join(missing_features)
        )

    values_per_key: list[list[int]] = []
    lengths_per_key: list[int] = []
    for key in feature_names:
        try:
            values = [int(record[key]) for record in records]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{group_name} feature {key!r} must be convertible to int: {exc}"
            ) from exc
        values_per_key.append(values)
        lengths_per_key.append(len(values))

    return list(feature_names), values_per_key, lengths_per_key
