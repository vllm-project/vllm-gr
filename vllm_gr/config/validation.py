# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re

from vllm_gr.config.models import (
    CONSTRAINT_TABLE_FORMAT,
    GR_CONFIG_SCHEMA_VERSION,
    _BEAM_MAX_DECODE_STEPS_DEFAULT,
    ConstraintTableBackend,
    GRAttentionBackend,
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_schema_version(value: object) -> int:
    if isinstance(value, bool) or value != GR_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported vllm-gr schema_version {value!r}; "
            f"expected {GR_CONFIG_SCHEMA_VERSION}"
        )
    return GR_CONFIG_SCHEMA_VERSION


def validate_constraint_table_enabled(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("vllm-gr constraint_table.enabled must be a boolean")
    return value


def validate_constraint_table_backend(value: object) -> ConstraintTableBackend:
    if isinstance(value, ConstraintTableBackend):
        return value
    if not isinstance(value, str):
        raise ValueError("vllm-gr constraint_table.backend must be a string")
    try:
        return ConstraintTableBackend(value.lower())
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ConstraintTableBackend)
        raise ValueError(
            f"unsupported vllm-gr constraint_table.backend {value!r}; supported: {supported}"
        ) from exc


def validate_constraint_table_path(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("vllm-gr constraint_table.path must be a string or null")
    if not value.strip():
        raise ValueError("vllm-gr constraint_table.path must not be empty")
    return value


def validate_constraint_table_format(value: object) -> str:
    if value != CONSTRAINT_TABLE_FORMAT:
        raise ValueError(
            f"unsupported vllm-gr constraint_table.format {value!r}; "
            f"expected {CONSTRAINT_TABLE_FORMAT!r}"
        )
    return CONSTRAINT_TABLE_FORMAT


def _validate_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"vllm-gr {name} must be 'sha256:' followed by 64 lowercase hex digits")
    return value


def validate_artifact_digest(value: object) -> str | None:
    return _validate_sha256(value, "constraint_table.artifact_digest")


def validate_tokenizer_digest(value: object) -> str | None:
    return _validate_sha256(value, "constraint_table.tokenizer_digest")


def validate_attention_backend(value: object) -> GRAttentionBackend | None:
    if value is None:
        return None
    if isinstance(value, GRAttentionBackend):
        return value
    if not isinstance(value, str):
        raise ValueError("vllm-gr attention_backend must be a string or null")
    try:
        return GRAttentionBackend(value.upper())
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in GRAttentionBackend)
        raise ValueError(
            f"unsupported vllm-gr attention_backend {value!r}; supported: {supported}"
        ) from exc


def validate_beam_graph_enabled(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("vllm-gr beam.graph_enabled must be a boolean")
    return value


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"vllm-gr {name} must be an integer")
    if value <= 0:
        raise ValueError(f"vllm-gr {name} must be a positive integer")
    return value


def validate_beam_max_width(value: object) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, "beam.max_width")


def validate_beam_max_decode_steps(value: object) -> int:
    return _validate_positive_int(
        value if value is not None else _BEAM_MAX_DECODE_STEPS_DEFAULT,
        "beam.max_decode_steps",
    )


def validate_max_top_k(value: object) -> int:
    normalized = _validate_positive_int(value, "constraint_table.max_top_k")
    if normalized > 256:
        raise ValueError("vllm-gr constraint_table.max_top_k must be <= 256")
    return normalized
