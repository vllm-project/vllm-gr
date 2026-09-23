# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import regex as re

from vllm_gr.config.models import (
    _BEAM_MAX_DECODE_STEPS_DEFAULT,
    CONSTRAINT_TABLE_FORMAT,
    GR_CONFIG_SCHEMA_VERSION,
    MAX_BEAM_WIDTH,
    BeamExecutionMode,
    ConstraintBackend,
    ConstraintTableBackendEnum,
    GRAttentionBackend,
    NPUDecodeExecution,
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_beam_execution_mode(value: object) -> BeamExecutionMode:
    if isinstance(value, str):
        try:
            return BeamExecutionMode(value)
        except ValueError:
            pass
    raise ValueError("vllm-gr beam.execution_mode must be 'legacy' or 'v1'")


def validate_schema_version(value: object) -> int:
    if type(value) is not int or value != GR_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported vllm-gr schema_version {value!r}; "
            f"expected integer {GR_CONFIG_SCHEMA_VERSION}"
        )
    return GR_CONFIG_SCHEMA_VERSION


def validate_constraint_table_enabled(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("vllm-gr constraint_table.enabled must be a boolean")
    return value


def validate_constraint_table_backend(
    value: object,
) -> ConstraintTableBackendEnum:
    if isinstance(value, ConstraintTableBackendEnum):
        return value
    if not isinstance(value, str):
        raise ValueError("vllm-gr constraint_table.backend must be a string")
    try:
        return ConstraintTableBackendEnum(value.lower())
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ConstraintTableBackendEnum)
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
    if not Path(value).is_absolute():
        raise ValueError(
            "vllm-gr constraint_table.path must be absolute; relative paths "
            "are supported only while loading a YAML config"
        )
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


def validate_constraint_table_max_top_k(value: object) -> int:
    return _validate_positive_int(value, "constraint_table.max_top_k")


def validate_catalog_path(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("vllm-gr catalog_path must be a string or null")
    if not value.strip():
        raise ValueError("vllm-gr catalog_path must not be empty")
    return value


def validate_constraint_backend(value: object) -> ConstraintBackend:
    if isinstance(value, ConstraintBackend):
        return value
    if not isinstance(value, str):
        raise ValueError("vllm-gr constraint_backend must be a string")
    try:
        return ConstraintBackend(value.lower())
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ConstraintBackend)
        raise ValueError(
            f"unsupported vllm-gr constraint_backend {value!r}; supported: {supported}"
        ) from exc


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


def validate_npu_decode_execution(value: object) -> NPUDecodeExecution | None:
    if value is None:
        return None
    if isinstance(value, NPUDecodeExecution):
        return value
    if not isinstance(value, str):
        raise ValueError("vllm-gr npu_decode_execution must be a string")
    try:
        return NPUDecodeExecution(value.lower())
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in NPUDecodeExecution)
        raise ValueError(
            f"unsupported vllm-gr npu_decode_execution {value!r}; supported: {supported}"
        ) from exc


def validate_vocab_offset(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"vllm-gr vocab_offset must be a non-negative integer or null, got {value!r}"
        )
    try:
        val: int = int(cast(Any, value))
        if val < 0:
            raise ValueError("vllm-gr vocab_offset must be non-negative")
        return val
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"vllm-gr vocab_offset must be a non-negative integer or null, got {value!r}"
        ) from exc


def validate_vocab_slice_size(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"vllm-gr vocab_slice_size must be a positive integer or null, got {value!r}"
        )
    try:
        val: int = int(cast(Any, value))
        if val <= 0:
            raise ValueError("vllm-gr vocab_slice_size must be positive")
        return val
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"vllm-gr vocab_slice_size must be a positive integer or null, got {value!r}"
        ) from exc


def validate_beam_graph_enabled(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    raise ValueError("vllm-gr beam_graph_enabled must be a boolean")


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"vllm-gr {name} must be an integer")
    if value <= 0:
        raise ValueError(f"vllm-gr {name} must be a positive integer")
    return value


def validate_beam_max_width(value: object) -> int | None:
    if value is None:
        return None
    width = _validate_positive_int(value, "beam_max_width")
    if width > MAX_BEAM_WIDTH:
        raise ValueError(f"vllm-gr beam_max_width must be <= {MAX_BEAM_WIDTH}, got {width}")
    return width


def validate_beam_max_batch_size(value: object) -> int:
    batch_size = _validate_positive_int(value, "beam_max_batch_size")
    if batch_size > 128:
        raise ValueError(
            f"vllm-gr beam_max_batch_size must be <= 128, got {batch_size}"
        )
    return batch_size


def validate_beam_width(value: object) -> int:
    width = _validate_positive_int(value, "beam_width")
    if width > MAX_BEAM_WIDTH:
        raise ValueError(f"vllm-gr beam_width must be <= {MAX_BEAM_WIDTH}, got {width}")
    return width


def validate_beam_worker_decision(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    raise ValueError("vllm-gr beam_worker_decision must be a boolean or null")


def validate_beam_engine_driven(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    raise ValueError("vllm-gr beam_engine_driven must be a boolean or null")


def validate_beam_max_decode_steps(value: object) -> int:
    return _validate_positive_int(
        value if value is not None else _BEAM_MAX_DECODE_STEPS_DEFAULT,
        "beam_max_decode_steps",
    )
