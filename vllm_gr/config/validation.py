# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from dataclasses import replace
from typing import cast

import regex as re

from vllm_gr.config.models import (
    CONSTRAINT_TABLE_FORMAT,
    MAX_BEAM_WIDTH,
    MAX_CANDIDATE_WIDTH,
    MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
    ConstraintBackendKind,
    ConstraintConfig,
    ConstraintMode,
    ConstraintTableSpec,
    DeploymentOptions,
    GRAttentionBackend,
    GRConfig,
    NPUDecodeExecution,
    absolute_config_path,
)
from vllm_gr.config.schema import config_default, config_fields

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_config_fields(config: object) -> None:
    """Normalize each declared field once, before checking model invariants."""
    for item, spec in config_fields(type(config)):
        validator = globals().get(spec.validator)
        if not callable(validator):
            raise ValueError(f"unknown config validator: {spec.validator}")
        object.__setattr__(config, item.name, validator(getattr(config, item.name)))


def validate_deployment(value: object) -> DeploymentOptions:
    if not isinstance(value, DeploymentOptions):
        raise ValueError("vllm-gr deployment must be DeploymentOptions")
    return value


def validate_optional_max_top_k(value: object) -> int | None:
    return None if value is None else validate_constraint_table_max_top_k(value)


def validate_beam_adapter_cache_enabled(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("beam.adapter_cache_enabled must be a boolean")
    return value


def validate_constraint_mode(value: object) -> ConstraintMode:
    if isinstance(value, ConstraintMode):
        return value
    if not isinstance(value, str):
        raise ValueError("vllm-gr constraint.mode must be a string")
    try:
        return ConstraintMode(value.lower())
    except ValueError as exc:
        raise ValueError("vllm-gr constraint.mode must be 'none' or 'table'") from exc


def validate_constraint_config(value: object) -> ConstraintConfig | None:
    """Require a complete artifact specification in table mode."""
    if value is None:
        return None
    if not isinstance(value, ConstraintConfig):
        raise ValueError("vllm-gr constraint must be a ConstraintConfig or null")
    mode = validate_constraint_mode(value.mode)
    if mode is ConstraintMode.NONE:
        if value.table is not None:
            raise ValueError("vllm-gr constraint.mode='none' forbids constraint.table")
        return ConstraintConfig(mode=mode)
    if value.table is None:
        raise ValueError("vllm-gr constraint.mode='table' requires constraint.table")
    table = value.table
    if not isinstance(table, ConstraintTableSpec):
        raise ValueError("vllm-gr constraint.table must be a ConstraintTableSpec")
    if table.path is None:
        raise ValueError("vllm-gr constraint.mode='table' requires constraint.table.path")
    if table.artifact_digest is None or table.tokenizer_digest is None:
        raise ValueError("vllm-gr constraint.mode='table' requires artifact and tokenizer digests")
    if table.max_top_k is None:
        raise ValueError("constraint.mode='table' requires constraint.table.max_top_k")
    return ConstraintConfig(mode, table if table.enabled else replace(table, enabled=True))


def validate_constraint_table_enabled(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("vllm-gr constraint_table.enabled must be a boolean")
    return value


def validate_constraint_table_backend(
    value: object,
) -> ConstraintBackendKind:
    if isinstance(value, ConstraintBackendKind):
        return value
    if not isinstance(value, str):
        raise ValueError("vllm-gr constraint_table.backend must be a string")
    try:
        return ConstraintBackendKind(value.lower())
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ConstraintBackendKind)
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
    return absolute_config_path(value, "gr.constraint_table.path")


def validate_constraint_table_format(value: object) -> str:
    if value not in (CONSTRAINT_TABLE_FORMAT, MULTI_TRIE_CONSTRAINT_TABLE_FORMAT):
        raise ValueError(
            f"unsupported vllm-gr constraint_table.format {value!r}; "
            f"expected {CONSTRAINT_TABLE_FORMAT!r} or {MULTI_TRIE_CONSTRAINT_TABLE_FORMAT!r}"
        )
    return cast(str, value)


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


def validate_candidate_width(value: object) -> int | None:
    if value is None:
        return None
    width = _validate_positive_int(value, "gr.candidate_width")
    if width > MAX_CANDIDATE_WIDTH:
        raise ValueError(
            f"vllm-gr gr.candidate_width must be <= {MAX_CANDIDATE_WIDTH}, got {width}"
        )
    return width


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


def validate_prefill_flash_attn_version(value: object) -> int | str | None:
    if value is None or (isinstance(value, str) and value == "auto"):
        return value
    if type(value) is int and value in (2, 3):
        return value
    raise ValueError("beam.prefill_flash_attn_version must be null, 'auto', 2 or 3")


def validate_enable_short_suffix_attention(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("beam.enable_short_suffix_attention must be a boolean")
    return value


def validate_beam_npu_unconstrained_topk_fast_path(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("gr.npu_unconstrained_topk_fast_path must be a boolean")
    return value


def validate_beam_sid_stage_vocab(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("beam.sid_stage_vocab must be a boolean")
    return value


def validate_beam_graph_enabled(value: object) -> bool:
    if value is None:
        return cast(bool, config_default(GRConfig, "beam_graph_enabled"))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    raise ValueError("vllm-gr beam_graph_enabled must be a boolean")


def validate_beam_graph_batch_size(value: object) -> int:
    """Validate the exact logical batch shape captured for Beam graph replay."""

    batch_size = _validate_positive_int(value, "beam_graph_batch_size")
    if batch_size > 128:
        raise ValueError(f"vllm-gr beam_graph_batch_size must be <= 128, got {batch_size}")
    return batch_size


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


def validate_beam_graph_widths(value: object) -> tuple[int, ...] | None:
    """Normalize explicit NPU Decode widths to prewarm at startup."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("beam.graph_widths must be a nonempty list of integers or null")
    widths = tuple(validate_beam_width(width) for width in value)
    if len(set(widths)) != len(widths):
        raise ValueError("beam.graph_widths must not contain duplicates")
    return tuple(sorted(widths))


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


def validate_beam_max_decode_steps(value: object) -> int:
    return _validate_positive_int(
        value if value is not None else config_default(GRConfig, "beam_max_decode_steps"),
        "beam_max_decode_steps",
    )
