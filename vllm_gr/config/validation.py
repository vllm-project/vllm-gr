# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_gr.config.models import GRAttentionBackend


def validate_catalog_path(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("vllm-gr catalog_path must be a string or null")
    if not value.strip():
        raise ValueError("vllm-gr catalog_path must not be empty")
    return value


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


def validate_vocab_offset(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"vllm-gr vocab_offset must be a non-negative integer or null, got {value!r}"
        )
    try:
        val = int(value)
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
        val = int(value)
        if val <= 0:
            raise ValueError("vllm-gr vocab_slice_size must be positive")
        return val
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"vllm-gr vocab_slice_size must be a positive integer or null, got {value!r}"
        ) from exc
