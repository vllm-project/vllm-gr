# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Model taxonomy for vllm-gr: GR recommendation vs LLM passthrough."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional

# Registered GR adapter families (Layer 1 registry).
GR_MODEL_FAMILIES: frozenset[str] = frozenset({"hstu", "fuxi"})

# CLI / explicit family aliases that force LLM passthrough.
LLM_MODEL_FAMILY_ALIASES: frozenset[str] = frozenset({"llm", "pangu", "pangu_llm"})

# HuggingFace config.json model_type values (normalized lowercase).
LLM_MODEL_TYPES: frozenset[str] = frozenset({
    "panguembedded",
    "openpanguembedded",
})

# config.json architectures[0] values.
LLM_ARCHITECTURES: frozenset[str] = frozenset({
    "PanguEmbeddedForCausalLM",
    "OpenPanguEmbeddedForCausalLM",
})


class ModelCategory(str, Enum):
    """Runtime routing category for vllm-gr engine/worker selection."""

    GR = "gr"
    LLM = "llm"


def _normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _config_from_sources(
    local_config: Optional[Mapping[str, Any]],
    hf_overrides: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(local_config, Mapping):
        merged.update(local_config)
    if isinstance(hf_overrides, Mapping):
        merged.update(hf_overrides)
    return merged


def classify_model(
    *,
    model_family: Optional[str] = None,
    local_config: Optional[Mapping[str, Any]] = None,
    hf_overrides: Optional[Mapping[str, Any]] = None,
) -> ModelCategory:
    """Classify a model as GR recommendation or standard LLM passthrough.

    Priority:
    1. Explicit ``model_family`` LLM aliases (``llm``, ``pangu``, ...).
    2. Explicit ``model_family`` in ``GR_MODEL_FAMILIES``.
    3. ``architectures`` / ``model_type`` from merged HF config.
    4. Explicit unknown family registered in adapter registry → GR.
    5. No model path bootstrap (``model_family`` resolved to ``hstu``) → GR.
    6. Otherwise → LLM passthrough for unknown causal-LM checkpoints.
    """
    explicit = _normalize(model_family)
    if explicit in LLM_MODEL_FAMILY_ALIASES:
        return ModelCategory.LLM
    if explicit in GR_MODEL_FAMILIES:
        return ModelCategory.GR

    config = _config_from_sources(local_config, hf_overrides)

    architectures = config.get("architectures") or []
    if architectures:
        arch = architectures[0]
        if isinstance(arch, str) and arch in LLM_ARCHITECTURES:
            return ModelCategory.LLM

    model_type = _normalize(config.get("model_type") if config else None)
    if model_type in LLM_MODEL_TYPES:
        return ModelCategory.LLM
    if model_type in GR_MODEL_FAMILIES:
        return ModelCategory.GR

    if explicit is not None:
        from vllm_gr.adapters.registry import get_registered_families

        if explicit in get_registered_families():
            return ModelCategory.GR
        # Unknown explicit family (e.g. panguembedded from config.json path).
        if model_type is not None or architectures:
            return ModelCategory.LLM

    if explicit is None:
        if not config:
            return ModelCategory.GR
        if model_type in GR_MODEL_FAMILIES:
            return ModelCategory.GR
        return ModelCategory.LLM

    if explicit in GR_MODEL_FAMILIES:
        return ModelCategory.GR

    return ModelCategory.LLM


def is_gr_category(category: ModelCategory | str | None) -> bool:
    if category is None:
        return False
    if isinstance(category, ModelCategory):
        return category == ModelCategory.GR
    return str(category).lower() == ModelCategory.GR.value


def is_llm_category(category: ModelCategory | str | None) -> bool:
    if category is None:
        return False
    if isinstance(category, ModelCategory):
        return category == ModelCategory.LLM
    return str(category).lower() == ModelCategory.LLM.value
