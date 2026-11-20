# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Typed configuration boundary for vLLM-GR."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_gr.config.models import (
        CONSTRAINT_TABLE_FORMAT,
        MAX_BEAM_WIDTH,
        MAX_CANDIDATE_WIDTH,
        MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
        SUPPORTED_BEAM_BATCH_SIZES,
        ConstraintBackendKind,
        ConstraintConfig,
        ConstraintMode,
        ConstraintTableBackendEnum,
        ConstraintTableSpec,
        GRAttentionBackend,
        GRConfig,
        NPUDecodeExecution,
        resolve_candidate_width,
    )
    from vllm_gr.config.parser import (
        GRConfigInput,
        gr_config_from_mapping,
        gr_config_to_mapping,
        load_gr_config,
        write_canonical_gr_config,
    )

__all__ = [
    "CONSTRAINT_TABLE_FORMAT",
    "MAX_BEAM_WIDTH",
    "MAX_CANDIDATE_WIDTH",
    "MULTI_TRIE_CONSTRAINT_TABLE_FORMAT",
    "SUPPORTED_BEAM_BATCH_SIZES",
    "ConstraintBackendKind",
    "ConstraintConfig",
    "ConstraintMode",
    "ConstraintTableBackendEnum",
    "ConstraintTableSpec",
    "GRAttentionBackend",
    "GRConfig",
    "NPUDecodeExecution",
    "GRConfigInput",
    "resolve_candidate_width",
    "gr_config_from_mapping",
    "gr_config_to_mapping",
    "load_gr_config",
    "write_canonical_gr_config",
]

_PARSER_EXPORTS = (
    "GRConfigInput",
    "gr_config_from_mapping",
    "gr_config_to_mapping",
    "load_gr_config",
    "write_canonical_gr_config",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    owner = ".parser" if name in _PARSER_EXPORTS else ".models"
    value = getattr(import_module(owner, __package__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
