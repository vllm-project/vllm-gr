# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Typed configuration boundary for vLLM-GR."""

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
