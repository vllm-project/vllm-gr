# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GR_CONFIG_SCHEMA_VERSION = 1
CONSTRAINT_TABLE_FORMAT = "constraint_table_v1"


class GRAttentionBackend(str, Enum):
    CUSTOM = "CUSTOM"


class ConstraintBackend(str, Enum):
    TRIE = "trie"
    CONSTRAINT_TABLE = "constraint_table"


class ConstraintTableBackendEnum(str, Enum):
    """Execution placement requested for a canonical constraint table."""

    CPU = "cpu"
    NPU = "npu"


# Default values for beam KV pool sizing — also used by parser.py (to decide
# whether a value is the default and can be omitted from additional_config) and
# by validation.py (as fallbacks).  Keep all three sites in sync.
_BEAM_MAX_DECODE_STEPS_DEFAULT: int = 3  # OneRec SID: max_tokens=5 minus begin/end


@dataclass(frozen=True, slots=True)
class ConstraintTableConfig:
    """Configuration for a prebuilt canonical constraint-table artifact."""

    max_top_k: int
    enabled: bool = False
    backend: ConstraintTableBackendEnum = ConstraintTableBackendEnum.CPU
    path: str | None = None
    format: str = CONSTRAINT_TABLE_FORMAT
    artifact_digest: str | None = None
    tokenizer_digest: str | None = None

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_artifact_digest,
            validate_constraint_table_backend,
            validate_constraint_table_enabled,
            validate_constraint_table_format,
            validate_constraint_table_max_top_k,
            validate_constraint_table_path,
            validate_tokenizer_digest,
        )

        object.__setattr__(self, "enabled", validate_constraint_table_enabled(self.enabled))
        object.__setattr__(self, "backend", validate_constraint_table_backend(self.backend))
        object.__setattr__(self, "path", validate_constraint_table_path(self.path))
        object.__setattr__(self, "format", validate_constraint_table_format(self.format))
        object.__setattr__(
            self,
            "artifact_digest",
            validate_artifact_digest(self.artifact_digest),
        )
        object.__setattr__(
            self,
            "tokenizer_digest",
            validate_tokenizer_digest(self.tokenizer_digest),
        )
        object.__setattr__(
            self,
            "max_top_k",
            validate_constraint_table_max_top_k(self.max_top_k),
        )

        if self.enabled:
            missing = [
                name
                for name, value in (
                    ("path", self.path),
                    ("format", self.format),
                    ("artifact_digest", self.artifact_digest),
                    ("tokenizer_digest", self.tokenizer_digest),
                )
                if value is None
            ]
            if missing:
                raise ValueError("enabled vllm-gr constraint_table requires " + ", ".join(missing))


class NPUDecodeExecution(str, Enum):
    """Execution mode for the NPU incremental beam decode branch."""

    CASCADE = "cascade"
    XLLM = "xllm"


@dataclass(frozen=True)
class GRConfig:
    """Validated vLLM-GR configuration reconstructed in each process."""

    catalog_path: str | None = None
    constraint_backend: ConstraintBackend = ConstraintBackend.TRIE
    attention_backend: GRAttentionBackend | None = None
    vocab_offset: int | None = None
    vocab_slice_size: int | None = None
    # Beam-decode FULL CUDA graph (startup dummy capture, runtime replay-only).
    # beam_max_width is the single upper bound on beam width: it sizes the
    # worker's persistent beam KV pool AND the captured CUDA graph shape
    # (num_tokens = width). None means beam decode is disabled — no pool
    # allocated, no graph captured. When beam_graph_enabled is True and
    # beam_max_width is set, a FULL graph is captured at startup; requests
    # with beam_width <= beam_max_width are served (graph replay for the
    # exact-width case, eager otherwise).
    beam_graph_enabled: bool = True
    beam_max_width: int | None = None
    beam_max_decode_steps: int = _BEAM_MAX_DECODE_STEPS_DEFAULT
    # None = unset: the runtime picks per-platform defaults (xllm on NPU when
    # the operator library is present, cascade otherwise and on CUDA).
    npu_decode_execution: NPUDecodeExecution | None = None
    # Canonical schema-v1 state. Keep these fields after every legacy field so
    # positional construction of the pre-schema dataclass remains compatible.
    schema_version: int = GR_CONFIG_SCHEMA_VERSION
    constraint_table: ConstraintTableConfig | None = None

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_attention_backend,
            validate_beam_graph_enabled,
            validate_beam_max_decode_steps,
            validate_beam_max_width,
            validate_catalog_path,
            validate_constraint_backend,
            validate_npu_decode_execution,
            validate_schema_version,
            validate_vocab_offset,
            validate_vocab_slice_size,
        )

        object.__setattr__(self, "catalog_path", validate_catalog_path(self.catalog_path))
        object.__setattr__(
            self,
            "constraint_backend",
            validate_constraint_backend(self.constraint_backend),
        )
        object.__setattr__(
            self,
            "attention_backend",
            validate_attention_backend(self.attention_backend),
        )
        object.__setattr__(self, "vocab_offset", validate_vocab_offset(self.vocab_offset))
        object.__setattr__(
            self, "vocab_slice_size", validate_vocab_slice_size(self.vocab_slice_size)
        )
        object.__setattr__(
            self,
            "beam_graph_enabled",
            validate_beam_graph_enabled(self.beam_graph_enabled),
        )
        object.__setattr__(self, "beam_max_width", validate_beam_max_width(self.beam_max_width))
        object.__setattr__(
            self,
            "beam_max_decode_steps",
            validate_beam_max_decode_steps(self.beam_max_decode_steps),
        )
        object.__setattr__(
            self,
            "npu_decode_execution",
            validate_npu_decode_execution(self.npu_decode_execution),
        )
        object.__setattr__(self, "schema_version", validate_schema_version(self.schema_version))
        if self.constraint_table is not None and not isinstance(
            self.constraint_table, ConstraintTableConfig
        ):
            raise ValueError("vllm-gr constraint_table must be a ConstraintTableConfig or None")
        if (
            self.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
            and self.catalog_path is None
        ):
            raise ValueError("vllm-gr constraint_backend='constraint_table' requires catalog_path")

    @property
    def beam_graph_active(self) -> bool:
        """True iff the startup beam-graph capture should run."""
        return self.beam_graph_enabled and self.beam_max_width is not None

    @property
    def enabled(self) -> bool:
        return (
            self.catalog_path is not None
            or self.constraint_backend is not ConstraintBackend.TRIE
            or self.attention_backend is not None
            or self.vocab_offset is not None
            or self.vocab_slice_size is not None
            or self.npu_decode_execution is not None
            or (self.constraint_table is not None and self.constraint_table.enabled)
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: object | None) -> GRConfig:
        from vllm_gr.config.parser import parse_gr_config

        return parse_gr_config(vllm_config)

    def write_to_vllm_config(self, vllm_config: object) -> None:
        from vllm_gr.config.parser import write_gr_config

        write_gr_config(vllm_config, self)
