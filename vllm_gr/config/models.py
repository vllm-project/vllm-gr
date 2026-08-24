# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Typed, process-serializable vLLM-GR configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

GR_CONFIG_SCHEMA_VERSION = 1
CONSTRAINT_TABLE_FORMAT = "constraint_table_v1"
CONSTRAINT_TABLE_MAX_GENERATION_STEPS = 3
DEFAULT_BEAM_MAX_DECODE_STEPS = 3
DEFAULT_CONSTRAINT_MAX_TOP_K = 128


class GRAttentionBackend(str, Enum):
    """Internal marker used to restore the custom Beam attention backend."""

    CUSTOM = "CUSTOM"


class ConstraintTableBackend(str, Enum):
    """Storage/execution placement for an enabled constraint table."""

    CPU = "cpu"
    NPU = "npu"


@dataclass(frozen=True, slots=True)
class BeamConfig:
    """Beam resource bounds shared by Driver and Workers."""

    max_width: int | None = None
    max_decode_steps: int = DEFAULT_BEAM_MAX_DECODE_STEPS
    graph_enabled: bool = True

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_beam_graph_enabled,
            validate_beam_max_decode_steps,
            validate_beam_max_width,
        )

        object.__setattr__(self, "max_width", validate_beam_max_width(self.max_width))
        object.__setattr__(
            self,
            "max_decode_steps",
            validate_beam_max_decode_steps(self.max_decode_steps),
        )
        object.__setattr__(
            self,
            "graph_enabled",
            validate_beam_graph_enabled(self.graph_enabled),
        )

    @property
    def graph_active(self) -> bool:
        return self.graph_enabled and self.max_width is not None


@dataclass(frozen=True, slots=True)
class ConstraintTableConfig:
    """Configuration required to load a prebuilt six-table Artifact."""

    enabled: bool = False
    backend: ConstraintTableBackend = ConstraintTableBackend.CPU
    path: str | None = None
    format: str = CONSTRAINT_TABLE_FORMAT
    artifact_digest: str | None = None
    tokenizer_digest: str | None = None
    max_top_k: int = DEFAULT_CONSTRAINT_MAX_TOP_K

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_artifact_digest,
            validate_constraint_table_backend,
            validate_constraint_table_enabled,
            validate_constraint_table_format,
            validate_constraint_table_path,
            validate_max_top_k,
            validate_tokenizer_digest,
        )

        object.__setattr__(self, "enabled", validate_constraint_table_enabled(self.enabled))
        object.__setattr__(
            self,
            "backend",
            validate_constraint_table_backend(self.backend),
        )
        object.__setattr__(self, "path", validate_constraint_table_path(self.path))
        object.__setattr__(
            self,
            "format",
            validate_constraint_table_format(self.format),
        )
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
        object.__setattr__(self, "max_top_k", validate_max_top_k(self.max_top_k))

        if self.enabled:
            missing = [
                name
                for name, value in (
                    ("path", self.path),
                    ("artifact_digest", self.artifact_digest),
                    ("tokenizer_digest", self.tokenizer_digest),
                )
                if value is None
            ]
            if missing:
                raise ValueError("enabled vllm-gr constraint_table requires " + ", ".join(missing))


@dataclass(frozen=True, slots=True)
class GRConfig:
    """Validated configuration reconstructed from ``VllmConfig`` by Workers.

    The YAML path is deliberately absent. Only the Driver reads YAML; Workers
    receive this normalized value in ``additional_config['vllm_gr']``.
    """

    schema_version: int = GR_CONFIG_SCHEMA_VERSION
    beam: BeamConfig = field(default_factory=BeamConfig)
    constraint_table: ConstraintTableConfig = field(default_factory=ConstraintTableConfig)
    attention_backend: GRAttentionBackend | None = None

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_attention_backend,
            validate_schema_version,
        )

        object.__setattr__(self, "schema_version", validate_schema_version(self.schema_version))
        if not isinstance(self.beam, BeamConfig):
            raise ValueError("vllm-gr beam must be a BeamConfig")
        if not isinstance(self.constraint_table, ConstraintTableConfig):
            raise ValueError("vllm-gr constraint_table must be a ConstraintTableConfig")
        object.__setattr__(
            self,
            "attention_backend",
            validate_attention_backend(self.attention_backend),
        )
        if self.constraint_table.enabled:
            if self.beam.max_width is None:
                raise ValueError("enabled vllm-gr constraint_table requires beam.max_width")
            if self.beam.max_width > self.constraint_table.max_top_k:
                raise ValueError(
                    "vllm-gr beam.max_width must be <= constraint_table.max_top_k when "
                    "constraint_table is enabled "
                    f"(beam.max_width={self.beam.max_width}, "
                    f"constraint_table.max_top_k={self.constraint_table.max_top_k})"
                )

    @property
    def beam_graph_active(self) -> bool:
        return self.beam.graph_active

    @property
    def beam_graph_enabled(self) -> bool:
        return self.beam.graph_enabled

    @property
    def beam_max_width(self) -> int | None:
        return self.beam.max_width

    @property
    def beam_max_decode_steps(self) -> int:
        return self.beam.max_decode_steps

    @property
    def enabled(self) -> bool:
        return (
            self.beam.max_width is not None
            or self.constraint_table.enabled
            or self.attention_backend is not None
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: object | None) -> GRConfig:
        from vllm_gr.config.parser import parse_gr_config

        return parse_gr_config(vllm_config)

    def write_to_vllm_config(self, vllm_config: object) -> None:
        from vllm_gr.config.parser import write_gr_config

        write_gr_config(vllm_config, self)
