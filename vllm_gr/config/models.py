# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GRAttentionBackend(str, Enum):
    CUSTOM = "CUSTOM"


class ConstraintBackend(str, Enum):
    TRIE = "trie"
    CONSTRAINT_TABLE = "constraint_table"


class BeamControlMode(str, Enum):
    SERVING = "serving"
    ENGINE_PER_STEP_REQUEST = "engine_per_step_request"


@dataclass(frozen=True)
class GRConfig:
    """Validated vLLM-GR configuration reconstructed in each process."""

    catalog_path: str | None = None
    constraint_backend: ConstraintBackend = ConstraintBackend.TRIE
    beam_control_mode: BeamControlMode = BeamControlMode.SERVING
    attention_backend: GRAttentionBackend | None = None
    vocab_offset: int | None = None
    vocab_slice_size: int | None = None

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_attention_backend,
            validate_beam_control_mode,
            validate_catalog_path,
            validate_constraint_backend,
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
            "beam_control_mode",
            validate_beam_control_mode(self.beam_control_mode),
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
        if (
            self.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
            and self.catalog_path is None
        ):
            raise ValueError("vllm-gr constraint_backend='constraint_table' requires catalog_path")

    @property
    def enabled(self) -> bool:
        return (
            self.catalog_path is not None
            or self.constraint_backend is not ConstraintBackend.TRIE
            or self.beam_control_mode is not BeamControlMode.SERVING
            or self.attention_backend is not None
            or self.vocab_offset is not None
            or self.vocab_slice_size is not None
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: object | None) -> GRConfig:
        from vllm_gr.config.parser import parse_gr_config

        return parse_gr_config(vllm_config)

    def write_to_vllm_config(self, vllm_config: object) -> None:
        from vllm_gr.config.parser import write_gr_config

        write_gr_config(vllm_config, self)
