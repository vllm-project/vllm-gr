# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GR_CONFIG_SCHEMA_VERSION = 1
CONSTRAINT_TABLE_FORMAT = "constraint_table_v1"
# Temporary global Beam width ceiling. Keep every configuration, request,
# runtime contract, and fused kernel on this single limit until wider kernels
# are implemented and qualified end to end.
MAX_BEAM_WIDTH = 256


class GRAttentionBackend(str, Enum):
    CUSTOM = "CUSTOM"


class ConstraintBackend(str, Enum):
    TRIE = "trie"
    CONSTRAINT_TABLE = "constraint_table"


class ConstraintTableBackendEnum(str, Enum):
    """Execution placement requested for a canonical constraint table."""

    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"


# Default values for beam KV pool sizing — also used by parser.py (to decide
# whether a value is the default and can be omitted from additional_config) and
# by validation.py (as fallbacks).  Keep all three sites in sync.
_BEAM_MAX_DECODE_STEPS_DEFAULT: int = 3  # OneRec SID: max_tokens=5 minus begin/end

# Sentinel catalog_path for the NPU default Constraint Table backend: the
# tables are auto-built from the tokenizer's s_a/s_b/s_c token groups instead
# of a catalog triples document (ACS-aligned OneRec default).
AUTO_CONSTRAINT_TABLE_PATH: str = "__vllm_gr_auto_constraint_table__"


@dataclass(frozen=True, slots=True)
class ConstraintTableConfig:
    """Configuration for a prebuilt canonical constraint-table artifact."""

    max_top_k: int
    enabled: bool = False
    backend: ConstraintTableBackendEnum = ConstraintTableBackendEnum.CUDA
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


class GRSchedulingMode(str, Enum):
    """Scheduling topology for a persistent GR request."""

    ASYNC = "async"
    MULTI_STEP = "multi_step"


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
    # None = unset (the Driver applies platform defaults).  True moves the beam
    # decision (beam_search_group / beam_search_rec_final_select +
    # select_unshared_kv) into the accelerator worker and returns the result via
    # ModelRunnerOutput.beam_search_result, aligning with the ACS save flow.
    # CUDA beam deployments default to True; NPU requires xllm execution.
    # False explicitly keeps the frontend NumPy path.
    beam_worker_decision: bool | None = None
    # Compatibility tombstone for pre-persistent configs. True is rejected;
    # False/None decode older config files without restoring self-advance.
    beam_engine_driven: bool | None = None
    # Persistent execution is opt-in so existing request behavior is unchanged.
    persistent_request: bool = False
    persistent_runner_state: bool = False
    scheduling_mode: GRSchedulingMode = GRSchedulingMode.ASYNC

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_attention_backend,
            validate_beam_engine_driven,
            validate_beam_graph_enabled,
            validate_beam_max_decode_steps,
            validate_beam_max_width,
            validate_beam_worker_decision,
            validate_catalog_path,
            validate_constraint_backend,
            validate_gr_scheduling_mode,
            validate_npu_decode_execution,
            validate_persistent_request,
            validate_persistent_runner_state,
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
        object.__setattr__(
            self,
            "beam_worker_decision",
            validate_beam_worker_decision(self.beam_worker_decision),
        )
        object.__setattr__(
            self,
            "beam_engine_driven",
            validate_beam_engine_driven(self.beam_engine_driven),
        )
        object.__setattr__(
            self,
            "persistent_request",
            validate_persistent_request(self.persistent_request),
        )
        object.__setattr__(
            self,
            "persistent_runner_state",
            validate_persistent_runner_state(self.persistent_runner_state),
        )
        object.__setattr__(
            self,
            "scheduling_mode",
            validate_gr_scheduling_mode(self.scheduling_mode),
        )
        if self.scheduling_mode is GRSchedulingMode.MULTI_STEP:
            raise ValueError(
                "vllm-gr scheduling_mode='multi_step' is not supported; "
                "use scheduling_mode='async'"
            )
        if self.beam_engine_driven:
            raise ValueError(
                "vllm-gr beam_engine_driven was removed with the legacy GR "
                "scheduler; use persistent_request=True"
            )
        if self.persistent_runner_state and not self.persistent_request:
            raise ValueError(
                "vllm-gr persistent_runner_state=True requires persistent_request=True"
            )
        if self.persistent_request and not self.persistent_runner_state:
            raise ValueError(
                "vllm-gr persistent_request=True requires persistent_runner_state=True"
            )
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
            or self.beam_worker_decision is not None
            or self.beam_engine_driven is not None
            or self.persistent_request
            or self.persistent_runner_state
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: object | None) -> GRConfig:
        from vllm_gr.config.parser import parse_gr_config

        return parse_gr_config(vllm_config)

    def write_to_vllm_config(self, vllm_config: object) -> None:
        from vllm_gr.config.parser import write_gr_config

        write_gr_config(vllm_config, self)
