# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GR_CONFIG_SCHEMA_VERSION = 1
CONSTRAINT_TABLE_FORMAT = "constraint_table_v1"
CONSTRAINT_TABLE_MAX_DECODE_STEPS = 3
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
# of a catalog triples document (OneRec default).
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


class BeamExecutionMode(str, Enum):
    """Engine-lifetime Beam interface, independent of native async scheduling."""

    LEGACY = "legacy"
    V1 = "v1"


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
    # ModelRunnerOutput.beam_search_result, aligning with the asynchronous scheduling save flow.
    # CUDA beam deployments default to True; NPU requires xllm execution.
    # False explicitly keeps the frontend NumPy path.
    beam_worker_decision: bool | None = None
    # None = unset (the Driver enables this by default for accelerator beam
    # deployments with worker decisions).  True makes EngineCore chain both
    # the prefill -> step0 transition AND every decode-step transition
    # in-process from the worker's beam decision.  The frontend still
    # consumes each step's output (EOS / materialize / streaming) but skips
    # the per-step RPC when the engine chained that step.  Safe under async
    # scheduling: the advance runs in the busy-loop iteration that processes
    # the step's outputs.  Only valid with beam_worker_decision=True (the
    # chain consumes worker beam decisions).
    beam_engine_driven: bool | None = None
    beam_execution_mode: BeamExecutionMode = BeamExecutionMode.LEGACY
    # Exact logical B captured by the NPU Beam graph adapter. This is a graph
    # signature dimension, not a scheduler capacity: other B values do not use
    # the Beam FULL graph (native PIECEWISE/NONE dispatch remains available).
    beam_graph_batch_size: int = 1

    enable_short_suffix_attention: bool = False
    """Fuse CUDA suffix attention and prefix merging when explicitly enabled."""

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_attention_backend,
            validate_beam_engine_driven,
            validate_beam_execution_mode,
            validate_beam_graph_batch_size,
            validate_beam_graph_enabled,
            validate_beam_max_decode_steps,
            validate_beam_max_width,
            validate_beam_worker_decision,
            validate_catalog_path,
            validate_constraint_backend,
            validate_enable_short_suffix_attention,
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
        object.__setattr__(
            self,
            "beam_graph_batch_size",
            validate_beam_graph_batch_size(self.beam_graph_batch_size),
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
            self, "beam_execution_mode", validate_beam_execution_mode(self.beam_execution_mode)
        )
        validate_enable_short_suffix_attention(self.enable_short_suffix_attention)
        if self.enable_short_suffix_attention and self.beam_max_width is None:
            raise ValueError("beam.enable_short_suffix_attention requires max_width")
        if self.beam_engine_driven and not self.beam_worker_decision:
            raise ValueError(
                "vllm-gr beam_engine_driven=True requires beam_worker_decision=True "
                "(the engine-driven session consumes worker beam decisions)"
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
            or self.beam_execution_mode is not BeamExecutionMode.LEGACY
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: object | None) -> GRConfig:
        """Parse the GR config from a ``VllmConfig``'s additional_config.

        The parse is cached on the ``vllm_config`` object itself
        (``_vllm_gr_parsed_config``): the same engine hands the SAME
        ``vllm_config`` to this classmethod on every request/frame path
        (``_engine_constraint_table_active``, ``_beam_graph_enabled_for_run``,
        beam metadata construction, ...), and re-running the full
        validator-suite + ``__post_init__`` each time costs ~0.13ms per
        request.  ``write_to_vllm_config`` invalidates the cache so
        write→read round trips observe the new state.  Hosts that reject
        attribute writes (frozen mocks) silently fall back to per-call
        parsing.
        """
        if vllm_config is None:
            return GRConfig()
        cached = getattr(vllm_config, "_vllm_gr_parsed_config", None)
        if cached is not None:
            return cached
        from vllm_gr.config.parser import parse_gr_config

        parsed = parse_gr_config(vllm_config)
        try:
            object.__setattr__(vllm_config, "_vllm_gr_parsed_config", parsed)
        except (AttributeError, TypeError):
            pass
        return parsed

    def write_to_vllm_config(self, vllm_config: object) -> None:
        from vllm_gr.config.parser import write_gr_config

        write_gr_config(vllm_config, self)
        # Invalidate the parse cache so a subsequent from_vllm_config
        # observes the freshly written state (round-trip semantics).
        try:
            object.__setattr__(vllm_config, "_vllm_gr_parsed_config", None)
        except (AttributeError, TypeError):
            pass
