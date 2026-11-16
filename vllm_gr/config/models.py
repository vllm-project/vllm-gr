# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GR_CONFIG_SCHEMA_VERSION = 1
CONSTRAINT_TABLE_FORMAT = "constraint_table_v1"
MULTI_TRIE_CONSTRAINT_TABLE_FORMAT = "constraint_table_v2"
CONSTRAINT_TABLE_MAX_DECODE_STEPS = 3
# Global Beam width ceiling.  This constant also gates the raw Beam operators,
# so it is raised only together with the kernels that make W > 256 correct:
# the hierarchical Top-K used to size every local list from LOCAL_TILE_SIZE
# (256) regardless of the selection width, which left the merge stage reading
# slots the local stage never wrote.  With _beam_selection_tiling shared by
# allocation and launch, a list is now at least as wide as the selection, so
# widths above the fused-kernel whitelist ({64, 128, 256}) are correct on the
# hierarchical path.
#
# A width above MAX_CANDIDATE_WIDTH is servable only when the candidate width
# is stated independently: unset, K follows the Beam width and the operators
# still cap K at 256.
MAX_BEAM_WIDTH = 1024
# Candidate width (top-k per beam row) is an independent dimension: the
# RecConstrainedTopK operator caps it at 256, so 256 is the single ceiling
# shared by every configuration, request contract, and Triton launch.
MAX_CANDIDATE_WIDTH = 256
# Conservative startup resource policy, including the implicit max-width bank.
# This bounds retained width count; it is not a guarantee that graphs fit memory.
MAX_NPU_BEAM_GRAPH_WIDTHS = 8
SUPPORTED_BEAM_BATCH_SIZES = tuple(range(1, 9))
DEFAULT_ONLINE_BATCH_SIZES = (1, 2, 4)


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
    # Independent candidate width K: how many top-k candidates each beam row
    # expands into per step.  None keeps today's behaviour (K == beam_width).
    # K is an engine-lifetime constant because the postprocess workspace is
    # prewarmed at a fixed address.
    candidate_width: int | None = None

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_artifact_digest,
            validate_candidate_width,
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
        object.__setattr__(
            self,
            "candidate_width",
            validate_candidate_width(self.candidate_width),
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
    #
    # beam_max_width is the ROW count only. The per-row candidate count K is
    # ``constraint_table.candidate_width`` and is independent of it: K follows
    # beam_max_width when unset, and a beam wider than MAX_CANDIDATE_WIDTH must
    # state K explicitly (see resolve_candidate_width).
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
    # Explicit False is preserved; it does not enable a legacy frontend path.
    beam_worker_decision: bool | None = None
    # Deprecated constructor input only. Validation consumes True with a
    # warning and rejects False; no engine-driven state is serialized.
    beam_engine_driven: bool | None = None
    # Unset keeps native inference enabled without selecting GR Beam.
    beam_execution_mode: BeamExecutionMode | None = None
    constrained_generation: bool | None = None
    # Exact logical B captured by the NPU Beam graph adapter. This is a graph
    # signature dimension, not a scheduler capacity: other B values do not use
    # the Beam FULL graph (native PIECEWISE/NONE dispatch remains available).
    beam_graph_batch_size: int = 1

    # Like beam_graph_enabled, this flag is dormant without a Beam pool.
    enable_short_suffix_attention: bool = True
    """Fuse CUDA suffix attention and prefix merging by default; False opts out."""

    prefill_flash_attn_version: int | str | None = "auto"
    """CUDA prefill only: auto (default) prefers FA3; None inherits decode; or explicit 2/3."""

    beam_graph_widths: tuple[int, ...] | None = None
    """Additional exact NPU Decode input widths to capture at startup."""

    beam_online_batch_sizes: tuple[int, ...] = DEFAULT_ONLINE_BATCH_SIZES
    """Online batching policy and default CUDA capture sizes, not engine capacity."""

    beam_sid_stage_vocab: bool = False
    """Opt into catalog-free NPU SA/SB/SC domains and stage-local probabilities."""

    beam_adapter_cache_enabled: bool = False
    """Opt into CUDA stage-adapter and Decode graph reuse across batch sizes."""

    enable_kv_pool_ping_pong: bool = True
    """Gather CUDA fixed-width KV into the next bank; False restores write-back."""

    def __post_init__(self) -> None:
        if type(self.beam_adapter_cache_enabled) is not bool:
            raise ValueError("beam.adapter_cache_enabled must be a boolean")
        from vllm_gr.config.validation import (
            validate_attention_backend,
            validate_beam_engine_driven,
            validate_beam_execution_mode,
            validate_beam_graph_batch_size,
            validate_beam_graph_enabled,
            validate_beam_graph_widths,
            validate_beam_max_decode_steps,
            validate_beam_max_width,
            validate_beam_online_batch_sizes,
            validate_beam_sid_stage_vocab,
            validate_beam_worker_decision,
            validate_catalog_path,
            validate_constrained_generation,
            validate_constraint_backend,
            validate_enable_kv_pool_ping_pong,
            validate_enable_short_suffix_attention,
            validate_npu_decode_execution,
            validate_prefill_flash_attn_version,
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
            self, "beam_graph_widths", validate_beam_graph_widths(self.beam_graph_widths)
        )
        if self.beam_graph_widths is not None and (
            self.beam_max_width is None or max(self.beam_graph_widths) > self.beam_max_width
        ):
            raise ValueError("beam.graph_widths requires max_width >= every captured width")
        if self.beam_graph_active:
            capture_widths = set(self.beam_graph_widths or ()) | {self.beam_max_width}
            if len(capture_widths) > MAX_NPU_BEAM_GRAPH_WIDTHS:
                raise ValueError(
                    "beam.graph_widths allows at most "
                    f"{MAX_NPU_BEAM_GRAPH_WIDTHS} captured widths including max_width, "
                    f"got {len(capture_widths)}; use fewer widths or disable beam graphs"
                )
        object.__setattr__(
            self,
            "beam_online_batch_sizes",
            validate_beam_online_batch_sizes(self.beam_online_batch_sizes),
        )
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
        object.__setattr__(
            self,
            "constrained_generation",
            validate_constrained_generation(self.constrained_generation),
        )
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
        object.__setattr__(
            self, "beam_sid_stage_vocab", validate_beam_sid_stage_vocab(self.beam_sid_stage_vocab)
        )
        if self.beam_sid_stage_vocab and (
            self.constrained_generation is not False
            or self.beam_execution_mode is not BeamExecutionMode.V1
            or self.beam_worker_decision is not True
            or self.beam_max_decode_steps > CONSTRAINT_TABLE_MAX_DECODE_STEPS
        ):
            raise ValueError(
                "beam.sid_stage_vocab requires constrained_generation=False, "
                "beam.execution_mode='v1', beam.worker_decision=True, and "
                "beam.max_decode_steps <= 3"
            )
        validate_enable_short_suffix_attention(self.enable_short_suffix_attention)
        validate_enable_kv_pool_ping_pong(self.enable_kv_pool_ping_pong)
        validate_prefill_flash_attn_version(self.prefill_flash_attn_version)
        if (
            self.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
            and self.catalog_path is None
        ):
            raise ValueError("vllm-gr constraint_backend='constraint_table' requires catalog_path")
        if self.constrained_generation is False and (
            self.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
            or self.vocab_offset is not None
            or self.vocab_slice_size is not None
            or (self.constraint_table is not None and self.constraint_table.enabled)
        ):
            raise ValueError(
                "constrained_generation=False cannot be combined with an enabled "
                "constraint table or a configured vocabulary slice"
            )
        if (
            (self.constraint_table is None or self.constraint_table.candidate_width is None)
            and self.beam_max_width is not None
            and self.beam_max_width > MAX_CANDIDATE_WIDTH
        ):
            # K follows W unless stated, so leaving candidate_width unset --
            # with no table, or with a table that does not state it -- implies
            # K at the Beam width: a Beam wider than the candidate ceiling
            # would describe a K the operators cannot take.  An explicit
            # ``constraint_table.candidate_width`` lifts this, because the
            # candidate width is then stated independently of the beam.
            raise ValueError(
                f"vllm-gr beam_max_width={self.beam_max_width} exceeds the candidate "
                f"width ceiling {MAX_CANDIDATE_WIDTH}; K follows the Beam width when "
                "constraint_table.candidate_width is unset. Set "
                "constraint_table.candidate_width to the per-row candidate count, or "
                f"keep beam_max_width <= {MAX_CANDIDATE_WIDTH}."
            )

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
            or self.constrained_generation is not None
            or self.beam_worker_decision is not None
            or self.beam_execution_mode is not None
            or self.beam_sid_stage_vocab
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


def resolve_candidate_width(config: GRConfig) -> int:
    """Resolve the effective per-row candidate width K.

    The single derivation point: ``constraint_table.candidate_width`` wins when
    set, otherwise K follows ``beam_max_width`` (the pre-decoupling behaviour).
    Every workspace bucket key, admission gate, and kernel launch must call
    this instead of spelling out ``(width, width, dtype)``.
    """

    # ``getattr`` keeps this usable from partial config objects (the worker
    # admits a structural stand-in in several tests and in the V2 path).
    table = getattr(config, "constraint_table", None)
    if table is not None and getattr(table, "candidate_width", None) is not None:
        return int(table.candidate_width)
    if config.beam_max_width is None:
        raise ValueError("vllm-gr cannot resolve candidate_width without beam_max_width")
    candidate_width = int(config.beam_max_width)
    if candidate_width > MAX_CANDIDATE_WIDTH:
        # Following the beam width is only valid while it fits the operator
        # ceiling; a wider beam must state its per-row candidate width.
        raise ValueError(
            f"vllm-gr beam_max_width={candidate_width} exceeds the candidate width "
            f"ceiling {MAX_CANDIDATE_WIDTH}; K follows the Beam width when "
            "constraint_table.candidate_width is unset. Set "
            "constraint_table.candidate_width to the per-row candidate count, or "
            f"keep beam_max_width <= {MAX_CANDIDATE_WIDTH}."
        )
    return candidate_width
