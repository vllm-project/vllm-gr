# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import cast

from vllm_gr.config.schema import ConfigFieldSpec

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


# Sentinel catalog_path for the NPU default Constraint Table backend: the
# tables are auto-built from the tokenizer's s_a/s_b/s_c token groups instead
# of a catalog triples document (OneRec default).
AUTO_CONSTRAINT_TABLE_PATH: str = "__vllm_gr_auto_constraint_table__"


@dataclass(frozen=True, slots=True)
class ConstraintTableConfig:
    """Configuration for a prebuilt canonical constraint-table artifact."""

    max_top_k: int = field(
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_max_top_k",
                canonical_path=("max_top_k",),
                canonical_required=True,
                canonical_emit="always",
                required=True,
            )
        }
    )
    """Required candidate capacity of the compiled constraint table."""
    enabled: bool = field(
        default=False,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_enabled",
                canonical_path=("enabled",),
                canonical_emit="always",
            )
        },
    )
    """Enable constraint-table filtering for generation."""
    backend: ConstraintTableBackendEnum = field(
        default=ConstraintTableBackendEnum.CUDA,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_backend",
                canonical_path=("backend",),
                canonical_emit="always",
            )
        },
    )
    """Execution placement of the constraint table."""
    implementation_id: str | None = field(
        default=None,
        kw_only=True,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_implementation_id",
                canonical_path=("implementation_id",),
            )
        },
    )
    """Registered implementation that executes the table.

    ``None`` keeps the historical behavior: the ``backend`` enum selects the
    implementation that has always served it. An explicit ID is resolved through
    the constraint registry, so an implementation registered by an external
    package runs without editing a core enum or a selection table.
    """
    path: str | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_path", canonical_path=("path",), canonical_emit="always"
            )
        },
    )
    """Absolute artifact path; YAML loading resolves relative paths first."""
    format: str = field(
        default=CONSTRAINT_TABLE_FORMAT,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_format",
                canonical_path=("format",),
                canonical_emit="always",
            )
        },
    )
    """Supported constraint-table artifact format."""
    artifact_digest: str | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_artifact_digest",
                canonical_path=("artifact_digest",),
                canonical_emit="always",
            )
        },
    )
    """Expected artifact SHA-256 digest, required when enabled."""
    tokenizer_digest: str | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_tokenizer_digest",
                canonical_path=("tokenizer_digest",),
                canonical_emit="always",
            )
        },
    )
    """Expected tokenizer SHA-256 digest, required when enabled."""
    # Independent candidate width K: how many top-k candidates each beam row
    # expands into per step.  None keeps today's behaviour (K == beam_width).
    # K is an engine-lifetime constant because the postprocess workspace is
    # prewarmed at a fixed address.
    candidate_width: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec("validate_candidate_width", canonical_path=("candidate_width",))
        },
    )
    """Candidates per Beam row; None follows the configured Beam width."""

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import validate_config_fields

        validate_config_fields(self)

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

    catalog_path: str | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec("validate_catalog_path", legacy_key="vllm_gr_catalog_path")
        },
    )
    """Legacy catalog path; unset leaves catalog loading disabled."""
    constraint_backend: ConstraintBackend = field(
        default=ConstraintBackend.TRIE,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_backend", legacy_key="vllm_gr_constraint_backend"
            )
        },
    )
    """Legacy constraint backend; table mode also requires a catalog path."""
    attention_backend: GRAttentionBackend | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_attention_backend",
                canonical_path=("attention_backend",),
                legacy_key="vllm_gr_attention_backend",
                canonical_emit="always",
            )
        },
    )
    """Optional attention backend override."""
    vocab_offset: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec("validate_vocab_offset", legacy_key="vllm_gr_vocab_offset")
        },
    )
    """Optional first token ID of a contiguous vocabulary slice."""
    vocab_slice_size: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_vocab_slice_size", legacy_key="vllm_gr_vocab_slice_size"
            )
        },
    )
    """Optional size of a contiguous vocabulary slice."""
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
    beam_graph_enabled: bool = field(
        default=True,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_graph_enabled",
                canonical_path=("beam", "graph_enabled"),
                legacy_key="vllm_gr_beam_graph_enabled",
                canonical_emit="always",
                canonical_strict_bool=True,
            )
        },
    )
    """Enable Beam graph capture when a Beam pool is configured."""
    beam_max_width: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_max_width",
                canonical_path=("beam", "max_width"),
                legacy_key="vllm_gr_beam_max_width",
                canonical_emit="always",
            )
        },
    )
    """Maximum Beam row count; None disables the persistent Beam pool."""
    beam_max_decode_steps: int = field(
        default=3,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_max_decode_steps",
                canonical_path=("beam", "max_decode_steps"),
                legacy_key="vllm_gr_beam_max_decode_steps",
                canonical_emit="always",
            )
        },
    )
    """Maximum Beam decode steps, excluding the SID begin and end tokens."""
    # None = unset: the runtime picks per-platform defaults (xllm on NPU when
    # the operator library is present, cascade otherwise and on CUDA).
    npu_decode_execution: NPUDecodeExecution | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_npu_decode_execution", legacy_key="vllm_gr_npu_decode_execution"
            )
        },
    )
    """NPU decode mode; None leaves platform selection to the runtime."""
    # Canonical schema-v1 state. Keep these fields after every legacy field so
    # positional construction of the pre-schema dataclass remains compatible.
    schema_version: int = field(
        default=GR_CONFIG_SCHEMA_VERSION,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_schema_version",
                canonical_path=("schema_version",),
                canonical_required=True,
                canonical_emit="always",
            )
        },
    )
    """Canonical configuration schema version; required in mapping input."""
    constraint_table: ConstraintTableConfig | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table",
                canonical_path=("constraint_table",),
                canonical_emit="always",
            )
        },
    )
    """Validated canonical artifact configuration, or None for legacy input."""
    # None = unset (the Driver applies platform defaults).  True moves the beam
    # decision (beam_search_group / beam_search_rec_final_select +
    # select_unshared_kv) into the accelerator worker and returns the result via
    # ModelRunnerOutput.beam_search_result, aligning with the asynchronous scheduling save flow.
    # CUDA beam deployments default to True; NPU requires xllm execution.
    # Explicit False is preserved; it does not enable a legacy frontend path.
    beam_worker_decision: bool | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_worker_decision",
                canonical_path=("beam", "worker_decision"),
                legacy_key="vllm_gr_beam_worker_decision",
            )
        },
    )
    """Worker decision override; None defers selection and False opts out."""
    # Deprecated constructor input only. Validation consumes True with a
    # warning and rejects False; no engine-driven state is serialized.
    beam_engine_driven: bool | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_engine_driven",
                canonical_path=("beam", "engine_driven"),
                legacy_key="vllm_gr_beam_engine_driven",
                canonical_emit="never",
                legacy_emit="never",
            )
        },
    )
    """Retired input; True warns and is consumed, and False is rejected."""
    # Unset keeps native inference enabled without selecting GR Beam.
    beam_execution_mode: BeamExecutionMode | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_execution_mode",
                canonical_path=("beam", "execution_mode"),
                legacy_key="vllm_gr_beam_execution_mode",
            )
        },
    )
    """Beam interface selection; None leaves native inference selected."""
    constrained_generation: bool | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constrained_generation",
                canonical_path=("constrained_generation",),
                legacy_key="vllm_gr_constrained_generation",
            )
        },
    )
    """Constraint override; explicit False requests unconstrained generation."""
    # Exact logical B captured by the NPU Beam graph adapter. This is a graph
    # signature dimension, not a scheduler capacity: other B values do not use
    # the Beam FULL graph (native PIECEWISE/NONE dispatch remains available).
    beam_graph_batch_size: int = field(
        default=1,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_graph_batch_size",
                canonical_path=("beam", "graph_batch_size"),
                legacy_key="vllm_gr_beam_graph_batch_size",
            )
        },
    )
    """Exact logical batch size captured by the Beam graph adapter."""

    # Like beam_graph_enabled, this flag is dormant without a Beam pool.
    enable_short_suffix_attention: bool = field(
        default=True,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_enable_short_suffix_attention",
                canonical_path=("beam", "enable_short_suffix_attention"),
                legacy_key="vllm_gr_enable_short_suffix_attention",
            )
        },
    )
    """Fuse CUDA suffix attention and prefix merging by default; False opts out."""

    prefill_flash_attn_version: int | str | None = field(
        default="auto",
        metadata={
            "gr": ConfigFieldSpec(
                "validate_prefill_flash_attn_version",
                canonical_path=("beam", "prefill_flash_attn_version"),
                legacy_key="vllm_gr_prefill_flash_attn_version",
            )
        },
    )
    """CUDA prefill only: auto (default) prefers FA3; None inherits decode; or explicit 2/3."""

    beam_graph_widths: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_graph_widths",
                canonical_path=("beam", "graph_widths"),
                legacy_key="vllm_gr_beam_graph_widths",
            )
        },
    )
    """Additional exact NPU Decode input widths to capture at startup."""

    beam_online_batch_sizes: tuple[int, ...] = field(
        default=DEFAULT_ONLINE_BATCH_SIZES,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_online_batch_sizes",
                canonical_path=("beam", "online_batch_sizes"),
                legacy_key="vllm_gr_beam_online_batch_sizes",
            )
        },
    )
    """Online batching policy and default CUDA capture sizes, not engine capacity."""

    beam_sid_stage_vocab: bool = field(
        default=False,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_sid_stage_vocab",
                canonical_path=("beam", "sid_stage_vocab"),
                legacy_key="vllm_gr_beam_sid_stage_vocab",
            )
        },
    )
    """Opt into catalog-free NPU SA/SB/SC domains and stage-local probabilities."""

    beam_adapter_cache_enabled: bool = field(
        default=False,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_adapter_cache_enabled",
                canonical_path=("beam", "adapter_cache_enabled"),
                legacy_key="vllm_gr_beam_adapter_cache_enabled",
            )
        },
    )
    """Opt into CUDA stage-adapter and Decode graph reuse across batch sizes."""

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import validate_config_fields

        validate_config_fields(self)

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
            return cast(GRConfig, cached)
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
