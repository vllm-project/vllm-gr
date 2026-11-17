# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, fields, replace
from enum import Enum

from vllm_gr.config.schema import ConfigFieldSpec

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
MAX_BEAM_WIDTH = 1024
# Unconstrained Top-K supports the full Beam capacity. Constraint-table
# operators have a separate output limit, checked when loading the backend.
MAX_CANDIDATE_WIDTH = MAX_BEAM_WIDTH
MAX_CONSTRAINT_CANDIDATE_WIDTH = 256
# Conservative startup resource policy, including the implicit max-width bank.
# This bounds retained width count; it is not a guarantee that graphs fit memory.
MAX_NPU_BEAM_GRAPH_WIDTHS = 8
SUPPORTED_BEAM_BATCH_SIZES = tuple(range(1, 9))


DEFAULT_BEAM_BATCH_SIZES = (1, 2, 4)
DEFAULT_PREFILL_TOKEN_BUCKETS = tuple(range(128, 8193, 128))


@dataclass(frozen=True, slots=True)
class DeploymentOptions:
    """Settings serialized with the engine, including request/worker defaults."""

    beam_width: tuple[int, ...] | None = None
    candidate_width: tuple[int, ...] | None = None
    result_width: int | None = None
    prefill_graph_enabled: bool = True
    dense_prefill: str = "off"
    prefill_token_buckets: tuple[int, ...] | None = DEFAULT_PREFILL_TOKEN_BUCKETS
    beam_batch_sizes: tuple[int, ...] = DEFAULT_BEAM_BATCH_SIZES
    prefill_kv_bound: int | None = None
    npu_bsnd_capture_specs: str = ""
    npu_stage_lm_head: bool = True
    head_slice: int | str = "auto"
    head_slice_min_rows: int = 1024
    head_slice_max_mb: float = 128
    unconstrained_head_layout: str | None = "transposed"
    topk_keep: int = 8
    topk_select: str = "i64"
    prefix_policy: str = "off"
    prefix_min_chain_us: float = 150.0
    prefix_calibration: bool = True
    prefix_calibration_path: str | None = None
    trie_retention_seconds: float = 60
    admin_host: str = "127.0.0.1"
    admin_port: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping) -> DeploymentOptions:
        if not isinstance(data, Mapping):
            raise ValueError("GR deployment settings must be a mapping")
        unknown = data.keys() - OPTION_KEYS
        if unknown:
            raise ValueError(f"Unknown gr deployment settings: {sorted(unknown, key=str)}")
        values = dict(data)
        if values.get("beam_batch_sizes") is None:
            values["beam_batch_sizes"] = DEFAULT_BEAM_BATCH_SIZES
        for key in (
            "beam_width",
            "candidate_width",
            "prefill_token_buckets",
            "beam_batch_sizes",
        ):
            value = values.get(key)
            if value is None:
                continue
            if (
                not isinstance(value, (list, tuple))
                or not value
                or any(type(v) is not int or v < 1 for v in value)
            ):
                raise ValueError(f"gr.{key} requires a nonempty list of positive integers")
            if key in ("prefill_token_buckets", "beam_batch_sizes") and list(value) != sorted(
                set(value)
            ):
                raise ValueError(f"gr.{key} must be strictly increasing")
            values[key] = tuple(value)
        options = cls(**values)
        for key in (
            "result_width",
            "prefill_kv_bound",
            "topk_keep",
        ):
            value = getattr(options, key)
            if value is None and key in ("result_width", "prefill_kv_bound"):
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"gr.{key} requires a positive integer")
        if any(v > 8 for v in options.beam_batch_sizes):
            raise ValueError("GR beam batch sizes must be in 1..8")
        if options.topk_keep > 256 or options.topk_select not in ("i32", "i64"):
            raise ValueError("gr.topk_keep must be in 1..256 and topk_select must be i32 or i64")
        for key in ("prefill_graph_enabled", "npu_stage_lm_head", "prefix_calibration"):
            if type(getattr(options, key)) is not bool:
                raise ValueError(f"gr.{key} requires a boolean")
        if options.admin_port is not None and (
            type(options.admin_port) is not int or not 1 <= options.admin_port <= 65535
        ):
            raise ValueError("gr.admin_port must be in 1..65535")
        if not isinstance(options.admin_host, str) or not options.admin_host:
            raise ValueError("gr.admin_host requires a hostname or address")
        if options.head_slice != "auto" and (
            type(options.head_slice) is not int or options.head_slice < 0
        ):
            raise ValueError("gr.head_slice requires auto or a nonnegative integer")
        if type(options.head_slice_min_rows) is not int or options.head_slice_min_rows < 0:
            raise ValueError("gr.head_slice_min_rows requires a nonnegative integer")
        for name in ("head_slice_max_mb", "trie_retention_seconds", "prefix_min_chain_us"):
            value = getattr(options, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"gr.{name} requires a finite nonnegative number")
        if options.prefix_policy not in (
            "auto",
            "off",
        ) or options.unconstrained_head_layout not in (None, "transposed"):
            raise ValueError("Unsupported prefix_policy or unconstrained_head_layout")
        if options.dense_prefill not in ("off", "fa2", "cudnn"):
            raise ValueError("gr.dense_prefill must be off, fa2 or cudnn")
        if not isinstance(options.npu_bsnd_capture_specs, str):
            raise ValueError("gr.npu_bsnd_capture_specs requires a string")
        if options.prefix_calibration_path is not None and not isinstance(
            options.prefix_calibration_path, str
        ):
            raise ValueError("gr.prefix_calibration_path requires a path")
        return options

    def to_mapping(self) -> dict:
        return {
            f.name: list(v) if isinstance(v, tuple) else v
            for f in fields(self)
            if (v := getattr(self, f.name)) != f.default
        }


DEFAULT_DEPLOYMENT = DeploymentOptions()


OPTION_KEYS = frozenset(f.name for f in fields(DeploymentOptions))


class GRAttentionBackend(str, Enum):
    CUSTOM = "CUSTOM"


class ConstraintMode(str, Enum):
    """Resolved switch between no constraint and a canonical table."""

    NONE = "none"
    TABLE = "table"


class ConstraintBackendKind(str, Enum):
    """Resolved constraint-table placement, including automatic selection."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"


ConstraintTableBackendEnum = ConstraintBackendKind


@dataclass(frozen=True, slots=True)
class ConstraintTableSpec:
    """Canonical constraint-table source; runtime loads one artifact."""

    path: str | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_path", canonical_path=("path",), canonical_emit="always"
            )
        },
    )
    """Absolute artifact path; YAML loading resolves relative paths first."""
    backend: ConstraintBackendKind = field(
        default=ConstraintBackendKind.AUTO,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_backend",
                canonical_path=("backend",),
                canonical_emit="always",
            )
        },
    )
    """Execution placement of the constraint table."""
    max_top_k: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_optional_max_top_k",
                canonical_path=("max_top_k",),
                canonical_emit="omit_none",
            )
        },
    )
    """Candidate capacity required by table-mode deployments."""
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
                canonical_path=("digests", "artifact"),
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
                canonical_path=("digests", "tokenizer"),
                canonical_emit="always",
            )
        },
    )
    """Expected tokenizer SHA-256 digest, required when enabled."""
    enabled: bool = field(
        default=False,
        metadata={"gr": ConfigFieldSpec("validate_constraint_table_enabled")},
    )
    """Enable constraint-table filtering for generation."""
    candidate_width: int | None = field(
        default=None,
        metadata={"gr": ConfigFieldSpec("validate_candidate_width")},
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


@dataclass(frozen=True, slots=True)
class ConstraintConfig:
    """Resolved constraint mode and its optional table specification."""

    mode: ConstraintMode = ConstraintMode.NONE
    table: ConstraintTableSpec | None = None


# Sentinel catalog_path for the NPU default Constraint Table backend: the
# tables are auto-built from the tokenizer's s_a/s_b/s_c token groups instead
# of a catalog triples document (OneRec default).
AUTO_CONSTRAINT_TABLE_PATH: str = "__vllm_gr_auto_constraint_table__"


class NPUDecodeExecution(str, Enum):
    """Execution mode for the NPU incremental beam decode branch."""

    CASCADE = "cascade"
    XLLM = "xllm"


@dataclass(frozen=True)
class GRConfig:
    """Validated vLLM-GR configuration reconstructed in each process."""

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
                canonical_path=("gr", "beam_graph_enabled"),
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
                canonical_path=("gr", "beam_width"),
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
                canonical_path=("gr", "max_decode_steps"),
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
                "validate_npu_decode_execution",
                canonical_path=("gr", "npu_decode_execution"),
                legacy_key="vllm_gr_npu_decode_execution",
            )
        },
    )
    """NPU decode mode; None leaves platform selection to the runtime."""
    # None = unset (the Driver applies platform defaults).  True moves the beam
    # decision (beam_search_group / beam_search_rec_final_select +
    # select_unshared_kv) into the V1 stage runner, which owns the result.
    # CUDA beam deployments default to True; NPU requires xllm execution.
    # Explicit False is preserved; it does not enable a legacy frontend path.
    beam_worker_decision: bool | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_worker_decision",
                canonical_path=("gr", "worker_decision"),
                legacy_key="vllm_gr_beam_worker_decision",
            )
        },
    )
    """Worker decision override; None defers selection and False opts out."""
    # Exact logical B captured by the NPU Beam graph adapter. This is a graph
    # signature dimension, not a scheduler capacity: other B values do not use
    # the Beam FULL graph (native PIECEWISE/NONE dispatch remains available).
    beam_graph_batch_size: int = field(
        default=1,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_graph_batch_size",
                canonical_path=("gr", "decode_graph_batch_size"),
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
                canonical_path=("gr", "enable_short_suffix_attention"),
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
                canonical_path=("gr", "prefill_flash_attn_version"),
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
                canonical_path=("gr", "decode_graph_widths"),
                legacy_key="vllm_gr_beam_graph_widths",
            )
        },
    )
    """Additional exact NPU Decode input widths to capture at startup."""

    beam_sid_stage_vocab: bool = field(
        default=False,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_sid_stage_vocab",
                canonical_path=("gr", "sid_stage_vocab"),
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
                canonical_path=("gr", "adapter_cache_enabled"),
                legacy_key="vllm_gr_beam_adapter_cache_enabled",
            )
        },
    )
    """Opt into CUDA stage-adapter and Decode graph reuse across batch sizes."""

    constraint: ConstraintConfig | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_config",
                canonical_emit="omit_none",
            )
        },
    )
    """Constraint mode and optional table descriptor."""
    beam_candidate_width: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_candidate_width", canonical_path=("gr", "candidate_width")
            )
        },
    )
    """Engine-wide candidate width per parent beam."""

    deployment: DeploymentOptions = field(
        default=DEFAULT_DEPLOYMENT,
        metadata={"gr": ConfigFieldSpec("validate_deployment")},
    )
    """Resolved startup settings shared by the driver and workers."""
    _runtime_table: ConstraintTableSpec | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={"gr": ConfigFieldSpec("validate_constraint_table")},
    )
    """Cached table with device placement and candidate capacity resolved."""

    # Compatibility storage for retired constructor-only controls.
    _vocab_offset: int | None = field(
        default=None, metadata={"gr": ConfigFieldSpec("validate_vocab_offset")}
    )
    """Compatibility storage for the vocabulary offset."""
    _vocab_slice_size: int | None = field(
        default=None, metadata={"gr": ConfigFieldSpec("validate_vocab_slice_size")}
    )
    """Compatibility storage for the vocabulary slice size."""
    _beam_engine_driven: bool | None = field(
        default=None, metadata={"gr": ConfigFieldSpec("validate_beam_engine_driven")}
    )
    """Compatibility storage for the retired engine-driven flag."""

    # Constructor-only retired inputs. They are not dataclass fields.
    catalog_path: InitVar[str | None] = None
    constraint_backend: InitVar[str | None] = None
    vocab_offset: InitVar[int | None] = None
    vocab_slice_size: InitVar[int | None] = None
    beam_engine_driven: InitVar[bool | None] = None
    constrained_generation: InitVar[bool | None] = None
    constraint_table: InitVar[ConstraintTableSpec | None] = None

    def __post_init__(
        self,
        catalog_path: str | None,
        constraint_backend: str | None,
        vocab_offset: int | None,
        vocab_slice_size: int | None,
        beam_engine_driven: bool | None,
        constrained_generation: bool | None,
        constraint_table: ConstraintTableSpec | None,
    ) -> None:
        from vllm_gr.config.validation import (
            validate_beam_engine_driven,
            validate_catalog_path,
            validate_config_fields,
            validate_constrained_generation,
            validate_constraint_config,
            validate_legacy_constraint_backend,
            validate_vocab_offset,
            validate_vocab_slice_size,
        )

        validate_config_fields(self)
        catalog_path = validate_catalog_path(catalog_path)
        constraint_backend = (
            "trie"
            if constraint_backend is None
            else validate_legacy_constraint_backend(constraint_backend)
        )
        object.__setattr__(self, "_vocab_offset", validate_vocab_offset(vocab_offset))
        object.__setattr__(self, "_vocab_slice_size", validate_vocab_slice_size(vocab_slice_size))
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
        constrained_generation = validate_constrained_generation(constrained_generation)
        if constraint_table is not None and not isinstance(constraint_table, ConstraintTableSpec):
            raise ValueError("vllm-gr constraint_table must be a ConstraintTableSpec or None")
        constraint = self.constraint
        if constraint is None and constraint_table is not None and constraint_table.enabled:
            constraint = ConstraintConfig(
                mode=ConstraintMode.TABLE,
                table=ConstraintTableSpec(
                    max_top_k=constraint_table.max_top_k,
                    backend=ConstraintBackendKind(constraint_table.backend.value),
                    path=constraint_table.path,
                    format=constraint_table.format,
                    artifact_digest=constraint_table.artifact_digest,
                    tokenizer_digest=constraint_table.tokenizer_digest,
                    enabled=True,
                ),
            )
            if self.beam_candidate_width is None:
                object.__setattr__(self, "beam_candidate_width", constraint_table.candidate_width)
        elif constraint is None and (
            constraint_backend == "constraint_table" or constrained_generation is True
        ):
            table_path = (
                constraint_table.path
                if constraint_table is not None and constraint_table.path is not None
                else catalog_path
            )
            if table_path is None:
                raise ValueError(
                    "vllm-gr constraint_backend='constraint_table' requires catalog_path"
                )
            table_backend = (
                ConstraintBackendKind(constraint_table.backend.value)
                if constraint_table is not None
                else ConstraintBackendKind.CUDA
            )
            constraint = ConstraintConfig(
                mode=ConstraintMode.TABLE,
                table=ConstraintTableSpec(
                    max_top_k=constraint_table.max_top_k if constraint_table is not None else 1,
                    backend=table_backend,
                    path=table_path,
                    format=(
                        constraint_table.format
                        if constraint_table is not None
                        else CONSTRAINT_TABLE_FORMAT
                    ),
                    artifact_digest=(
                        constraint_table.artifact_digest if constraint_table is not None else None
                    ),
                    tokenizer_digest=(
                        constraint_table.tokenizer_digest if constraint_table is not None else None
                    ),
                    enabled=True,
                ),
            )
            if self.beam_candidate_width is None and constraint_table is not None:
                object.__setattr__(self, "beam_candidate_width", constraint_table.candidate_width)
        elif constraint is None and constrained_generation is False:
            constraint = ConstraintConfig(mode=ConstraintMode.NONE)
        if self.constraint is None and constraint is not None:
            constraint = validate_constraint_config(constraint)
        object.__setattr__(self, "constraint", constraint)
        if constraint is not None and constraint.mode is not ConstraintMode.NONE:
            table_spec = constraint.table
            assert table_spec is not None  # validated above
            backend = table_spec.backend
            max_top_k = table_spec.max_top_k
            if max_top_k is None and constraint_table is not None:
                max_top_k = constraint_table.max_top_k
            if max_top_k is None:
                raise ValueError(
                    "constraint.mode='table' requires constraint.table.max_top_k "
                    "until artifact metadata carries it"
                )
            normalized_table = replace(
                table_spec, enabled=True, backend=backend, max_top_k=max_top_k
            )
            object.__setattr__(
                self, "constraint", ConstraintConfig(ConstraintMode.TABLE, normalized_table)
            )
            # Workers read the legacy view on the request path. Derive it once
            # from the validated deployment settings.
            object.__setattr__(
                self,
                "_runtime_table",
                replace(
                    normalized_table,
                    enabled=True,
                    backend=ConstraintBackendKind.CUDA
                    if backend is ConstraintBackendKind.AUTO
                    else backend,
                    candidate_width=self.beam_candidate_width,
                ),
            )
        object.__setattr__(
            self,
            "_beam_engine_driven",
            validate_beam_engine_driven(beam_engine_driven),
        )
        unconstrained = constraint is not None and constraint.mode is ConstraintMode.NONE
        if self.beam_sid_stage_vocab and (
            not unconstrained
            or self.beam_worker_decision is not True
            or self.beam_max_decode_steps > CONSTRAINT_TABLE_MAX_DECODE_STEPS
        ):
            raise ValueError(
                "beam.sid_stage_vocab requires constrained_generation=False, "
                "gr.worker_decision=True, and "
                "beam.max_decode_steps <= 3"
            )
        if unconstrained and (
            constraint_table is not None
            and constraint_table.enabled
            or self._vocab_offset is not None
            or self._vocab_slice_size is not None
        ):
            raise ValueError(
                "constrained_generation=False cannot be combined with an enabled "
                "constraint table or a configured vocabulary slice"
            )
        if (
            self.beam_candidate_width is None
            and (self.constraint_table is None or self.constraint_table.candidate_width is None)
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
            self.constraint is not None
            or self.attention_backend is not None
            or self._vocab_offset is not None
            or self._vocab_slice_size is not None
            or self.npu_decode_execution is not None
            or self.beam_worker_decision is not None
            or self.beam_sid_stage_vocab
        )

    @property
    def constraint_config(self) -> ConstraintConfig:
        return (
            self.constraint
            if self.constraint is not None
            else ConstraintConfig(ConstraintMode.NONE)
        )

    def __getattr__(self, name: str) -> object:
        if name == "catalog_path":
            table = self.constraint_table
            return table.path if table is not None else None
        if name == "constraint_backend":
            if self.constraint is not None and self.constraint.mode is ConstraintMode.TABLE:
                return "constraint_table"
            return "trie"
        if name == "constrained_generation":
            if self.constraint is None:
                return None
            return False if self.constraint.mode is ConstraintMode.NONE else None
        if name == "vocab_offset":
            return self._vocab_offset
        if name == "vocab_slice_size":
            return self._vocab_slice_size
        if name == "beam_engine_driven":
            return self._beam_engine_driven
        if name == "constraint_table":
            return self._runtime_table
        raise AttributeError(name)

    @classmethod
    def from_vllm_config(cls, vllm_config: object | None) -> GRConfig:
        """Parse the GR config from a ``VllmConfig``'s additional_config.

        The parse is cached on the ``vllm_config`` object itself
        (``_vllm_gr_parsed_config``): the same engine hands the SAME
        ``vllm_config`` to this classmethod on every request/frame path
        (``_engine_constraint_table_active``, ``_beam_graph_enabled_for_run``,
        beam metadata construction, ...), and re-running the full
        validator-suite + ``__post_init__`` each time costs ~0.13ms per
        request.  ``write_to_vllm_config`` replaces the cached snapshot so
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


for _legacy_name in (
    "catalog_path",
    "constraint_backend",
    "vocab_offset",
    "vocab_slice_size",
    "beam_engine_driven",
    "constrained_generation",
    "constraint_table",
):
    if _legacy_name in GRConfig.__dict__:
        delattr(GRConfig, _legacy_name)


def configured_candidate_width(config: GRConfig) -> int | None:
    """Return the explicit per-parent capacity before request defaults."""
    width = getattr(config, "beam_candidate_width", None)
    if width is None:
        width = getattr(getattr(config, "constraint_table", None), "candidate_width", None)
    return int(width) if width is not None else None


def configured_constraint_mode(config: GRConfig) -> ConstraintMode | None:
    """Distinguish omitted constraints, explicit NONE, and active tables."""
    constraint = getattr(config, "constraint", None)
    if constraint is not None:
        return constraint.mode
    if getattr(config, "constrained_generation", None) is False:
        return ConstraintMode.NONE
    if getattr(getattr(config, "constraint_table", None), "enabled", False):
        return ConstraintMode.TABLE
    return None


def resolve_candidate_width(config: GRConfig) -> int:
    """Resolve the effective per-row candidate width K.

    The single derivation point: ``constraint_table.candidate_width`` wins when
    set, otherwise K follows ``beam_max_width`` (the pre-decoupling behaviour).
    Every workspace bucket key, admission gate, and kernel launch must call
    this instead of spelling out ``(width, width, dtype)``.
    """

    # ``getattr`` keeps this usable from partial config objects (the worker
    # admits a structural stand-in in several tests and in the V2 path).
    configured = configured_candidate_width(config)
    if configured is not None:
        return configured
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


def resolve_constraint_backend(config: GRConfig, device_type: str) -> GRConfig:
    """Bind AUTO to the driver device before publishing the worker payload."""
    constraint = config.constraint
    if (
        constraint is None
        or constraint.table is None
        or constraint.table.backend is not ConstraintBackendKind.AUTO
    ):
        return config
    backend = ConstraintBackendKind(device_type)
    if backend is ConstraintBackendKind.AUTO:
        raise ValueError("auto requires a concrete platform device type")
    return replace(
        config, constraint=replace(constraint, table=replace(constraint.table, backend=backend))
    )


def deployment_for(vllm_config: object | None) -> DeploymentOptions:
    if vllm_config is None:
        return DEFAULT_DEPLOYMENT
    return GRConfig.from_vllm_config(vllm_config).deployment
