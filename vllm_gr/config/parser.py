# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Final, TypeAlias, cast

from vllm_gr.config.models import (
    _BEAM_MAX_DECODE_STEPS_DEFAULT,
    CONSTRAINT_TABLE_FORMAT,
    DEFAULT_ONLINE_BATCH_SIZES,
    GR_CONFIG_SCHEMA_VERSION,
    ConstraintBackend,
    ConstraintTableConfig,
    GRConfig,
)
from vllm_gr.config.validation import (
    validate_artifact_digest,
    validate_attention_backend,
    validate_beam_engine_driven,
    validate_beam_execution_mode,
    validate_beam_graph_batch_size,
    validate_beam_graph_enabled,
    validate_beam_graph_widths,
    validate_beam_max_decode_steps,
    validate_beam_max_width,
    validate_beam_online_batch_sizes,
    validate_beam_worker_decision,
    validate_candidate_width,
    validate_catalog_path,
    validate_constrained_generation,
    validate_constraint_backend,
    validate_constraint_table_backend,
    validate_constraint_table_enabled,
    validate_constraint_table_format,
    validate_constraint_table_max_top_k,
    validate_constraint_table_path,
    validate_enable_short_suffix_attention,
    validate_npu_decode_execution,
    validate_schema_version,
    validate_tokenizer_digest,
    validate_vocab_offset,
    validate_vocab_slice_size,
)

GR_CONFIG_KEY: Final = "vllm_gr"
GRConfigInput: TypeAlias = str | Path | Mapping[str, object] | GRConfig
CATALOG_PATH_KEY: Final = "vllm_gr_catalog_path"
CONSTRAINT_BACKEND_KEY: Final = "vllm_gr_constraint_backend"
ATTENTION_BACKEND_KEY: Final = "vllm_gr_attention_backend"
VOCAB_OFFSET_KEY: Final = "vllm_gr_vocab_offset"
VOCAB_SLICE_SIZE_KEY: Final = "vllm_gr_vocab_slice_size"
SHORT_SUFFIX_ATTENTION_KEY: Final = "vllm_gr_enable_short_suffix_attention"
BEAM_GRAPH_ENABLED_KEY: Final = "vllm_gr_beam_graph_enabled"
BEAM_GRAPH_BATCH_SIZE_KEY: Final = "vllm_gr_beam_graph_batch_size"
BEAM_GRAPH_WIDTHS_KEY: Final = "vllm_gr_beam_graph_widths"
BEAM_ONLINE_BATCH_SIZES_KEY: Final = "vllm_gr_beam_online_batch_sizes"
BEAM_MAX_WIDTH_KEY: Final = "vllm_gr_beam_max_width"
BEAM_MAX_DECODE_STEPS_KEY: Final = "vllm_gr_beam_max_decode_steps"
NPU_DECODE_EXECUTION_KEY: Final = "vllm_gr_npu_decode_execution"
BEAM_WORKER_DECISION_KEY: Final = "vllm_gr_beam_worker_decision"
BEAM_ENGINE_DRIVEN_KEY: Final = "vllm_gr_beam_engine_driven"
BEAM_EXECUTION_MODE_KEY: Final = "vllm_gr_beam_execution_mode"
CONSTRAINED_GENERATION_KEY: Final = "vllm_gr_constrained_generation"
GR_CONFIG_KEY_PREFIX: Final = "vllm_gr_"
KNOWN_GR_CONFIG_KEYS: Final = frozenset(
    {
        CATALOG_PATH_KEY,
        CONSTRAINT_BACKEND_KEY,
        ATTENTION_BACKEND_KEY,
        VOCAB_OFFSET_KEY,
        VOCAB_SLICE_SIZE_KEY,
        BEAM_GRAPH_ENABLED_KEY,
        SHORT_SUFFIX_ATTENTION_KEY,
        BEAM_GRAPH_BATCH_SIZE_KEY,
        BEAM_GRAPH_WIDTHS_KEY,
        BEAM_ONLINE_BATCH_SIZES_KEY,
        BEAM_MAX_WIDTH_KEY,
        BEAM_MAX_DECODE_STEPS_KEY,
        NPU_DECODE_EXECUTION_KEY,
        BEAM_WORKER_DECISION_KEY,
        BEAM_ENGINE_DRIVEN_KEY,
        BEAM_EXECUTION_MODE_KEY,
        CONSTRAINED_GENERATION_KEY,
    }
)
_TOP_LEVEL_KEYS: Final = frozenset(
    {"schema_version", "beam", "constraint_table", "attention_backend", "constrained_generation"}
)
_BEAM_KEYS: Final = frozenset(
    {
        "graph_enabled",
        "enable_short_suffix_attention",
        "graph_batch_size",
        "graph_widths",
        "online_batch_sizes",
        "max_width",
        "max_decode_steps",
        "worker_decision",
        "engine_driven",
        "execution_mode",
    }
)
_CONSTRAINT_TABLE_KEYS: Final = frozenset(
    {
        "enabled",
        "backend",
        "path",
        "format",
        "artifact_digest",
        "tokenizer_digest",
        "max_top_k",
        "candidate_width",
    }
)


def _reject_unknown_gr_keys(additional_config: Mapping[object, object]) -> None:
    unknown_keys = sorted(
        key
        for key in additional_config
        if isinstance(key, str)
        and key.startswith(GR_CONFIG_KEY_PREFIX)
        and key not in KNOWN_GR_CONFIG_KEYS
    )
    if unknown_keys:
        unknown = ", ".join(repr(key) for key in unknown_keys)
        raise ValueError(f"unknown vLLM-GR additional_config key(s): {unknown}")


def _as_string_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"vllm-gr {name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"vllm-gr {name} keys must be strings")
    return cast(Mapping[str, object], value)


def _reject_unknown_mapping_keys(
    mapping: Mapping[str, object],
    allowed: frozenset[str],
    name: str,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"unknown vllm-gr {name} key(s): {rendered}")


def _normalize_artifact_path(
    value: object,
    *,
    base_dir: str | Path | None,
) -> object:
    if value is None or not isinstance(value, str) or not value.strip():
        return value
    artifact_path = Path(value).expanduser()
    if artifact_path.is_absolute():
        return str(artifact_path.resolve(strict=False))
    if base_dir is None:
        raise ValueError("vllm-gr constraint_table.path must be absolute for mapping input")
    normalized_base = Path(base_dir).expanduser().resolve(strict=False)
    return str((normalized_base / artifact_path).resolve(strict=False))


def gr_config_from_mapping(
    raw: object,
    *,
    base_dir: str | Path | None = None,
) -> GRConfig:
    """Build a validated config from a canonical schema-v1 mapping."""

    mapping = _as_string_mapping(raw, "configuration")
    _reject_unknown_mapping_keys(mapping, _TOP_LEVEL_KEYS, "configuration")
    if "schema_version" not in mapping:
        raise ValueError("vllm-gr configuration requires schema_version")

    beam_raw = _as_string_mapping(mapping.get("beam", {}), "beam")
    _reject_unknown_mapping_keys(beam_raw, _BEAM_KEYS, "beam")
    table_raw = _as_string_mapping(mapping.get("constraint_table", {}), "constraint_table")
    _reject_unknown_mapping_keys(table_raw, _CONSTRAINT_TABLE_KEYS, "constraint_table")
    if "max_top_k" not in table_raw:
        raise ValueError("vllm-gr configuration requires constraint_table.max_top_k")

    graph_enabled = beam_raw.get("graph_enabled", True)
    if not isinstance(graph_enabled, bool):
        raise ValueError("vllm-gr beam.graph_enabled must be a boolean")

    table = ConstraintTableConfig(
        enabled=validate_constraint_table_enabled(table_raw.get("enabled", False)),
        backend=validate_constraint_table_backend(table_raw.get("backend", "cuda")),
        path=validate_constraint_table_path(
            _normalize_artifact_path(table_raw.get("path"), base_dir=base_dir)
        ),
        format=validate_constraint_table_format(table_raw.get("format", CONSTRAINT_TABLE_FORMAT)),
        artifact_digest=validate_artifact_digest(table_raw.get("artifact_digest")),
        tokenizer_digest=validate_tokenizer_digest(table_raw.get("tokenizer_digest")),
        max_top_k=validate_constraint_table_max_top_k(table_raw["max_top_k"]),
        candidate_width=validate_candidate_width(table_raw.get("candidate_width")),
    )
    return GRConfig(
        attention_backend=validate_attention_backend(mapping.get("attention_backend")),
        beam_graph_enabled=graph_enabled,
        enable_short_suffix_attention=validate_enable_short_suffix_attention(
            beam_raw.get("enable_short_suffix_attention", True)
        ),
        beam_graph_batch_size=validate_beam_graph_batch_size(beam_raw.get("graph_batch_size", 1)),
        beam_graph_widths=validate_beam_graph_widths(beam_raw.get("graph_widths")),
        beam_online_batch_sizes=validate_beam_online_batch_sizes(
            beam_raw.get("online_batch_sizes", DEFAULT_ONLINE_BATCH_SIZES)
        ),
        beam_max_width=validate_beam_max_width(beam_raw.get("max_width")),
        beam_max_decode_steps=validate_beam_max_decode_steps(
            beam_raw.get("max_decode_steps", _BEAM_MAX_DECODE_STEPS_DEFAULT)
        ),
        beam_worker_decision=validate_beam_worker_decision(beam_raw.get("worker_decision")),
        beam_engine_driven=validate_beam_engine_driven(beam_raw.get("engine_driven")),
        beam_execution_mode=validate_beam_execution_mode(beam_raw.get("execution_mode")),
        schema_version=validate_schema_version(
            mapping.get("schema_version", GR_CONFIG_SCHEMA_VERSION)
        ),
        constraint_table=table,
        constrained_generation=validate_constrained_generation(
            mapping.get("constrained_generation")
        ),
    )


def _reject_lossy_canonical_conversion(config: GRConfig) -> None:
    legacy_values = []
    if config.catalog_path is not None:
        legacy_values.append("catalog_path")
    if config.constraint_backend is not ConstraintBackend.TRIE:
        legacy_values.append("constraint_backend")
    if config.vocab_offset is not None:
        legacy_values.append("vocab_offset")
    if config.vocab_slice_size is not None:
        legacy_values.append("vocab_slice_size")
    if config.npu_decode_execution is not None:
        legacy_values.append("npu_decode_execution")
    if legacy_values:
        raise ValueError(
            "cannot encode legacy-only GRConfig field(s) in canonical schema-v1: "
            + ", ".join(legacy_values)
        )


def gr_config_to_mapping(config: GRConfig) -> dict[str, object]:
    """Convert a canonical config to JSON/pickle-safe primitive values."""

    if not isinstance(config, GRConfig):
        raise TypeError("config must be a GRConfig")
    _reject_lossy_canonical_conversion(config)
    table = config.constraint_table
    if table is None:
        raise ValueError("cannot encode GRConfig without canonical constraint_table state")
    beam: dict[str, object] = {
        "graph_enabled": config.beam_graph_enabled,
        "max_width": config.beam_max_width,
        "max_decode_steps": config.beam_max_decode_steps,
    }
    if not config.enable_short_suffix_attention:
        beam["enable_short_suffix_attention"] = False
    if config.beam_graph_batch_size != 1:
        beam["graph_batch_size"] = config.beam_graph_batch_size
    if config.beam_graph_widths is not None:
        beam["graph_widths"] = list(config.beam_graph_widths)
    if config.beam_online_batch_sizes != DEFAULT_ONLINE_BATCH_SIZES:
        beam["online_batch_sizes"] = list(config.beam_online_batch_sizes)
    # Keep omitted values omitted so existing schema-v1 artifacts round-trip
    # byte-for-byte at the mapping level.  Explicit False is significant: it
    # is the CUDA runtime-default opt-out and must survive Driver -> Worker
    # serialization just like explicit True.
    if config.beam_worker_decision is not None:
        beam["worker_decision"] = config.beam_worker_decision
    if config.beam_execution_mode is not None:
        beam["execution_mode"] = config.beam_execution_mode.value

    constraint_table: dict[str, object] = {
        "enabled": table.enabled,
        "backend": table.backend.value,
        "path": table.path,
        "format": table.format,
        "artifact_digest": table.artifact_digest,
        "tokenizer_digest": table.tokenizer_digest,
        "max_top_k": table.max_top_k,
    }
    # Omitted means K == beam_max_width; keep it omitted so pre-decoupling
    # schema-v1 artifacts round-trip byte-for-byte at the mapping level.
    if table.candidate_width is not None:
        constraint_table["candidate_width"] = table.candidate_width

    return {
        "schema_version": config.schema_version,
        "beam": beam,
        "constraint_table": constraint_table,
        "attention_backend": (
            config.attention_backend.value if config.attention_backend is not None else None
        ),
        **(
            {"constrained_generation": config.constrained_generation}
            if config.constrained_generation is not None
            else {}
        ),
    }


def load_gr_config(path: str | Path) -> GRConfig:
    """Load canonical schema-v1 YAML on the Driver and normalize its paths."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - packaging contract
        raise RuntimeError("PyYAML is required to load a vLLM-GR config") from exc

    config_path = Path(path).expanduser().resolve(strict=False)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
    except OSError as exc:
        raise ValueError(f"cannot read vllm-gr config {str(config_path)!r}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid vllm-gr YAML {str(config_path)!r}: {exc}") from exc
    if raw is None:
        raise ValueError(f"vllm-gr config {str(config_path)!r} is empty")
    return gr_config_from_mapping(raw, base_dir=config_path.parent)


def normalize_gr_config_input(value: GRConfigInput) -> GRConfig:
    if isinstance(value, GRConfig):
        return value
    if isinstance(value, (str, Path)):
        return load_gr_config(value)
    if isinstance(value, Mapping):
        return gr_config_from_mapping(value)
    raise TypeError(
        f"vllm_gr_config must be a YAML path, mapping, or GRConfig; got {type(value).__name__}"
    )


def _legacy_keys_present(
    additional_config: Mapping[object, object],
) -> list[str]:
    return sorted(key for key in KNOWN_GR_CONFIG_KEYS if key in additional_config)


def parse_gr_config(vllm_config: object | None) -> GRConfig:
    if vllm_config is None:
        return GRConfig()
    additional_config = getattr(vllm_config, "additional_config", None)
    if additional_config is None:
        return GRConfig()
    if not isinstance(additional_config, Mapping):
        # Upstream also accepts opaque SupportsHash configuration objects.
        # They cannot carry primitive vllm_gr_* keys, so leave them untouched
        # and treat them as an empty GR configuration on the read path.
        return GRConfig()
    _reject_unknown_gr_keys(additional_config)
    canonical_present = GR_CONFIG_KEY in additional_config
    legacy_keys = _legacy_keys_present(additional_config)
    if canonical_present and legacy_keys:
        raise ValueError(
            "canonical additional_config['vllm_gr'] cannot be combined with "
            "legacy vLLM-GR additional_config key(s): "
            + ", ".join(repr(key) for key in legacy_keys)
        )
    if canonical_present:
        return gr_config_from_mapping(additional_config[GR_CONFIG_KEY])
    return GRConfig(
        catalog_path=validate_catalog_path(additional_config.get(CATALOG_PATH_KEY)),
        constraint_backend=validate_constraint_backend(
            additional_config.get(CONSTRAINT_BACKEND_KEY, "trie")
        ),
        attention_backend=validate_attention_backend(additional_config.get(ATTENTION_BACKEND_KEY)),
        vocab_offset=validate_vocab_offset(additional_config.get(VOCAB_OFFSET_KEY)),
        vocab_slice_size=validate_vocab_slice_size(additional_config.get(VOCAB_SLICE_SIZE_KEY)),
        enable_short_suffix_attention=validate_enable_short_suffix_attention(
            additional_config.get(SHORT_SUFFIX_ATTENTION_KEY, True)
        ),
        beam_graph_enabled=validate_beam_graph_enabled(
            additional_config.get(BEAM_GRAPH_ENABLED_KEY, True)
        ),
        beam_graph_batch_size=validate_beam_graph_batch_size(
            additional_config.get(BEAM_GRAPH_BATCH_SIZE_KEY, 1)
        ),
        beam_graph_widths=validate_beam_graph_widths(additional_config.get(BEAM_GRAPH_WIDTHS_KEY)),
        beam_online_batch_sizes=validate_beam_online_batch_sizes(
            additional_config.get(BEAM_ONLINE_BATCH_SIZES_KEY, DEFAULT_ONLINE_BATCH_SIZES)
        ),
        beam_max_width=validate_beam_max_width(additional_config.get(BEAM_MAX_WIDTH_KEY)),
        beam_max_decode_steps=validate_beam_max_decode_steps(
            additional_config.get(BEAM_MAX_DECODE_STEPS_KEY, _BEAM_MAX_DECODE_STEPS_DEFAULT)
        ),
        npu_decode_execution=validate_npu_decode_execution(
            additional_config.get(NPU_DECODE_EXECUTION_KEY)
        ),
        beam_worker_decision=validate_beam_worker_decision(
            additional_config.get(BEAM_WORKER_DECISION_KEY)
        ),
        beam_engine_driven=validate_beam_engine_driven(
            additional_config.get(BEAM_ENGINE_DRIVEN_KEY)
        ),
        beam_execution_mode=validate_beam_execution_mode(
            additional_config.get(BEAM_EXECUTION_MODE_KEY)
        ),
        constrained_generation=validate_constrained_generation(
            additional_config.get(CONSTRAINED_GENERATION_KEY)
        ),
    )


def write_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, MutableMapping):
        raise ValueError("vLLM-GR additional_config must be a mutable mapping")
    _reject_unknown_gr_keys(additional_config)
    if GR_CONFIG_KEY in additional_config:
        raise ValueError(
            "legacy vLLM-GR config cannot be combined with canonical additional_config['vllm_gr']"
        )
    if gr_config.constraint_table is not None:
        raise ValueError(
            "write_gr_config cannot serialize canonical constraint_table state; "
            "use write_canonical_gr_config"
        )

    catalog_path = validate_catalog_path(gr_config.catalog_path)
    constraint_backend = validate_constraint_backend(gr_config.constraint_backend)
    attention_backend = validate_attention_backend(gr_config.attention_backend)
    vocab_offset = validate_vocab_offset(gr_config.vocab_offset)
    vocab_slice_size = validate_vocab_slice_size(gr_config.vocab_slice_size)
    npu_decode_execution = validate_npu_decode_execution(gr_config.npu_decode_execution)
    constrained_generation = validate_constrained_generation(gr_config.constrained_generation)

    if catalog_path is None:
        additional_config.pop(CATALOG_PATH_KEY, None)
    else:
        additional_config[CATALOG_PATH_KEY] = catalog_path

    if constraint_backend.value == "trie":
        additional_config.pop(CONSTRAINT_BACKEND_KEY, None)
    else:
        additional_config[CONSTRAINT_BACKEND_KEY] = constraint_backend.value

    if attention_backend is None:
        additional_config.pop(ATTENTION_BACKEND_KEY, None)
    else:
        additional_config[ATTENTION_BACKEND_KEY] = attention_backend.value

    if vocab_offset is None:
        additional_config.pop(VOCAB_OFFSET_KEY, None)
    else:
        additional_config[VOCAB_OFFSET_KEY] = vocab_offset

    if vocab_slice_size is None:
        additional_config.pop(VOCAB_SLICE_SIZE_KEY, None)
    else:
        additional_config[VOCAB_SLICE_SIZE_KEY] = vocab_slice_size

    if gr_config.enable_short_suffix_attention:
        additional_config.pop(SHORT_SUFFIX_ATTENTION_KEY, None)
    else:
        additional_config[SHORT_SUFFIX_ATTENTION_KEY] = False

    if constrained_generation is None:
        additional_config.pop(CONSTRAINED_GENERATION_KEY, None)
    else:
        additional_config[CONSTRAINED_GENERATION_KEY] = constrained_generation

    if gr_config.beam_graph_enabled:
        additional_config.pop(BEAM_GRAPH_ENABLED_KEY, None)
    else:
        additional_config[BEAM_GRAPH_ENABLED_KEY] = False

    if gr_config.beam_graph_batch_size == 1:
        additional_config.pop(BEAM_GRAPH_BATCH_SIZE_KEY, None)
    else:
        additional_config[BEAM_GRAPH_BATCH_SIZE_KEY] = gr_config.beam_graph_batch_size

    if gr_config.beam_graph_widths is None:
        additional_config.pop(BEAM_GRAPH_WIDTHS_KEY, None)
    else:
        additional_config[BEAM_GRAPH_WIDTHS_KEY] = list(gr_config.beam_graph_widths)

    if gr_config.beam_max_width is None:
        additional_config.pop(BEAM_MAX_WIDTH_KEY, None)
    else:
        additional_config[BEAM_MAX_WIDTH_KEY] = gr_config.beam_max_width

    if gr_config.beam_online_batch_sizes == DEFAULT_ONLINE_BATCH_SIZES:
        additional_config.pop(BEAM_ONLINE_BATCH_SIZES_KEY, None)
    else:
        additional_config[BEAM_ONLINE_BATCH_SIZES_KEY] = list(gr_config.beam_online_batch_sizes)

    if gr_config.beam_max_decode_steps == _BEAM_MAX_DECODE_STEPS_DEFAULT:
        additional_config.pop(BEAM_MAX_DECODE_STEPS_KEY, None)
    else:
        additional_config[BEAM_MAX_DECODE_STEPS_KEY] = gr_config.beam_max_decode_steps

    if npu_decode_execution is None:
        # Unset stays out of the serialized config; the runtime resolves it
        # per platform at load time.
        additional_config.pop(NPU_DECODE_EXECUTION_KEY, None)
    else:
        # Both explicit modes round-trip: an explicit 'cascade' opt-out must
        # survive a write so it cannot be silently re-defaulted to xllm.
        additional_config[NPU_DECODE_EXECUTION_KEY] = npu_decode_execution.value

    beam_worker_decision = validate_beam_worker_decision(gr_config.beam_worker_decision)
    if beam_worker_decision is None:
        # Unset stays out of the serialized config; the driver resolves
        # platform defaults before worker initialization.
        additional_config.pop(BEAM_WORKER_DECISION_KEY, None)
    else:
        # Both explicit values round-trip (an opt-out must survive a write).
        additional_config[BEAM_WORKER_DECISION_KEY] = beam_worker_decision

    # Accepted only at ingress; never propagate a retired control to workers.
    additional_config.pop(BEAM_ENGINE_DRIVEN_KEY, None)

    if gr_config.beam_execution_mode is None:
        additional_config.pop(BEAM_EXECUTION_MODE_KEY, None)
    else:
        additional_config[BEAM_EXECUTION_MODE_KEY] = gr_config.beam_execution_mode.value


def write_canonical_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    """Write a complete canonical primitive mapping into ``VllmConfig``."""

    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, MutableMapping):
        raise ValueError("vLLM-GR additional_config must be a mutable mapping")
    _reject_unknown_gr_keys(additional_config)
    legacy_keys = _legacy_keys_present(additional_config)
    if legacy_keys:
        raise ValueError(
            "canonical vLLM-GR config cannot be combined with legacy "
            "additional_config key(s): " + ", ".join(repr(key) for key in legacy_keys)
        )
    additional_config[GR_CONFIG_KEY] = gr_config_to_mapping(gr_config)
    # EngineArgs reads the defaults before applying the canonical mapping.
    # UniProc keeps that same config object; it must not retain the old parse.
    try:
        object.__setattr__(vllm_config, "_vllm_gr_parsed_config", None)
    except (AttributeError, TypeError):
        pass
