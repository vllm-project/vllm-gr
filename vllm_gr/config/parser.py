# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import MISSING
from enum import Enum
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

from vllm_gr.config.models import ConstraintTableConfig, GRConfig
from vllm_gr.config.schema import (
    canonical_keys,
    config_fields,
    field_default,
    legacy_key,
    should_emit,
)

GR_CONFIG_KEY: Final = "vllm_gr"
GRConfigInput: TypeAlias = str | Path | Mapping[str, object] | GRConfig
CATALOG_PATH_KEY: Final = legacy_key(GRConfig, "catalog_path")
CONSTRAINT_BACKEND_KEY: Final = legacy_key(GRConfig, "constraint_backend")
ATTENTION_BACKEND_KEY: Final = legacy_key(GRConfig, "attention_backend")
VOCAB_OFFSET_KEY: Final = legacy_key(GRConfig, "vocab_offset")
VOCAB_SLICE_SIZE_KEY: Final = legacy_key(GRConfig, "vocab_slice_size")
PREFILL_FA_VERSION_KEY: Final = legacy_key(GRConfig, "prefill_flash_attn_version")
SHORT_SUFFIX_ATTENTION_KEY: Final = legacy_key(GRConfig, "enable_short_suffix_attention")
BEAM_GRAPH_ENABLED_KEY: Final = legacy_key(GRConfig, "beam_graph_enabled")
BEAM_ADAPTER_CACHE_ENABLED_KEY: Final = legacy_key(GRConfig, "beam_adapter_cache_enabled")
BEAM_GRAPH_BATCH_SIZE_KEY: Final = legacy_key(GRConfig, "beam_graph_batch_size")
BEAM_GRAPH_WIDTHS_KEY: Final = legacy_key(GRConfig, "beam_graph_widths")
BEAM_ONLINE_BATCH_SIZES_KEY: Final = legacy_key(GRConfig, "beam_online_batch_sizes")
BEAM_MAX_WIDTH_KEY: Final = legacy_key(GRConfig, "beam_max_width")
BEAM_MAX_DECODE_STEPS_KEY: Final = legacy_key(GRConfig, "beam_max_decode_steps")
NPU_DECODE_EXECUTION_KEY: Final = legacy_key(GRConfig, "npu_decode_execution")
BEAM_WORKER_DECISION_KEY: Final = legacy_key(GRConfig, "beam_worker_decision")
BEAM_ENGINE_DRIVEN_KEY: Final = legacy_key(GRConfig, "beam_engine_driven")
BEAM_EXECUTION_MODE_KEY: Final = legacy_key(GRConfig, "beam_execution_mode")
BEAM_SID_STAGE_VOCAB_KEY: Final = legacy_key(GRConfig, "beam_sid_stage_vocab")
CONSTRAINED_GENERATION_KEY: Final = legacy_key(GRConfig, "constrained_generation")
GR_CONFIG_KEY_PREFIX: Final = "vllm_gr_"
KNOWN_GR_CONFIG_KEYS: Final = frozenset(
    spec.legacy_key for _, spec in config_fields(GRConfig) if spec.legacy_key is not None
)
_TOP_LEVEL_KEYS: Final = canonical_keys(GRConfig)
_BEAM_KEYS: Final = canonical_keys(GRConfig, ("beam",))
_CONSTRAINT_TABLE_KEYS: Final = canonical_keys(ConstraintTableConfig)


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


def _canonical_kwargs(
    cls: type[Any], mapping: Mapping[str, object], *, prefix: str = ""
) -> dict[str, Any]:
    """Supply only explicit inputs; constructor defaults remain authoritative."""
    result: dict[str, Any] = {}
    for item, spec in config_fields(cls):
        if spec.canonical_path is None:
            continue
        value: object = mapping
        for part in spec.canonical_path:
            value = value.get(part, MISSING) if isinstance(value, Mapping) else MISSING
        name = prefix + ".".join(spec.canonical_path)
        if value is MISSING:
            if spec.canonical_required:
                raise ValueError(f"vllm-gr configuration requires {name}")
            continue
        if spec.canonical_strict_bool and not isinstance(value, bool):
            raise ValueError(f"vllm-gr {name} must be a boolean")
        result[item.name] = value
    return result


def _primitive(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, ConstraintTableConfig):
        return _canonical_mapping(value)
    return value


def _canonical_mapping(config: GRConfig | ConstraintTableConfig) -> dict[str, object]:
    result: dict[str, object] = {}
    for item, spec in config_fields(type(config)):
        path = spec.canonical_path
        value = getattr(config, item.name)
        if path is None or not should_emit(item, value, spec.canonical_emit):
            continue
        target = result
        for part in path[:-1]:
            target = cast(dict[str, object], target.setdefault(part, {}))
        target[path[-1]] = _primitive(value)
    return result


def gr_config_from_mapping(
    raw: object,
    *,
    base_dir: str | Path | None = None,
) -> GRConfig:
    """Build a validated config from a canonical schema-v1 mapping."""
    mapping = _as_string_mapping(raw, "configuration")
    _reject_unknown_mapping_keys(mapping, _TOP_LEVEL_KEYS, "configuration")
    beam_raw = _as_string_mapping(mapping.get("beam", {}), "beam")
    _reject_unknown_mapping_keys(beam_raw, _BEAM_KEYS, "beam")
    table_raw = _as_string_mapping(mapping.get("constraint_table", {}), "constraint_table")
    _reject_unknown_mapping_keys(table_raw, _CONSTRAINT_TABLE_KEYS, "constraint_table")

    values = _canonical_kwargs(GRConfig, {**mapping, "beam": beam_raw})
    table_values = _canonical_kwargs(ConstraintTableConfig, table_raw, prefix="constraint_table.")
    if "path" in table_values:
        table_values["path"] = _normalize_artifact_path(table_values["path"], base_dir=base_dir)
    values["constraint_table"] = ConstraintTableConfig(**table_values)
    return GRConfig(**values)


def _reject_lossy_canonical_conversion(config: GRConfig) -> None:
    legacy_values = [
        item.name
        for item, spec in config_fields(GRConfig)
        if spec.canonical_path is None
        and spec.legacy_key is not None
        and getattr(config, item.name) != field_default(item)
    ]
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
    if config.constraint_table is None:
        raise ValueError("cannot encode GRConfig without canonical constraint_table state")
    return _canonical_mapping(config)


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
        **{
            item.name: additional_config[spec.legacy_key]
            for item, spec in config_fields(GRConfig)
            if spec.legacy_key is not None and spec.legacy_key in additional_config
        }
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

    for item, spec in config_fields(GRConfig):
        if spec.legacy_key is None:
            continue
        value = getattr(gr_config, item.name)
        if should_emit(item, value, spec.legacy_emit):
            additional_config[spec.legacy_key] = _primitive(value)
        else:
            additional_config.pop(spec.legacy_key, None)


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
