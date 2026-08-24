# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""YAML and ``VllmConfig.additional_config`` configuration boundaries."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Final, TypeAlias

from vllm_gr.config.models import (
    CONSTRAINT_TABLE_FORMAT,
    DEFAULT_BEAM_MAX_DECODE_STEPS,
    DEFAULT_CONSTRAINT_MAX_TOP_K,
    GR_CONFIG_SCHEMA_VERSION,
    BeamConfig,
    ConstraintTableConfig,
    GRConfig,
)

GR_CONFIG_KEY: Final = "vllm_gr"
GRConfigInput: TypeAlias = str | Path | Mapping[str, object] | GRConfig
_TOP_LEVEL_KEYS = frozenset({"schema_version", "beam", "constraint_table", "attention_backend"})
_BEAM_KEYS = frozenset({"max_width", "max_decode_steps", "graph_enabled"})
_CONSTRAINT_TABLE_KEYS = frozenset(
    {
        "enabled",
        "backend",
        "path",
        "format",
        "artifact_digest",
        "tokenizer_digest",
        "max_top_k",
    }
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"vllm-gr {name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"vllm-gr {name} keys must be strings")
    return value  # type: ignore[return-value]


def _reject_unknown(mapping: Mapping[str, object], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"unknown vllm-gr {name} key(s): {rendered}")


def gr_config_from_mapping(raw: object) -> GRConfig:
    mapping = _mapping(raw, "configuration")
    _reject_unknown(mapping, _TOP_LEVEL_KEYS, "configuration")
    if "schema_version" not in mapping:
        raise ValueError("vllm-gr configuration requires schema_version")

    beam_raw = _mapping(mapping.get("beam", {}), "beam")
    _reject_unknown(beam_raw, _BEAM_KEYS, "beam")
    table_raw = _mapping(mapping.get("constraint_table", {}), "constraint_table")
    _reject_unknown(table_raw, _CONSTRAINT_TABLE_KEYS, "constraint_table")

    beam = BeamConfig(
        max_width=beam_raw.get("max_width"),
        max_decode_steps=beam_raw.get("max_decode_steps", DEFAULT_BEAM_MAX_DECODE_STEPS),
        graph_enabled=beam_raw.get("graph_enabled", True),
    )
    table = ConstraintTableConfig(
        enabled=table_raw.get("enabled", False),
        backend=table_raw.get("backend", "cpu"),
        path=table_raw.get("path"),
        format=table_raw.get("format", CONSTRAINT_TABLE_FORMAT),
        artifact_digest=table_raw.get("artifact_digest"),
        tokenizer_digest=table_raw.get("tokenizer_digest"),
        max_top_k=table_raw.get("max_top_k", DEFAULT_CONSTRAINT_MAX_TOP_K),
    )
    return GRConfig(
        schema_version=mapping.get("schema_version", GR_CONFIG_SCHEMA_VERSION),
        beam=beam,
        constraint_table=table,
        attention_backend=mapping.get("attention_backend"),
    )


def gr_config_to_mapping(config: GRConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "beam": {
            "max_width": config.beam.max_width,
            "max_decode_steps": config.beam.max_decode_steps,
            "graph_enabled": config.beam.graph_enabled,
        },
        "constraint_table": {
            "enabled": config.constraint_table.enabled,
            "backend": config.constraint_table.backend.value,
            "path": config.constraint_table.path,
            "format": config.constraint_table.format,
            "artifact_digest": config.constraint_table.artifact_digest,
            "tokenizer_digest": config.constraint_table.tokenizer_digest,
            "max_top_k": config.constraint_table.max_top_k,
        },
        "attention_backend": (
            config.attention_backend.value if config.attention_backend is not None else None
        ),
    }


def load_gr_config(path: str | Path) -> GRConfig:
    """Read and validate the Driver-only YAML file."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - packaging contract
        raise RuntimeError("PyYAML is required to read --vllm-gr-config") from exc

    config_path = Path(path).expanduser()
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read vllm-gr config {str(config_path)!r}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid vllm-gr YAML {str(config_path)!r}: {exc}") from exc
    if raw is None:
        raise ValueError(f"vllm-gr config {str(config_path)!r} is empty")
    if isinstance(raw, Mapping):
        table_raw = raw.get("constraint_table")
        if isinstance(table_raw, Mapping):
            artifact_path_raw = table_raw.get("path")
            if isinstance(artifact_path_raw, str) and artifact_path_raw.strip():
                artifact_path = Path(artifact_path_raw).expanduser()
                if not artifact_path.is_absolute():
                    artifact_path = config_path.parent / artifact_path
                normalized_raw = dict(raw)
                normalized_table = dict(table_raw)
                normalized_table["path"] = str(artifact_path.resolve(strict=False))
                normalized_raw["constraint_table"] = normalized_table
                raw = normalized_raw
    return gr_config_from_mapping(raw)


def normalize_gr_config_input(value: GRConfigInput) -> GRConfig:
    """Normalize one Driver-side Python configuration input.

    Only YAML inputs have a directory against which Artifact paths can be
    resolved. Mapping and ``GRConfig`` inputs therefore retain the common
    requirement that ``constraint_table.path`` is absolute.
    """

    if isinstance(value, GRConfig):
        return value
    if isinstance(value, (str, Path)):
        return load_gr_config(value)
    if isinstance(value, Mapping):
        return gr_config_from_mapping(value)
    raise TypeError(
        f"vllm_gr_config must be a YAML path, mapping, or GRConfig; got {type(value).__name__}"
    )


def _reject_legacy_keys(additional_config: Mapping[object, object]) -> None:
    legacy = sorted(
        key for key in additional_config if isinstance(key, str) and key.startswith("vllm_gr_")
    )
    if legacy:
        rendered = ", ".join(repr(key) for key in legacy)
        raise ValueError(
            f"legacy vLLM-GR additional_config key(s) are no longer supported: {rendered}; "
            "use --vllm-gr-config"
        )


def parse_gr_config(vllm_config: object | None) -> GRConfig:
    if vllm_config is None:
        return GRConfig()
    additional_config = getattr(vllm_config, "additional_config", None)
    if additional_config is None:
        return GRConfig()
    if not isinstance(additional_config, Mapping):
        return GRConfig()
    _reject_legacy_keys(additional_config)
    raw = additional_config.get(GR_CONFIG_KEY)
    return GRConfig() if raw is None else gr_config_from_mapping(raw)


def write_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, MutableMapping):
        raise ValueError("vLLM-GR additional_config must be a mutable mapping")
    _reject_legacy_keys(additional_config)
    if gr_config.enabled:
        additional_config[GR_CONFIG_KEY] = gr_config_to_mapping(gr_config)
    else:
        additional_config.pop(GR_CONFIG_KEY, None)
