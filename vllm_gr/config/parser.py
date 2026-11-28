# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""One public configuration format for files, Python, CLI, and worker transport."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import fields
from enum import Enum
from functools import cache
from pathlib import Path
from threading import Lock
from typing import Any, Final, TypeAlias

import yaml

from vllm_gr.config.models import (
    OPTION_KEYS,
    ConstraintConfig,
    ConstraintMode,
    ConstraintTableSpec,
    DeploymentOptions,
    GRConfig,
)
from vllm_gr.config.schema import config_fields
from vllm_gr.config.validation import validate_constraint_mode


def merge_settings(base: Mapping, override: Mapping) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merge = merge_gr_settings if key == "gr" else merge_settings
            value = merge(merged[key], value)
        merged[key] = value
    return merged


def merge_gr_settings(base: Mapping, override: Mapping) -> dict:
    """Merge GR mappings recursively; explicit values replace inherited values."""
    inherited = dict(base)
    if override.get("constraint_mode") == "none":
        inherited.pop("constraint_table", None)
    elif "constraint_mode" not in override and isinstance(
        override.get("constraint_table"), Mapping
    ):
        # A table supplied by this layer selects constrained generation.
        inherited["constraint_mode"] = "table"
    return merge_settings(inherited, override)


COMMON_ENGINE_CONFIG = Path(__file__).with_name("default_config.yaml").resolve()


_common_document_lock = Lock()


@cache
def _load_common_document() -> dict:
    return _read_settings(COMMON_ENGINE_CONFIG, ())


def _common_document() -> dict:
    # functools.cache permits concurrent misses. Serialize the initial file read.
    with _common_document_lock:
        return _load_common_document()


def common_engine_defaults() -> dict:
    """Return native vLLM defaults independent of the shared startup snapshot."""
    return {
        key.replace("-", "_"): deepcopy(value)
        for key, value in _common_document().items()
        if key != "gr"
    }


def read_settings(path: str | Path, parents: tuple[Path, ...] = ()) -> dict:
    path = Path(path).expanduser().resolve()
    if path in parents:
        chain = " -> ".join(map(str, (*parents, path)))
        raise ValueError(f"Configuration import cycle: {chain}")
    if path == COMMON_ENGINE_CONFIG:
        return deepcopy(_common_document())
    return _read_settings(path, parents)


def _read_settings(path: Path, parents: tuple[Path, ...]) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(data, dict) or any(not isinstance(k, str) for k in data):
        raise ValueError(f"Expected a mapping with string keys in {path}")
    # Native long options accept dashes and underscores. Canonicalize before
    # merging so aliases in different files refer to the same option.
    data = {key.replace("_", "-"): value for key, value in data.items()}
    # Resolve paths at the point of declaration, including inherited tables.
    gr = data.get("gr", {})
    if not isinstance(gr, dict):
        raise ValueError(f"Expected a mapping under gr in {path}")
    table = gr.get("constraint_table")
    if isinstance(table, dict) and isinstance(table.get("path"), str):
        table["path"] = str((path.parent / Path(table["path"]).expanduser()).resolve())
    if isinstance(gr.get("prefix_calibration_path"), str):
        gr["prefix_calibration_path"] = str(
            (path.parent / Path(gr["prefix_calibration_path"]).expanduser()).resolve()
        )
    if "import" not in data:
        return data
    source = data.pop("import")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"import in {path} must name one YAML file")
    return merge_settings(read_settings(path.parent / source, (*parents, path)), data)


GR_CONFIG_KEY: Final = "vllm_gr"
BACKEND_TRANSPORT_KEYS: Final = frozenset({"vllm_gr_limits"})
GRConfigInput: TypeAlias = str | Path
CAPACITY_FIELDS = ("beam_max_width", "beam_candidate_width")
CONFIG_ATTRIBUTES = {
    spec.canonical_path[1]: item.name
    for item, spec in config_fields(GRConfig)
    if spec.canonical_path is not None
    and len(spec.canonical_path) == 2
    and spec.canonical_path[0] == "gr"
}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} requires a mapping with string keys")
    return dict(value)


def _unknown_keys(value: Mapping, allowed: set | frozenset, name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} settings: {sorted(unknown, key=str)}")


def _input_path(value: str, base_dir: str | Path | None) -> str:
    if not value.strip():
        return value
    path = Path(value).expanduser()
    if base_dir is not None:
        path = Path(base_dir) / path
    return str(path.resolve()) if path.is_absolute() or base_dir is not None else value


def _table(value: object) -> ConstraintTableSpec:
    raw = _mapping(value, "gr.constraint_table")
    _unknown_keys(raw, {"path", "backend", "format", "max_top_k", "digests"}, "gr.constraint_table")
    digests = _mapping(raw.get("digests", {}), "gr.constraint_table.digests")
    _unknown_keys(digests, {"artifact", "tokenizer"}, "gr.constraint_table.digests")
    if type(raw.get("max_top_k")) is not int or raw["max_top_k"] < 1:
        raise ValueError("gr.constraint_table.max_top_k requires a positive integer")
    missing = [key for key in ("artifact", "tokenizer") if digests.get(key) is None]
    if missing:
        raise ValueError(f"gr.constraint_table.digests requires {missing}")
    kwargs = {k: raw[k] for k in ("backend", "format", "max_top_k") if k in raw}
    return ConstraintTableSpec(
        path=raw.get("path"),
        artifact_digest=digests.get("artifact"),
        tokenizer_digest=digests.get("tokenizer"),
        enabled=True,
        **kwargs,
    )


def _parse_constraint(gr: Mapping[str, Any]) -> ConstraintConfig | None:
    table = gr.get("constraint_table")
    if "constraint_mode" not in gr:
        if table is not None:
            raise ValueError("gr.constraint_table requires gr.constraint_mode=table")
        return None
    mode = validate_constraint_mode(gr["constraint_mode"])
    if mode is ConstraintMode.NONE:
        if table is not None:
            raise ValueError("gr.constraint_mode=none forbids gr.constraint_table")
        return ConstraintConfig(mode)
    if table is None:
        raise ValueError("gr.constraint_mode=table requires gr.constraint_table")
    return ConstraintConfig(mode, _table(table))


def gr_config_from_mapping(
    mapping: Mapping[str, object], *, base_dir: str | Path | None = None
) -> GRConfig:
    """Resolve paths on the driver before publishing the configuration."""
    return _decode_gr_config(mapping, resolve_paths=True, base_dir=base_dir)


def _decode_gr_config(
    mapping: Mapping[str, object], *, resolve_paths: bool, base_dir: str | Path | None = None
) -> GRConfig:
    document = _mapping(mapping, "Configuration")
    _unknown_keys(document, {"gr", "attention-config"}, "configuration")
    if "gr" not in document:
        raise ValueError("Configuration requires a gr mapping, using default_config.yaml format")
    gr = _mapping(document["gr"], "gr")
    _unknown_keys(
        gr,
        set(CONFIG_ATTRIBUTES) | OPTION_KEYS | {"constraint_mode", "constraint_table"},
        "gr",
    )
    kwargs = {attribute: gr[key] for key, attribute in CONFIG_ATTRIBUTES.items() if key in gr}
    options = {k: v for k, v in gr.items() if k in OPTION_KEYS}
    if resolve_paths:
        calibration = options.get("prefix_calibration_path")
        if isinstance(calibration, str):
            options["prefix_calibration_path"] = _input_path(calibration, base_dir)
        table = gr.get("constraint_table")
        if isinstance(table, Mapping) and isinstance(path := table.get("path"), str):
            gr["constraint_table"] = {**table, "path": _input_path(path, base_dir)}
    kwargs["deployment"] = DeploymentOptions.from_mapping(options)
    kwargs["constraint"] = _parse_constraint(gr)
    attention = document.get("attention-config")
    if attention is not None:
        attention = _mapping(attention, "attention-config")
        kwargs["attention_backend"] = attention.get("backend")
    return GRConfig(**kwargs)


def _primitive(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def gr_config_to_mapping(config: GRConfig) -> dict[str, object]:
    if not isinstance(config, GRConfig):
        raise TypeError("config must be a GRConfig")
    gr = {}
    options = config.deployment
    for public, attribute in CONFIG_ATTRIBUTES.items():
        value = getattr(config, attribute)
        # Schedules already determine both capacities. Standalone scalar
        # capacities carry no per-step request defaults.
        if public in CAPACITY_FIELDS and (options.beam_width is not None or value is None):
            continue
        gr[public] = _primitive(value)
    gr.update({f.name: _primitive(getattr(options, f.name)) for f in fields(options)})
    constraint = config.constraint
    if constraint is not None:
        gr["constraint_mode"] = constraint.mode.value
        table = constraint.table
        gr["constraint_table"] = (
            None
            if table is None
            else {
                "path": table.path,
                "backend": table.backend.value,
                "format": table.format,
                "max_top_k": table.max_top_k,
                "digests": {"artifact": table.artifact_digest, "tokenizer": table.tokenizer_digest},
            }
        )
    result: dict[str, object] = {"gr": gr}
    if config.attention_backend is not None:
        result["attention-config"] = {"backend": config.attention_backend.value}
    return result


def resolve_startup_config(
    document: Mapping,
    overrides: Mapping | None = None,
    *,
    worker_decision_default: bool | None = None,
) -> tuple[GRConfig, str, int | None]:
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("gr must be a mapping")
    document = _mapping(document, "Configuration")
    retired = {
        "schema_version",
        "schema-version",
        "beam",
        "deployment",
        "constraint",
    } & document.keys()
    if retired:
        raise ValueError(
            f"Removed configuration structure {sorted(retired)}; use gr keys from default_config.yaml"
        )
    base = read_settings(COMMON_ENGINE_CONFIG)["gr"]
    effective = merge_gr_settings(merge_gr_settings(base, document.get("gr", {})), overrides or {})
    host = effective.pop("admin_host")
    port = effective.pop("admin_port")
    if not isinstance(host, str) or not host:
        raise ValueError("gr.admin_host requires a hostname or address")
    if port is not None and (type(port) is not int or not 1 <= port <= 65535):
        raise ValueError("gr.admin_port must be in 1..65535")
    if effective.get("worker_decision") is None and worker_decision_default is not None:
        effective["worker_decision"] = worker_decision_default
    payload: dict[str, object] = {"gr": effective}
    attention = document.get("attention-config")
    if isinstance(attention, Mapping) and attention.get("backend") == "CUSTOM":
        payload["attention-config"] = attention
    return gr_config_from_mapping(payload, base_dir=Path.cwd()), host, port


def load_gr_config(
    path: str | Path = COMMON_ENGINE_CONFIG, overrides: Mapping | None = None
) -> GRConfig:
    if not isinstance(path, (str, Path)):
        raise TypeError("config must be a YAML file path")
    return resolve_startup_config(read_settings(path), overrides)[0]


def validate_additional_keys(additional: Mapping) -> None:
    invalid = sorted(
        key
        for key in additional
        if isinstance(key, str) and key.startswith("vllm_gr_") and key not in BACKEND_TRANSPORT_KEYS
    )
    if invalid:
        raise ValueError(
            f"Removed vLLM-GR configuration keys: {invalid}. "
            "Supply GR settings with gr=..., --gr-config, or --config"
        )


def parse_gr_config(vllm_config: object | None) -> GRConfig:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, Mapping):
        return GRConfig()
    validate_additional_keys(additional)
    if GR_CONFIG_KEY not in additional:
        return GRConfig()
    return _decode_gr_config(additional[GR_CONFIG_KEY], resolve_paths=False)


def write_canonical_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, MutableMapping):
        raise ValueError("vLLM-GR additional_config must be a mutable mapping")
    validate_additional_keys(additional)
    additional[GR_CONFIG_KEY] = gr_config_to_mapping(gr_config)
    object.__setattr__(vllm_config, "_vllm_gr_parsed_config", gr_config)
