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
    MAX_BEAM_WIDTH,
    MAX_CANDIDATE_WIDTH,
    OPTION_KEYS,
    ConstraintConfig,
    ConstraintMode,
    ConstraintTableSpec,
    DeploymentOptions,
    GRConfig,
    runtime_constraint_table,
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
BACKEND_TRANSPORT_KEYS: Final = frozenset({"vllm_gr_backend_plan", "vllm_gr_backend_options"})
GRConfigInput: TypeAlias = str | Path
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


def _step_widths(settings: Mapping) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    widths = settings.get("beam_width")
    candidates = settings.get("candidate_width")
    if widths is None:
        if candidates is not None:
            raise ValueError("gr.candidate_width requires gr.beam_width")
        return None, None
    for key, value, ceiling in (
        ("beam_width", widths, MAX_BEAM_WIDTH),
        ("candidate_width", candidates, MAX_CANDIDATE_WIDTH),
    ):
        if (
            not isinstance(value, (list, tuple))
            or not value
            or any(type(v) is not int or not 1 <= v <= ceiling for v in value)
        ):
            raise ValueError(f"gr.{key} requires a nonempty list of integers in 1..{ceiling}")
    assert candidates is not None
    if len(widths) != len(candidates):
        raise ValueError("gr.candidate_width must have one entry for each gr.beam_width entry")
    if widths[0] != candidates[0]:
        raise ValueError("first gr.candidate_width must equal first gr.beam_width")
    if any(w > previous * k for previous, k, w in zip(widths, candidates[1:], widths[1:])):
        raise ValueError("gr.beam_width exceeds the number of available candidates for a step")
    if (len(set(widths)) > 1 or len(set(candidates)) > 1) and len(widths) > 3:
        raise ValueError("varying gr.beam_width or gr.candidate_width supports up to three steps")
    return tuple(widths), tuple(candidates)


def _table(value: object, base_dir: str | Path | None) -> ConstraintTableSpec:
    raw = _mapping(value, "gr.constraint_table")
    _unknown_keys(raw, {"path", "backend", "format", "max_top_k", "digests"}, "gr.constraint_table")
    digests = _mapping(raw.get("digests", {}), "gr.constraint_table.digests")
    _unknown_keys(digests, {"artifact", "tokenizer"}, "gr.constraint_table.digests")
    path = raw.get("path")
    if isinstance(path, str):
        artifact = Path(path).expanduser()
        if not artifact.is_absolute():
            if base_dir is None:
                raise ValueError("gr.constraint_table.path must be absolute for mapping input")
            artifact = Path(base_dir) / artifact
        path = str(artifact.resolve())
    if type(raw.get("max_top_k")) is not int or raw["max_top_k"] < 1:
        raise ValueError("gr.constraint_table.max_top_k requires a positive integer")
    missing = [key for key in ("artifact", "tokenizer") if digests.get(key) is None]
    if missing:
        raise ValueError(f"gr.constraint_table.digests requires {missing}")
    kwargs = {k: raw[k] for k in ("backend", "format", "max_top_k") if k in raw}
    return ConstraintTableSpec(
        path=path,
        artifact_digest=digests.get("artifact"),
        tokenizer_digest=digests.get("tokenizer"),
        enabled=True,
        **kwargs,
    )


def gr_config_from_mapping(
    mapping: Mapping[str, object], *, base_dir: str | Path | None = None
) -> GRConfig:
    document = _mapping(mapping, "Configuration")
    _unknown_keys(document, {"gr", "attention-config"}, "configuration")
    if "gr" not in document:
        raise ValueError("Configuration requires a gr mapping, using default_config.yaml format")
    gr = _mapping(document["gr"], "gr")
    _unknown_keys(
        gr, set(CONFIG_ATTRIBUTES) | OPTION_KEYS | {"constraint_mode", "constraint_table"}, "gr"
    )
    widths, candidates = _step_widths(gr)
    kwargs = {
        attribute: gr[key]
        for key, attribute in CONFIG_ATTRIBUTES.items()
        if key in gr and key not in ("beam_width", "candidate_width")
    }
    kwargs["beam_max_width"] = max(widths) if widths else None
    candidate_capacity = max(candidates) if candidates else None
    if widths and widths == candidates and len(set(widths)) > 1:
        candidate_capacity = None
    kwargs["beam_candidate_width"] = candidate_capacity

    options = {k: v for k, v in gr.items() if k in OPTION_KEYS}
    calibration = options.get("prefix_calibration_path")
    if isinstance(calibration, str):
        calibration_path = Path(calibration).expanduser()
        if not calibration_path.is_absolute():
            if base_dir is None:
                raise ValueError("gr.prefix_calibration_path must be absolute for mapping input")
            calibration_path = Path(base_dir) / calibration_path
        options["prefix_calibration_path"] = str(calibration_path.resolve())
    kwargs["deployment"] = DeploymentOptions.from_mapping(options)
    mode = validate_constraint_mode(gr.get("constraint_mode", "none"))
    table = gr.get("constraint_table")
    if mode is ConstraintMode.NONE and table is not None:
        raise ValueError("gr.constraint_mode=none forbids gr.constraint_table")
    if mode is ConstraintMode.TABLE:
        if table is None:
            raise ValueError("gr.constraint_mode=table requires gr.constraint_table")
        if widths is not None and len(widths) != 3:
            raise ValueError("constraint tables require exactly three Beam steps")
        table = _table(table, base_dir)
    kwargs["constraint"] = ConstraintConfig(mode, table)
    attention = document.get("attention-config")
    if attention is not None:
        attention = _mapping(attention, "attention-config")
        kwargs["attention_backend"] = attention.get("backend")
    config = GRConfig(**kwargs)
    if widths is not None and len(widths) > config.beam_max_decode_steps:
        raise ValueError(
            "gr.max_decode_steps must be at least the number of entries in gr.beam_width"
        )
    if (
        config.deployment.result_width is not None
        and widths
        and config.deployment.result_width > widths[-1]
    ):
        raise ValueError("gr.result_width must not exceed the final gr.beam_width")
    return config


def _primitive(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def gr_config_to_mapping(config: GRConfig) -> dict[str, object]:
    if not isinstance(config, GRConfig):
        raise TypeError("config must be a GRConfig")
    retired = [
        name for name in ("vocab_offset", "vocab_slice_size") if getattr(config, name) is not None
    ]
    if retired:
        raise ValueError(f"Removed configuration controls: {retired}")
    gr = {}
    for public, attribute in CONFIG_ATTRIBUTES.items():
        gr[public] = _primitive(getattr(config, attribute))
    options = config.deployment
    gr.update({f.name: _primitive(getattr(options, f.name)) for f in fields(options)})
    if options.beam_width is None and config.beam_max_width is not None:
        gr["beam_width"] = [config.beam_max_width] * config.beam_max_decode_steps
    if options.candidate_width is None and config.beam_max_width is not None:
        width = config.beam_candidate_width
        table_config = runtime_constraint_table(config)
        if width is None and table_config is not None:
            width = table_config.candidate_width
        gr["candidate_width"] = [width or config.beam_max_width] * config.beam_max_decode_steps
    constraint = config.constraint_config
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


def _load_document(
    document: Mapping,
    overrides: Mapping | None = None,
    *,
    worker_decision_default: bool | None = None,
) -> GRConfig:
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
    if effective.get("worker_decision") is None and worker_decision_default is not None:
        effective["worker_decision"] = worker_decision_default
    payload: dict[str, object] = {"gr": effective}
    attention = document.get("attention-config")
    if isinstance(attention, Mapping) and attention.get("backend") == "CUSTOM":
        payload["attention-config"] = attention
    return gr_config_from_mapping(payload, base_dir=Path.cwd())


def load_gr_config(path: str | Path, overrides: Mapping | None = None) -> GRConfig:
    if not isinstance(path, (str, Path)):
        raise TypeError("config must be a YAML file path")
    return _load_document(read_settings(path), overrides)


def load_engine_gr_config(
    path: GRConfigInput | None,
    additional: object,
    overrides: Mapping | None = None,
    *,
    startup_document: Mapping | None = None,
    worker_decision_default: bool | None = None,
) -> GRConfig:
    if path is not None and not isinstance(path, (str, Path)):
        raise TypeError("config must be a YAML file path")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("gr must be a mapping")
    if isinstance(additional, Mapping) and (
        GR_CONFIG_KEY in additional or _retired_gr_keys(additional)
    ):
        raise ValueError("Supply GR settings with gr=..., --gr-config, or --config")
    if startup_document is not None:
        return _load_document(
            startup_document, overrides, worker_decision_default=worker_decision_default
        )
    if path is not None:
        return _load_document(
            read_settings(path), overrides, worker_decision_default=worker_decision_default
        )
    return _load_document({"gr": {}}, overrides, worker_decision_default=worker_decision_default)


def _retired_gr_keys(additional: Mapping) -> list[str]:
    return [
        key
        for key in additional
        if isinstance(key, str) and key.startswith("vllm_gr_") and key not in BACKEND_TRANSPORT_KEYS
    ]


def parse_gr_config(vllm_config: object | None) -> GRConfig:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, Mapping):
        return GRConfig()
    retired = _retired_gr_keys(additional)
    if retired:
        raise ValueError(f"Removed vLLM-GR configuration keys: {sorted(retired)}; use gr")
    if GR_CONFIG_KEY not in additional:
        return GRConfig()
    return gr_config_from_mapping(additional[GR_CONFIG_KEY])


def write_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    write_canonical_gr_config(vllm_config, gr_config)


def write_canonical_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, MutableMapping):
        raise ValueError("vLLM-GR additional_config must be a mutable mapping")
    retired = _retired_gr_keys(additional)
    known = {spec.legacy_key for _, spec in config_fields(GRConfig)} | {
        "vllm_gr_catalog_path",
        "vllm_gr_constraint_backend",
        "vllm_gr_vocab_offset",
        "vllm_gr_vocab_slice_size",
        "vllm_gr_constrained_generation",
        "vllm_gr_beam_engine_driven",
    }
    unknown = set(retired) - known
    if unknown:
        raise ValueError(f"unknown vLLM-GR configuration keys: {sorted(unknown)}")
    payload = gr_config_to_mapping(gr_config)
    for key in retired:
        additional.pop(key)
    additional[GR_CONFIG_KEY] = payload
    object.__setattr__(vllm_config, "_vllm_gr_parsed_config", gr_config)
