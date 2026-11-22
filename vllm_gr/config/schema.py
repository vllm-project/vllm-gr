# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Field declarations shared by configuration readers, writers and validation."""

from __future__ import annotations

from dataclasses import MISSING, Field, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ConfigFieldSpec:
    validator: str
    canonical_path: tuple[str, ...] | None = None
    required: bool = False


_FIELD_CACHE: dict[type[Any], tuple[tuple[Field[Any], ConfigFieldSpec], ...]] = {}


def config_fields(cls: type[Any]) -> tuple[tuple[Field[Any], ConfigFieldSpec], ...]:
    """Read and check declarations once, without constructing a config instance."""
    if cls in _FIELD_CACHE:
        return _FIELD_CACHE[cls]
    result = []
    paths: set[tuple[str, ...]] = set()
    for item in fields(cls):
        spec = item.metadata.get("gr")
        if not isinstance(spec, ConfigFieldSpec):
            raise TypeError(f"{cls.__name__}.{item.name} requires ConfigFieldSpec metadata")
        missing_default = item.default is MISSING and item.default_factory is MISSING
        if spec.required != missing_default:
            raise ValueError(f"{cls.__name__}.{item.name} must declare a default or be required")
        if spec.canonical_path is not None:
            path = spec.canonical_path
            if not path or any(not part for part in path):
                raise ValueError(f"invalid canonical config path for {item.name}")
            if any(path[: len(other)] == other or other[: len(path)] == path for other in paths):
                raise ValueError(f"overlapping canonical config path: {'.'.join(path)}")
            paths.add(path)
        result.append((item, spec))
    _FIELD_CACHE[cls] = tuple(result)
    return _FIELD_CACHE[cls]


def field_default(item: Field[Any]) -> object:
    """Return the declared default, calling a factory separately for each use."""
    if item.default is not MISSING:
        return item.default
    if item.default_factory is not MISSING:
        return item.default_factory()
    raise ValueError(f"required config field {item.name} has no default")


def config_default(cls: type[Any], name: str) -> object:
    for item, _ in config_fields(cls):
        if item.name == name:
            return field_default(item)
    raise ValueError(f"unknown config field: {cls.__name__}.{name}")
