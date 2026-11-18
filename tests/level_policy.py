# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Declarative classification of collected pytest cases.

Rule keys are pytest node IDs without parameter IDs; specifications match
actual collected parameters and command-line options.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, cast

LEVELS = frozenset({"l1", "l2", "l3", "l4", "l5"})


class ClassificationError(ValueError):
    """The inventory or a parameter specification cannot be classified."""


def canonical_nodeid(nodeid: str) -> str:
    """Remove only pytest's parameter suffix, preserving module and class."""
    return nodeid.split("[", 1)[0].replace("\\", "/")


def normalize(value: Any) -> Any:
    """Make common pytest parameter values comparable with JSON literals."""
    if isinstance(value, tuple):
        return [normalize(v) for v in value]
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return normalize(asdict(value))
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "dtype":
        return str(value).removeprefix("torch.")
    if value_type.__module__ == "torch" and value_type.__name__ == "device":
        return value.type
    return value


def _lookup(ref: str, params: Mapping[str, Any], options: Mapping[str, Any]) -> Any:
    parts = ref.split(".")
    if len(parts) < 2 or parts[0] not in ("params", "options"):
        raise ClassificationError(f"Invalid reference {ref!r}")
    obj: Any = params if parts[0] == "params" else options
    try:
        for part in parts[1:]:
            obj = obj[part] if isinstance(obj, Mapping) else getattr(obj, part)
    except (KeyError, AttributeError, TypeError) as exc:
        raise ClassificationError(f"Missing parameter or option {ref!r}") from exc
    return normalize(obj)


def matches(expr: Mapping[str, Any], params: Mapping[str, Any], options: Mapping[str, Any]) -> bool:
    """Evaluate a JSON-safe Boolean rule without eval or repository imports."""
    if len(expr) == 1 and "all" in expr:
        return all(matches(e, params, options) for e in expr["all"])
    if len(expr) == 1 and "any" in expr:
        return any(matches(e, params, options) for e in expr["any"])
    if len(expr) == 1 and "not" in expr:
        return not matches(expr["not"], params, options)
    if "ref" not in expr or len(expr) != 2:
        raise ClassificationError(f"Invalid selector: {expr!r}")
    value = _lookup(expr["ref"], params, options)
    op = next(k for k in expr if k != "ref")
    target = normalize(expr[op])
    try:
        if op == "eq":
            return bool(value == target)
        if op == "in":
            return bool(value in target)
        if op == "lt":
            return bool(value < target)
        if op == "le":
            return bool(value <= target)
        if op == "gt":
            return bool(value > target)
        if op == "ge":
            return bool(value >= target)
    except (TypeError, ValueError) as exc:
        raise ClassificationError(f"Invalid comparison for {expr['ref']!r}") from exc
    raise ClassificationError(f"Unknown comparison {op!r}")


def _references(expr: Mapping[str, Any]) -> set[str]:
    """Check every declared input, including short-circuited branches."""
    if "ref" in expr:
        return {expr["ref"]}
    refs: set[str] = set()
    for group in ("all", "any"):
        for child in expr.get(group, []):
            refs.update(_references(child))
    if "not" in expr:
        refs.update(_references(expr["not"]))
    return refs


def resolve_level(
    inventory: Mapping[str, Any],
    nodeid: str,
    params: Mapping[str, Any],
    options: Mapping[str, Any],
    *,
    strict: bool = True,
) -> str | None:
    """Resolve exactly one level. Unlisted nodes fail only in strict mode."""
    key = canonical_nodeid(nodeid)
    if key not in inventory:
        if strict:
            raise ClassificationError(f"Unclassified test {key}")
        return None
    policy = inventory[key]
    if isinstance(policy, str):
        if policy not in LEVELS:
            raise ClassificationError(f"Invalid level {policy!r} for {key}")
        return policy
    if not isinstance(policy, Mapping):
        raise ClassificationError(f"Invalid policy for {key}")
    found: list[str] = []
    for clause in policy.get("rules", []):
        level = clause.get("level")
        if level not in LEVELS:
            raise ClassificationError(f"Invalid level {level!r} for {key}")
        for ref in _references(clause["when"]):
            _lookup(ref, params, options)
        if matches(clause["when"], params, options):
            found.append(cast(str, level))
    if len(found) > 1:
        raise ClassificationError(f"Overlapping rules for {nodeid}: {found}")
    if found:
        return found[0]
    default = policy.get("default")
    if default not in LEVELS:
        raise ClassificationError(f"No matching rule for {nodeid}")
    return cast(str, default)
