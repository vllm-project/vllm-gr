# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Assign the reviewed CI test levels before pytest's ``-m`` filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tests.level_policy import (
    LEVELS,
    ClassificationError,
    canonical_nodeid,
    resolve_level,
)

_DEFAULT_MANIFEST = Path(__file__).with_name("level_inventory.json")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("vllm-gr test levels")
    group.addoption(
        "--strict-test-levels",
        action="store_true",
        default=False,
        help="Reject collected tests without exactly one valid L1-L5 classification.",
    )
    group.addoption(
        "--test-level-manifest",
        default=str(_DEFAULT_MANIFEST),
        help="Path to the test-level inventory (defaults to the repository inventory).",
    )


def pytest_configure(config: pytest.Config) -> None:
    for level in sorted(LEVELS):
        config.addinivalue_line("markers", f"{level}: vllm-gr {level.upper()} CI test level")


def _read_inventory(path: str) -> dict[str, object]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise pytest.UsageError(f"Cannot read test-level inventory {path}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or not isinstance(data.get("tests"), dict)
    ):
        raise pytest.UsageError(f"Invalid test-level inventory: {path}")
    return cast(dict[str, object], data["tests"])


@pytest.hookimpl(tryfirst=True)  # type: ignore[misc]
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    inventory = _read_inventory(config.getoption("--test-level-manifest"))
    strict = config.getoption("--strict-test-levels")
    errors: list[str] = []
    for item in items:
        key = canonical_nodeid(item.nodeid)
        params = getattr(getattr(item, "callspec", None), "params", {})
        options: dict[str, object] = {"loops": config.getoption("loops", default=1)}
        existing = {name for name in LEVELS if item.get_closest_marker(name) is not None}
        try:
            level = resolve_level(inventory, item.nodeid, params, options, strict=False)
            if level is None:
                if len(existing) == 1:
                    level = next(iter(existing))
                elif len(existing) > 1:
                    raise ClassificationError(
                        f"Conflicting explicit levels for {item.nodeid}: {existing}"
                    )
                elif strict:
                    raise ClassificationError(
                        f"Unclassified test {key}; assign exactly one @pytest.mark.l1 through "
                        "@pytest.mark.l5, or add a parameter rule to tests/level_inventory.json. "
                        "See docs/tools/README.md#test-levels-and-ci-selection."
                    )
                else:
                    continue
            if existing and existing != {level}:
                raise ClassificationError(
                    f"Explicit levels {sorted(existing)} conflict with {level} for {item.nodeid}"
                )
            if not existing:
                item.add_marker(getattr(pytest.mark, level))
        except ClassificationError as exc:
            # Broken reviewed policies must never silently assign a wrong
            # level, even outside strict mode. Unknown tests are the only
            # permitted omission in ordinary local pytest invocations.
            errors.append(str(exc))
    if errors:
        details = "\n".join(errors[:20])
        if len(errors) > 20:
            details += f"\n... and {len(errors) - 20} more"
        raise pytest.UsageError("Test-level classification failed:\n" + details)
