# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Additive hardware selection; legacy gates retain their existing behavior."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import cast

import pytest

from tests.helpers.hardware import (
    CARD_COUNTS,
    LEGACY_MARKERS,
    SKUS,
    DeviceInventory,
    Hardware,
    parse_target,
    probe_devices,
    validate_inventory,
)

_TARGET: pytest.StashKey[Hardware | None] = pytest.StashKey()
_INVENTORIES: pytest.StashKey[dict[str, DeviceInventory]] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("vllm-gr hardware").addoption(
        "--test-hardware",
        help="Select host plus compatible tests and verify hardware: host, cuda:H100, npu:A3",
    )


def pytest_configure(config: pytest.Config) -> None:
    value = config.getoption("test_hardware")
    try:
        config.stash[_TARGET] = parse_target(value) if value is not None else None
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    target = config.stash[_TARGET]
    legacy_target = os.environ.get("VLLM_GR_TARGET_DEVICE", "")
    if legacy_target == "cpu":
        legacy_target = "host"
    if target is not None and legacy_target and legacy_target != target.backend:
        raise pytest.UsageError(
            f"--test-hardware={value} conflicts with VLLM_GR_TARGET_DEVICE={legacy_target}"
        )
    config.stash[_INVENTORIES] = {}
    # Also support isolated policy tests loading this plugin outside root conftest.
    markers = {
        "hardware": "Structured hardware requirement; use hardware_test/hardware_marks",
        **{name: f"Test execution platform: {name}" for name in LEGACY_MARKERS},
        **{name: "Legacy platform compatibility" for name in LEGACY_MARKERS.values()},
        **{sku: f"Test hardware SKU ({backend})" for backend in SKUS for sku in SKUS[backend]},
        **{f"cards_{n}": f"Requires {n} accelerator devices" for n in CARD_COUNTS},
    }
    registered = {line.split(":", 1)[0] for line in config.getini("markers")}
    for name, description in markers.items():
        if name not in registered:
            config.addinivalue_line("markers", f"{name}: {description}")


@pytest.fixture  # type: ignore[misc]
def _gr_hardware_variant(request: pytest.FixtureRequest) -> Hardware:
    return cast(Hardware, request.param)


def _requirements(item: pytest.Item) -> list[Hardware]:
    requirements = []
    for mark in item.iter_markers("hardware"):
        if len(mark.args) != 1 or not isinstance(mark.args[0], Hardware) or mark.kwargs:
            raise pytest.UsageError(f"{item.nodeid}: use hardware_test/hardware_marks")
        requirements.append(mark.args[0])
    return requirements


def _normalize(item: pytest.Item) -> None:
    # Aliases are added before pytest -m selection. Accelerator declarations
    # continue to override an inherited cpu_test/host for runner selection.
    names = {mark.name for mark in item.iter_markers()}
    for modern, legacy in LEGACY_MARKERS.items():
        if modern in names or legacy in names:
            for name in (modern, legacy):
                if name not in names:
                    item.add_marker(getattr(pytest.mark, name))
                    names.add(name)
    platforms = set(LEGACY_MARKERS).intersection(names)
    for backend, skus in SKUS.items():
        if set(skus).intersection(names) and backend not in platforms:
            raise pytest.UsageError(f"{item.nodeid}: SKU marker requires platform {backend}")
    counts = {n for n in CARD_COUNTS if f"cards_{n}" in names}
    if len(counts) > 1:
        raise pytest.UsageError(f"{item.nodeid}: conflicting card counts {sorted(counts)}")
    if counts and not platforms.intersection(SKUS):
        raise pytest.UsageError(f"{item.nodeid}: card count requires an accelerator platform")
    _requirements(item)


def _matches(item: pytest.Item, target: Hardware) -> bool:
    requirements = _requirements(item)
    if requirements:
        return any(
            requirement.backend == target.backend
            and bool(set(requirement.models).intersection(target.models))
            for requirement in requirements
        )
    names = {mark.name for mark in item.iter_markers()}
    accelerators = set(SKUS).intersection(names)
    if not accelerators:
        return "host" in names
    if target.backend not in accelerators:
        return False
    models = set(SKUS[target.backend]).intersection(names)
    return not models or target.models[0] in models


@pytest.hookimpl(hookwrapper=True, tryfirst=True)  # type: ignore[misc]
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> Generator[None, None, None]:
    for item in items:
        _normalize(item)
    yield
    # Directory defaults have now run. Respect pytest's -m/-k deselection too.
    for item in items:
        _normalize(item)
    target = config.stash[_TARGET]
    if target is not None:
        removed = [item for item in items if not _matches(item, target)]
        items[:] = [item for item in items if _matches(item, target)]
        if removed:
            config.hook.pytest_deselected(items=removed)


def _inventory(config: pytest.Config, backend: str) -> DeviceInventory:
    inventories = config.stash[_INVENTORIES]
    if backend not in inventories:
        inventories[backend] = probe_devices(backend)
    return cast(DeviceInventory, inventories[backend])


def _verify(item: pytest.Item, target: Hardware) -> None:
    counts = [n for n in CARD_COUNTS if item.get_closest_marker(f"cards_{n}") is not None]
    requirement = Hardware(target.backend, target.models, max(counts, default=1))
    validate_inventory(requirement, _inventory(item.config, target.backend))


def pytest_collection_finish(session: pytest.Session) -> None:
    target = session.config.stash[_TARGET]
    if target is None or target.backend == "host" or session.config.option.collectonly:
        return
    try:
        # Verify even a host-only selection: a claimed accelerator lane must
        # never pass just because only host cases happened to match.
        validate_inventory(target, _inventory(session.config, target.backend))
        for item in session.items:
            if item.get_closest_marker(target.backend) is not None:
                _verify(item, target)
    except (ImportError, RuntimeError, ValueError, AssertionError, AttributeError, OSError) as exc:
        raise pytest.UsageError(f"Hardware preflight failed: {exc}") from exc


def pytest_runtest_setup(item: pytest.Item) -> None:
    requirements = _requirements(item)
    if not requirements or item.config.stash[_TARGET] is not None:
        return
    failures = []
    for requirement in requirements:
        try:
            validate_inventory(requirement, _inventory(item.config, requirement.backend))
            return
        except (
            ImportError,
            RuntimeError,
            ValueError,
            AssertionError,
            AttributeError,
            OSError,
        ) as exc:
            failures.append(str(exc))
    pytest.fail("Hardware preflight failed: " + "; ".join(failures), pytrace=False)
