# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import pytest

from tests.helpers.hardware import (
    CARD_COUNTS,
    PLATFORMS,
    SKUS,
    DeviceInventory,
    Hardware,
    marker_expression,
    parse_target,
    probe_devices,
    validate_inventory,
)

pytest_plugins = ("tests.level_plugin",)
_DEVICES: pytest.StashKey[dict[str, DeviceInventory]] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--loops", action="store", default=1, type=int, help="number to test loops")
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--test-hardware", type=parse_target, help="host, cuda:<SKU> or npu:<SKU>")


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_DEVICES] = {}
    target = config.getoption("test_hardware")
    if target:
        selection = marker_expression(target)
        expr = config.option.markexpr
        config.option.markexpr = f"({expr}) and ({selection})" if expr else selection


def pytest_runtest_setup(item: pytest.Item) -> None:
    # Pytest has already evaluated skip/skipif (including the slow-test skip).
    marks = {mark.name for mark in item.iter_markers()}
    target = item.config.getoption("test_hardware")
    counts = [n for n in CARD_COUNTS if f"cards_{n}" in marks]
    models = {sku for skus in SKUS.values() for sku in skus if sku in marks}
    if not (target or models or counts):
        return
    backends = [backend for backend, mark in PLATFORMS.items() if mark in marks]
    if len(counts) > 1 or ((models or counts) and not backends):
        raise pytest.UsageError(f"{item.nodeid}: invalid hardware markers")
    if not backends:
        return
    devices = item.config.stash[_DEVICES]
    failures = []
    for backend in backends:
        if target and backend != target.backend:
            continue
        allowed = tuple(sku for sku in SKUS[backend] if sku in models)
        if models and not allowed:
            continue
        try:
            if backend not in devices:
                devices[backend] = probe_devices(backend)
            inventory = devices[backend]
            validate_inventory(target or Hardware(backend, allowed or SKUS[backend]), inventory)
        except (
            ImportError,
            RuntimeError,
            ValueError,
            AssertionError,
            AttributeError,
            OSError,
        ) as exc:
            if target:
                pytest.exit(
                    f"Hardware preflight failed: {exc}", returncode=pytest.ExitCode.USAGE_ERROR
                )
            failures.append(str(exc))
            continue
        required = counts[0] if counts else 1
        if len(inventory.names) >= required:
            return
        failures.append(f"requires {required} {backend} cards, {len(inventory.names)} visible")
    if not failures:
        raise pytest.UsageError(f"{item.nodeid}: accelerator model requires its platform marker")
    pytest.skip("Hardware unavailable: " + "; ".join(failures))


# Marker registration lives in pyproject.toml so IDE test runners and
# `pytest --markers` see the same list. Only `slow`/`slow_test` skip themselves,
# matching the upstream convention that hardware markers are descriptive and
# selected with `-m` by the runner.
_SLOW_MARKERS = ("slow", "slow_test")
# Every test declares where it runs; see docs/contributing/ci/test_levels.md#platform-markers.
_TIER_MARKERS = ("cpu_test", "gpu_test", "npu_test")


def _require_tier_markers(items: list[pytest.Item]) -> None:
    # Directory conftests under tests/kernels and tests/e2e add their defaults
    # before this hook runs, and pytest's own -m deselection runs after it, so
    # this sees every collected item.
    missing = sorted(
        {
            item.nodeid.split("::")[0]
            for item in items
            if not any(item.get_closest_marker(marker) for marker in _TIER_MARKERS)
        }
    )
    if missing:
        raise pytest.UsageError(
            "Test modules without a tier marker (cpu_test, gpu_test or npu_test); "
            "see docs/contributing/ci/test_levels.md#platform-markers:\n  " + "\n  ".join(missing)
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _require_tier_markers(items)
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if any(marker in item.keywords for marker in _SLOW_MARKERS):
            item.add_marker(skip_slow)


@pytest.fixture  # type: ignore
def loops(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--loops"))
