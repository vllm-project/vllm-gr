# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import pytest

from tests.helpers.artifact_fixtures import (  # noqa: F401
    onerec_dataset,
    onerec_iid_constraints,
    onerec_model,
    onerec_pid_constraints,
)

pytest_plugins = ("tests.level_plugin",)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--loops", action="store", default=1, type=int, help="number to test loops")
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


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
            if not any(marker in item.keywords for marker in _TIER_MARKERS)
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
