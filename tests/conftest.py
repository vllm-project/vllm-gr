# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import pytest

from tools.hardware_marks import (
    CARD_MARKERS,
    HARDWARE_PLATFORM_FIXTURE,
    PLATFORM_MARKERS,
    PLATFORM_SKUS,
    matches_sku,
)

pytest_plugins = ("tests.level_plugin",)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--loops", action="store", default=1, type=int, help="number to test loops")
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


# Platform markers are registered in pyproject.toml and selected with `-m`.
_SLOW_MARKERS = ("slow", "slow_test")
# Every test declares where it runs; see docs/tools/README.md#test-tiers.
_TIER_MARKERS = ("cpu_test", "gpu_test", "npu_test")
_DEFAULT_SKUS = {"cuda": "A100", "npu": "A3"}


def _require_tier_markers(items: list[pytest.Item]) -> None:
    # Directory conftests under tests/kernels and tests/st add their defaults
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
            "see docs/tools/README.md#test-tiers:\n  " + "\n  ".join(missing)
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _require_tier_markers(items)
    run_slow = config.getoption("--run-slow")
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        platforms = [
            platform for platform, marker in PLATFORM_MARKERS.items() if marker in item.keywords
        ]
        for platform in platforms:
            if not any(sku in item.keywords for sku in PLATFORM_SKUS[platform]):
                item.add_marker(getattr(pytest.mark, _DEFAULT_SKUS[platform]))
        if platforms and not any(marker in item.keywords for marker in CARD_MARKERS.values()):
            item.add_marker(getattr(pytest.mark, CARD_MARKERS[1]))
        if not run_slow and any(marker in item.keywords for marker in _SLOW_MARKERS):
            item.add_marker(skip_slow)


@pytest.hookimpl(tryfirst=True)  # type: ignore[misc]
def pytest_runtest_setup(item: pytest.Item) -> None:
    counts = [count for count, marker in CARD_MARKERS.items() if marker in item.keywords]
    requirements = {
        platform: tuple(sku for sku in skus if sku in item.keywords)
        for platform, skus in PLATFORM_SKUS.items()
        if any(sku in item.keywords for sku in skus)
    }
    if not requirements and counts:
        requirements = {
            platform: () for platform, mark in PLATFORM_MARKERS.items() if mark in item.keywords
        }
    if not requirements:
        return
    if len(counts) > 1:
        raise pytest.UsageError(f"{item.nodeid} has conflicting card requirements: {counts}")

    count = counts[0] if counts else 1
    failures = []
    for platform, allowed in requirements.items():
        visible = _visible_devices(platform)
        if len(visible) >= count and all(matches_sku(platform, name, allowed) for name in visible):
            return
        failures.append(f"{count}+ {platform} devices ({', '.join(allowed)}); visible: {visible}")
    pytest.skip("Requires " + " or ".join(failures))


def _visible_devices(platform: str) -> list[str]:
    if platform == "cuda":
        import torch

        if torch.version.cuda is None:
            return []
        backend = torch.cuda
    else:
        try:
            from torch_npu import npu as backend
        except ModuleNotFoundError as exc:
            if exc.name != "torch_npu":
                raise
            return []
    if not backend.is_available():
        return []
    return [backend.get_device_name(index) for index in range(backend.device_count())]


@pytest.fixture(name=HARDWARE_PLATFORM_FIXTURE)  # type: ignore[misc]
def hardware_platform(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture  # type: ignore
def loops(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--loops"))


@pytest.fixture(autouse=True)  # type: ignore
def reset_vocab_slice_config() -> None:
    from vllm_gr.ops.sliced_logits import set_vocab_slice_config

    set_vocab_slice_config(None, None)
    yield
    set_vocab_slice_config(None, None)
