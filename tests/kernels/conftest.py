# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Default kernel tests to the GPU tier.

Markers are descriptive, so a CPU runner selects them away with ``-m``
rather than relying on an implicit skip. A test that declares any tier itself
(the NPU suites, or host-side reference cases marked ``cpu_test``) keeps it.
"""

import gc
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.helpers.hardware import PLATFORM_MARKERS

# Any declared tier wins, including an explicit cpu_test for host-side cases.
_TIER_MARKERS = PLATFORM_MARKERS
_DIRECTORY = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # pytest passes every collected item to this hook, not only the ones
    # under this directory, so scope the default explicitly.
    for item in items:
        if not item.path.resolve().is_relative_to(_DIRECTORY):
            continue
        if not any(item.get_closest_marker(marker) is not None for marker in _TIER_MARKERS):
            item.add_marker(pytest.mark.gpu_test)
            item.add_marker(pytest.mark.cuda)


@pytest.fixture(autouse=True)  # type: ignore[misc]
def isolate_cuda_graph_pool(request: pytest.FixtureRequest) -> Iterator[None]:
    """Isolate GPU tests from retired pools and deferred graph destruction."""
    if request.node.get_closest_marker("gpu_test") is None:
        yield
        return
    from vllm.platforms import current_platform

    if current_platform.device_type != "cuda":
        yield
        return
    # An automatic cyclic GC inside capture can destroy a previous graph and
    # invalidate the active capture. Collection can resume between tests.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    # vLLM caches the handle at class scope. A previous test may destroy its
    # last graph while an output allocation remains alive; reusing that pool
    # in a later test then trips PyTorch's allocator assertion.
    platform_type = type(current_platform)
    platform_type._global_graph_pool = None
    try:
        yield
    finally:
        # Do not restore a previous handle: its last graph may be gone.
        platform_type._global_graph_pool = None
        if gc_was_enabled:
            gc.enable()
        else:
            gc.disable()
