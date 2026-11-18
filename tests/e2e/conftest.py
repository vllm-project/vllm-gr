# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Default the system tests to the GPU tier.

These suites usually drive a real engine. A test that declares any tier itself
keeps it, so NPU suites and host-side helper cases (``cpu_test``) are not
relabelled.
"""

from pathlib import Path

import pytest

# Any declared tier wins, including an explicit cpu_test for host-side cases.
_TIER_MARKERS = ("cpu_test", "gpu_test", "npu_test")
_DIRECTORY = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # pytest passes every collected item to this hook, not only the ones
    # under this directory, so scope the default explicitly.
    for item in items:
        if not item.path.resolve().is_relative_to(_DIRECTORY):
            continue
        if not any(marker in item.keywords for marker in _TIER_MARKERS):
            item.add_marker(pytest.mark.gpu_test)
