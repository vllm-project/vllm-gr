# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the lane classification in :mod:`tests.gpu_lanes` honest.

A renamed or deleted test file must not linger in the explicit lists, and
the policy invariants must hold, otherwise ``-m requires_gpu`` silently
selects the wrong lane.  This file itself lives in the CPU lane.
"""

from __future__ import annotations

from tests.gpu_lanes import CPU_OVERRIDES, GPU_DIRECTORIES, GPU_TEST_FILES, TESTS_ROOT


def _test_files() -> set[str]:
    return {p.relative_to(TESTS_ROOT).as_posix() for p in TESTS_ROOT.rglob("test_*.py")}


def test_listed_files_exist() -> None:
    on_disk = _test_files()
    for name, listed in (("GPU_TEST_FILES", GPU_TEST_FILES), ("CPU_OVERRIDES", CPU_OVERRIDES)):
        missing = sorted(set(listed) - on_disk)
        assert not missing, f"{name} references files that no longer exist: {missing}"


def test_overrides_stay_inside_gpu_directories() -> None:
    strays = sorted(name for name in CPU_OVERRIDES if name.split("/", 1)[0] not in GPU_DIRECTORIES)
    assert not strays, f"CPU_OVERRIDES must only relax GPU directories: {strays}"


def test_gpu_files_stay_outside_gpu_directories() -> None:
    strays = sorted(name for name in GPU_TEST_FILES if name.split("/", 1)[0] in GPU_DIRECTORIES)
    assert not strays, f"GPU_TEST_FILES is redundant for GPU directories: {strays}"


def test_both_lanes_are_populated() -> None:
    on_disk = _test_files()
    gpu = {name for name in on_disk if name.split("/", 1)[0] in GPU_DIRECTORIES}
    gpu = (gpu - set(CPU_OVERRIDES)) | (set(GPU_TEST_FILES) & on_disk)
    assert gpu, "GPU lane is empty; the classification drifted"
    assert on_disk - gpu, "CPU lane is empty; every test was pushed onto the card"
