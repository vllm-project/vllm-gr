# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Partition real-model tests requiring access to the in-process engine."""

import os

INPROC_TEST_FILES = frozenset(
    {
        "tests/kernels/test_gpu_dynamic_beam_model.py",
        "tests/kernels/test_gpu_dynamic_beam_graph_parity.py",
        "tests/kernels/test_gpu_dynamic_beam_parity.py",
    }
)


def pytest_collection_modifyitems(config, items):
    selection = os.environ["VLLM_GR_INPROC_TEST_SELECTION"]
    if selection not in {"include", "exclude"}:
        raise ValueError(f"Invalid in-process selection: {selection!r}")
    selected, deselected = [], []
    for item in items:
        test_file = item.nodeid.split("::", 1)[0].replace("\\", "/")
        is_inproc = test_file in INPROC_TEST_FILES
        if is_inproc == (selection == "include"):
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
