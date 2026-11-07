# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--loops", action="store", default=1, type=int, help="number to test loops")
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers",
        "npu_integration: require a complete, usable Ascend NPU runtime",
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: test needs a CUDA device and only runs in the GPU lane"
        ' (select lanes with -m requires_gpu / -m "not requires_gpu";'
        " see tests/gpu_lanes.py)",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Keep GPU-lane files out of CPU lane collection entirely.

    ``-m`` filtering happens *after* import, so a GPU-lane module that
    imports CUDA-only dependencies would abort collection on a CPU-only
    host even though its tests are deselected.  ``run_tests.sh cpu`` sets
    ``VLLM_GR_TEST_LANE=cpu`` and this hook skips those files before they
    are imported; every other invocation collects as before.
    """
    if os.environ.get("VLLM_GR_TEST_LANE") != "cpu":
        return None
    if collection_path.suffix != ".py":
        return None
    from tests.gpu_lanes import requires_gpu

    return requires_gpu(collection_path)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Lane marking must run before the --run-slow early return so the GPU
    # lane (which passes --run-slow) still gets its markers.
    from tests.gpu_lanes import requires_gpu

    gpu_marker = pytest.mark.requires_gpu
    for item in items:
        if requires_gpu(item.path):
            item.add_marker(gpu_marker)

    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture  # type: ignore
def loops(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--loops"))


@pytest.fixture(autouse=True)  # type: ignore
def reset_vocab_slice_config() -> None:
    from vllm_gr.ops.sliced_logits import set_vocab_slice_config

    set_vocab_slice_config(None, None)
    yield
    set_vocab_slice_config(None, None)
