# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--loops", action="store", default=1, type=int, help="number to test loops")
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "npu: require a usable Ascend NPU runtime")
    config.addinivalue_line("markers", "gpu: require a usable NVIDIA GPU runtime")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _npu_available() -> bool:
    try:
        import torch

        npu = getattr(torch, "npu", None)
        return npu is not None and bool(npu.is_available())
    except ImportError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not _cuda_available():
        skip_gpu = pytest.mark.skip(reason="CUDA unavailable")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    if not _npu_available():
        skip_npu = pytest.mark.skip(reason="NPU runtime unavailable")
        for item in items:
            if "npu" in item.keywords:
                item.add_marker(skip_npu)


@pytest.fixture  # type: ignore
def loops(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--loops"))


@pytest.fixture(autouse=True)  # type: ignore
def reset_vocab_slice_config() -> None:
    from vllm_gr.ops.sliced_logits import set_vocab_slice_config

    set_vocab_slice_config(None, None)
    yield
    set_vocab_slice_config(None, None)
