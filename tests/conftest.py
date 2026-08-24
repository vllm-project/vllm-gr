# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--loops", action="store", default=1, type=int, help="number to test loops")
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--require-npu",
        action="store_true",
        default=False,
        help="fail collection unless a usable Ascend NPU and constrained Top-K op exist",
    )


def _enforce_required_npu() -> None:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise pytest.UsageError(f"--require-npu: torch-npu is unavailable: {exc}") from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise pytest.UsageError("--require-npu: no usable Ascend NPU is available")

    from vllm_gr.ops.npu.rec_constrained_topk import is_available, unavailable_reason

    if not is_available():
        raise pytest.UsageError(
            f"--require-npu: NPU rec_constrained_topk is unavailable: {unavailable_reason()}"
        )
    try:
        torch.empty(1, device="npu")
        torch.npu.synchronize()
    except Exception as exc:
        raise pytest.UsageError(f"--require-npu: NPU allocation failed: {exc}") from exc


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers",
        "npu_integration: require a complete, usable Ascend NPU runtime",
    )
    if config.getoption("--require-npu"):
        _enforce_required_npu()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
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
