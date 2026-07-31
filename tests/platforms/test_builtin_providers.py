# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for the built-in CUDA and NPU providers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.platforms import cuda, npu
from vllm_gr.platforms.builtins import register_builtin_backends
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
    resolve_backend,
)


def setup_function() -> None:
    _reset_backend_registry_for_tests()


def teardown_function() -> None:
    _reset_backend_registry_for_tests()


def test_builtin_providers_register_idempotently() -> None:
    register_builtin_backends()
    register_builtin_backends()

    assert resolve_backend("cuda") is cuda.CUDA_BACKEND
    assert resolve_backend("npu") is npu.NPU_BACKEND


def test_cuda_provider_lazily_returns_platform_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_runner = object()
    gpu_attention = object()
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return {
            "vllm_gr.v1.worker.gpu_model_runner_patch": gpu_runner,
            "vllm_gr.v1.attention.backends.beam_attn_gpu": gpu_attention,
        }[name]

    monkeypatch.setattr(cuda, "import_module", import_module)

    assert cuda.CUDA_BACKEND.get_model_runner_hooks() is gpu_runner
    assert cuda.CUDA_BACKEND.get_attention_execution() is gpu_attention
    assert imported == [
        "vllm_gr.v1.worker.gpu_model_runner_patch",
        "vllm_gr.v1.attention.backends.beam_attn_gpu",
    ]


def test_npu_provider_lazily_returns_platform_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npu_runner = object()
    npu_attention = object()
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return {
            "vllm_gr.v1.worker.npu_model_runner_patch": npu_runner,
            "vllm_gr.v1.attention.backends.beam_attn_npu": npu_attention,
        }[name]

    monkeypatch.setattr(npu, "import_module", import_module)

    assert npu.NPU_BACKEND.get_model_runner_hooks() is npu_runner
    assert npu.NPU_BACKEND.get_attention_execution() is npu_attention
    assert imported == [
        "vllm_gr.v1.worker.npu_model_runner_patch",
        "vllm_gr.v1.attention.backends.beam_attn_npu",
    ]


def test_cuda_runtime_patch_is_noop() -> None:
    assert cuda.CUDA_BACKEND.apply_runtime_patches() == 0
    assert cuda.CUDA_BACKEND.apply_runtime_patches(force=True) == 0


def test_npu_runtime_patches_preserve_order_and_sum_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import v1

    events: list[tuple[str, bool | None]] = []

    def apply_attention(*, force: bool = False) -> int:
        events.append(("attention", force))
        return 2

    def patch_worker() -> int:
        events.append(("worker", None))
        return 3

    ascend_compat = SimpleNamespace(
        apply_ascend_attention_backend_patch=apply_attention,
        patch_npu_worker_init_device=patch_worker,
    )
    monkeypatch.setattr(v1, "ascend_compat", ascend_compat, raising=False)

    assert npu.NPU_BACKEND.apply_runtime_patches(force=True) == 5
    assert events == [("attention", True), ("worker", None)]


def test_cuda_environment_check_validates_platform_and_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import _env_check

    vllm_module = ModuleType("vllm")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="cuda")  # type: ignore[attr-defined]
    vllm_module.platforms = platforms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    report = {"vllm": "0.22.1"}
    check_versions = mock.Mock(return_value=report)
    monkeypatch.setattr(_env_check, "check_runtime_versions", check_versions)

    assert cuda.CUDA_BACKEND.check_environment() == {
        "vllm": "0.22.1",
        "platform": "cuda",
    }
    check_versions.assert_called_once_with()

    platforms.current_platform.device_type = "cpu"
    with pytest.raises(RuntimeError, match="requires the CUDA platform"):
        cuda.CUDA_BACKEND.check_environment()


def test_npu_environment_check_delegates_complete_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import _env_check

    expected = {"platform": "npu"}
    check_ascend = mock.Mock(return_value=expected)
    monkeypatch.setattr(
        _env_check,
        "check_ascend_environment",
        check_ascend,
    )

    assert npu.NPU_BACKEND.check_environment() is expected
    check_ascend.assert_called_once_with(require_npu_platform=True)
