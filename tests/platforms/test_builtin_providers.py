# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts for the built-in CUDA and NPU providers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.platforms.ascend import provider as npu
from vllm_gr.platforms.cuda import provider as cuda
from vllm_gr.platforms.loader import load_backend_plugins
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
    resolve_backend,
)


def setup_function() -> None:
    _reset_backend_registry_for_tests()


def teardown_function() -> None:
    _reset_backend_registry_for_tests()


def test_builtin_providers_register_idempotently() -> None:
    load_backend_plugins()
    load_backend_plugins()

    assert resolve_backend("cuda") is cuda.CUDA_BACKEND
    assert resolve_backend("ascend") is npu.NPU_BACKEND


def test_cuda_provider_lazily_returns_platform_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_runner = object()
    gpu_attention = object()
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return {
            "vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch": gpu_runner,
            "vllm_gr.v1.attention.backends.beam_attn_gpu": gpu_attention,
        }[name]

    monkeypatch.setattr(cuda, "import_module", import_module)

    assert cuda.CUDA_BACKEND.get_model_runner_hooks() is gpu_runner
    assert cuda.CUDA_BACKEND.get_attention_execution() is gpu_attention
    assert imported == [
        "vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch",
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
            "vllm_gr.v1.worker.npu.npu_model_runner_patch": npu_runner,
            "vllm_gr.v1.attention.backends.beam_attn_npu": npu_attention,
        }[name]

    monkeypatch.setattr(npu, "import_module", import_module)

    assert npu.NPU_BACKEND.get_model_runner_hooks() is npu_runner
    assert npu.NPU_BACKEND.get_attention_execution() is npu_attention
    assert imported == [
        "vllm_gr.v1.worker.npu.npu_model_runner_patch",
        "vllm_gr.v1.attention.backends.beam_attn_npu",
    ]


def test_cuda_runtime_patch_is_noop() -> None:
    assert cuda.CUDA_BACKEND.apply_runtime_patches() == 0
    assert cuda.CUDA_BACKEND.apply_runtime_patches(force=True) == 0


def test_npu_runtime_patches_preserve_order_and_sum_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import npu_spawn_patch, v1

    events: list[tuple[str, bool | None]] = []

    def apply_spawn() -> bool:
        events.append(("spawn", None))
        return True

    def patch_worker() -> int:
        events.append(("worker", None))
        return 3

    ascend_compat = SimpleNamespace(
        patch_npu_worker_init_device=patch_worker,
    )
    monkeypatch.setattr(v1, "ascend_compat", ascend_compat, raising=False)
    monkeypatch.setattr(npu_spawn_patch, "apply_npu_spawn_patch", apply_spawn)

    assert npu.NPU_BACKEND.apply_runtime_patches(force=True) == 4
    assert events == [("spawn", None), ("worker", None)]


def test_cuda_environment_check_validates_platform_and_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import _env_check

    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(  # type: ignore[attr-defined]
        _enum="CUDA", device_type="cuda", device_name="cuda"
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    report = {"vllm": "0.30.0"}
    check_versions = mock.Mock(return_value=report)
    monkeypatch.setattr(_env_check, "check_runtime_versions", check_versions)

    assert cuda.CUDA_BACKEND.check_environment() == {
        "vllm": "0.30.0",
        "platform": "cuda",
    }
    check_versions.assert_called_once_with()

    platforms.current_platform = SimpleNamespace(  # type: ignore[attr-defined]
        _enum="ROCM", device_type="cuda", device_name="cuda"
    )
    with pytest.raises(RuntimeError, match="requires the native CUDA platform"):
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


pytestmark = pytest.mark.cpu_test
