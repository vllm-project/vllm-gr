# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for the CUDA backend provider."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.platforms import cuda


def test_cuda_provider_lazily_returns_platform_hooks(
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
    assert cuda.CUDA_BACKEND.get_attention_hooks() is gpu_attention
    assert imported == [
        "vllm_gr.v1.worker.gpu_model_runner_patch",
        "vllm_gr.v1.attention.backends.beam_attn_gpu",
    ]


def test_cuda_runtime_patch_is_noop() -> None:
    assert cuda.CUDA_BACKEND.apply_runtime_patches() == 0
    assert cuda.CUDA_BACKEND.apply_runtime_patches(force=True) == 0


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


def test_provider_releases_cuda_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from vllm_gr.platforms import memory

    cleanup = mock.Mock()
    monkeypatch.setattr(memory, "cleanup_torch_backend", cleanup)

    cuda.CUDA_BACKEND.release_accelerator_memory()

    cleanup.assert_called_once_with(torch.cuda, collect_ipc=True)
