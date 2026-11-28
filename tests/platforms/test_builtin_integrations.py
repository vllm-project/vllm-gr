# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts for the built-in CUDA and NPU integrations."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from tests.helpers.ascend_platform import AscendGRPlatform
from tests.helpers.cuda_platform import CudaGRPlatform

cuda = CudaGRPlatform()
npu = AscendGRPlatform()


def test_cuda_integration_lazily_returns_platform_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envs = ModuleType("vllm.envs")
    setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)

    gpu_attention = object()
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return {
            "vllm_gr.v1.attention.backends.beam_attn_gpu": gpu_attention,
        }[name]

    monkeypatch.setattr(f"{CudaGRPlatform.__module__}.import_module", import_module)

    assert cuda.get_gr_worker_cls() == "vllm_gr.v1.worker.cuda.worker.CudaGRWorker"
    assert cuda.get_attention_execution() is gpu_attention
    assert imported == [
        "vllm_gr.v1.attention.backends.beam_attn_gpu",
    ]


def test_npu_integration_lazily_returns_platform_modules(
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

    monkeypatch.setattr(f"{AscendGRPlatform.__module__}.import_module", import_module)

    assert npu.get_gr_worker_cls() == "vllm_gr.v1.worker.npu.worker.AscendGRWorker"
    assert npu.get_attention_execution() is npu_attention
    assert imported == [
        "vllm_gr.v1.attention.backends.beam_attn_npu",
    ]


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("mode", ["cascade", "xllm"])  # type: ignore[misc]
def test_ascend_binding_uses_resolved_decode_implementation(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from vllm_gr.platforms.ascend import decode

    monkeypatch.setattr(decode, "engine_mode", lambda config: mode)
    monkeypatch.setattr(
        f"{AscendGRPlatform.__module__}.import_module",
        lambda path: SimpleNamespace(is_available=lambda: True, __name__=path),
    )
    selected = npu.get_attention_execution()
    assert selected.__name__.endswith("npu_xllm" if mode == "xllm" else "attn_npu")


def test_cuda_runtime_patch_is_noop() -> None:
    assert cuda.prepare_gr_engine() == 0


def test_npu_runtime_patches_preserve_order_and_sum_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.platforms.ascend import config_patch, spawn_patch

    events = []

    def spawn() -> bool:
        events.append("spawn")
        return True

    monkeypatch.setattr(spawn_patch, "apply_npu_spawn_patch", spawn)
    monkeypatch.setattr(
        config_patch, "apply_engine_core_config_sanitizer", lambda cls: events.append("sanitize")
    )
    assert npu.prepare_gr_engine() == 1
    assert events == ["spawn", "sanitize"]


def test_cuda_environment_check_validates_platform_and_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.platforms import compat

    platforms = ModuleType("vllm.platforms")
    from tests.helpers.native_platform import native_platform

    monkeypatch.setattr(
        platforms, "current_platform", native_platform("CUDA", "cuda"), raising=False
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    report = {"vllm": "0.30.0"}
    check_versions = mock.Mock(return_value=report)
    monkeypatch.setattr(compat, "check_vllm_runtime", check_versions)

    assert cuda.check_environment() == {
        "vllm": "0.30.0",
        "platform": "cuda",
    }
    check_versions.assert_called_once_with()

    monkeypatch.setattr(
        platforms, "current_platform", native_platform("ROCM", "cuda"), raising=False
    )
    with pytest.raises(RuntimeError, match="requires the native CUDA platform"):
        cuda.check_environment()


pytestmark = pytest.mark.cpu_test
