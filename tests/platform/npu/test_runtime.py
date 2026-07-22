# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from types import ModuleType
from unittest import mock

import vllm_gr.platform.npu.runtime as npu_runtime


def test_non_npu_runtime_is_noop() -> None:
    with (
        mock.patch.object(npu_runtime, "is_npu_platform", return_value=False),
        mock.patch.object(npu_runtime, "install_npu_attention_backend_compat") as install_attention,
    ):
        assert npu_runtime.install_npu_runtime_compat(force=True) == 0

    install_attention.assert_not_called()


def test_worker_installs_only_attention_compat_before_original_init() -> None:
    events: list[str] = []

    class FakeNPUWorker:
        def init_device(self) -> str:
            events.append("original")
            return "initialized"

    ascend = ModuleType("vllm_ascend")
    worker_package = ModuleType("vllm_ascend.worker")
    worker_module = ModuleType("vllm_ascend.worker.worker")
    setattr(worker_module, "NPUWorker", FakeNPUWorker)
    modules: dict[str, ModuleType] = {
        "vllm_ascend": ascend,
        "vllm_ascend.worker": worker_package,
        "vllm_ascend.worker.worker": worker_module,
    }

    def install_attention(*, force: bool = False) -> int:
        del force
        events.append("attention")
        return 0

    npu_runtime._PATCHED_WORKER_CLASS = None
    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch.object(npu_runtime, "is_npu_platform", return_value=True),
        mock.patch.object(
            npu_runtime,
            "install_npu_attention_backend_compat",
            side_effect=install_attention,
        ),
    ):
        assert npu_runtime._install_npu_worker_attention_hook() == 1
        assert FakeNPUWorker().init_device() == "initialized"

    assert events == ["attention", "original"]


def test_runtime_installs_attention_before_worker_hook() -> None:
    events: list[str] = []

    def install_attention(*, force: bool = False) -> int:
        del force
        events.append("attention")
        return 1

    def install_worker_hook() -> int:
        events.append("worker")
        return 1

    with (
        mock.patch.object(npu_runtime, "is_npu_platform", return_value=True),
        mock.patch.object(
            npu_runtime,
            "install_npu_attention_backend_compat",
            side_effect=install_attention,
        ),
        mock.patch.object(
            npu_runtime,
            "_install_npu_worker_attention_hook",
            side_effect=install_worker_hook,
        ),
    ):
        assert npu_runtime.install_npu_runtime_compat(force=True) == 2

    assert events == ["attention", "worker"]
