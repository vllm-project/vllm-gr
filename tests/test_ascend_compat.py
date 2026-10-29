# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest

from vllm_gr.config import GRAttentionBackend
from vllm_gr.v1 import ascend_compat


def _install_selector_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, mock.Mock]:
    import vllm.v1.attention.selector as selector

    native_backend = object()
    native_selector = mock.Mock(return_value=native_backend)
    monkeypatch.setattr(selector, "_cached_get_attn_backend", native_selector)
    monkeypatch.setattr(
        ascend_compat,
        "_ASCEND_ORIGINAL_CACHED_GET_ATTN",
        None,
    )
    monkeypatch.setattr(
        ascend_compat,
        "_ASCEND_ATTN_BACKEND_PATCH_APPLIED",
        False,
    )

    assert ascend_compat.apply_ascend_attention_backend_patch() == 1
    return native_backend, native_selector


def test_import_does_not_load_npu_worker() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = """
import sys
import vllm_gr.v1.ascend_compat
assert "vllm_ascend.worker.worker" not in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_selector_routes_custom_marker_and_native_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.selector as selector
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    native_backend, native_selector = _install_selector_patch(monkeypatch)
    beam_backend = object()
    get_beam_backend = mock.Mock(return_value=beam_backend)
    monkeypatch.setattr(
        ascend_compat,
        "_get_custom_beam_backend",
        get_beam_backend,
    )

    assert (
        selector._cached_get_attn_backend(
            AttentionBackendEnum.CUSTOM,
            "selector-config",
            num_heads=8,
        )
        is beam_backend
    )

    monkeypatch.setattr(
        ascend_compat,
        "_get_current_gr_attention_backend",
        lambda: GRAttentionBackend.CUSTOM,
    )
    assert (
        selector._cached_get_attn_backend(
            None,
            "selector-config",
            num_heads=8,
        )
        is beam_backend
    )

    monkeypatch.setattr(
        ascend_compat,
        "_get_current_gr_attention_backend",
        lambda: None,
    )
    assert (
        selector._cached_get_attn_backend(
            None,
            "selector-config",
            num_heads=8,
        )
        is native_backend
    )
    assert (
        selector._cached_get_attn_backend(
            AttentionBackendEnum.FLASH_ATTN,
            "selector-config",
            num_heads=4,
        )
        is native_backend
    )

    assert get_beam_backend.call_count == 2
    assert native_selector.call_args_list == [
        mock.call(None, "selector-config", num_heads=8),
        mock.call(
            AttentionBackendEnum.FLASH_ATTN,
            "selector-config",
            num_heads=4,
        ),
    ]


def test_selector_installation_is_idempotent_and_force_uses_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.selector as selector
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    _, native_selector = _install_selector_patch(monkeypatch)
    first_wrapper = selector._cached_get_attn_backend

    assert ascend_compat.apply_ascend_attention_backend_patch() == 0
    assert selector._cached_get_attn_backend is first_wrapper
    assert ascend_compat.apply_ascend_attention_backend_patch(force=True) == 1
    assert selector._cached_get_attn_backend is not first_wrapper

    selector._cached_get_attn_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "selector-config",
        num_heads=2,
    )
    native_selector.assert_called_once_with(
        AttentionBackendEnum.FLASH_ATTN,
        "selector-config",
        num_heads=2,
    )


def test_custom_backend_import_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.selector as selector
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setitem(sys.modules, "vllm_gr.v1.attention.backends.beam_attn", None)
    native_backend, native_selector = _install_selector_patch(monkeypatch)
    assert (
        selector._cached_get_attn_backend(AttentionBackendEnum.FLASH_ATTN, "config")
        is native_backend
    )
    native_selector.assert_called_once()
    with pytest.raises(ModuleNotFoundError, match="beam_attn"):
        selector._cached_get_attn_backend(AttentionBackendEnum.CUSTOM, "config")


def _install_fake_npu_worker(
    monkeypatch: pytest.MonkeyPatch,
    init_device: Any,
) -> type[Any]:
    ascend_module = ModuleType("vllm_ascend")
    worker_package = ModuleType("vllm_ascend.worker")
    worker_module = ModuleType("vllm_ascend.worker.worker")
    setattr(ascend_module, "__path__", [])
    setattr(worker_package, "__path__", [])

    class NPUWorker:
        def init_device(self) -> Any:
            return init_device(self)

    setattr(worker_module, "NPUWorker", NPUWorker)
    setattr(ascend_module, "worker", worker_package)
    setattr(worker_package, "worker", worker_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", worker_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", worker_module)
    return NPUWorker


def test_npu_worker_patch_is_lazy_and_preserves_lifecycle_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    monkeypatch.setattr(ascend_compat, "_NPU_WORKER_PATCH_APPLIED", False)
    monkeypatch.setattr(ascend_compat, "is_npu_platform", lambda: False)
    monkeypatch.delitem(sys.modules, "vllm_ascend.worker.worker", raising=False)
    assert ascend_compat.patch_npu_worker_init_device() == 0
    assert "vllm_ascend.worker.worker" not in sys.modules

    monkeypatch.setattr(ascend_compat, "is_npu_platform", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", None)
    assert ascend_compat.patch_npu_worker_init_device() == 0

    events: list[str] = []
    raise_from_original = False

    def original_init_device(self: object) -> str:
        events.append("original")
        if raise_from_original:
            raise RuntimeError("init failed")
        return "ready"

    worker_class = _install_fake_npu_worker(monkeypatch, original_init_device)

    def re_apply_patches() -> int:
        events.append("re_apply")
        return 0

    monkeypatch.setattr(
        runtime_patch,
        "re_apply_patches",
        re_apply_patches,
    )

    assert ascend_compat.patch_npu_worker_init_device() == 1
    assert ascend_compat.patch_npu_worker_init_device() == 0
    assert worker_class().init_device() == "ready"
    assert events == ["re_apply", "original"]

    from vllm_gr.v1.worker.model_runner_common import (
        freeze_gc_after_warmup,
        get_beam_graph_runtime_stats,
    )

    assert worker_class.__dict__["get_gr_beam_graph_runtime_stats"] is get_beam_graph_runtime_stats
    assert worker_class.__dict__["gr_freeze_gc_after_warmup"] is freeze_gc_after_warmup

    events.clear()
    raise_from_original = True
    with pytest.raises(RuntimeError, match="init failed"):
        worker_class().init_device()
    assert events == ["re_apply", "original"]
