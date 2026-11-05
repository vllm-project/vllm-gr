# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    from vllm_gr.v1 import npu_kv_layout

    monkeypatch.setattr(ascend_compat, "_NPU_WORKER_PATCH_APPLIED", False)
    install_filter = mock.Mock()
    monkeypatch.setattr(ascend_compat, "_install_init_ascend_config_sanitizer", install_filter)
    monkeypatch.setattr(ascend_compat, "is_npu_platform", lambda: False)
    monkeypatch.delitem(sys.modules, "vllm_ascend.worker.worker", raising=False)
    assert ascend_compat.patch_npu_worker_init_device() == 0
    assert "vllm_ascend.worker.worker" not in sys.modules

    monkeypatch.setattr(ascend_compat, "is_npu_platform", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", None)
    assert ascend_compat.patch_npu_worker_init_device() == 0
    install_filter.assert_not_called()

    events: list[str] = []
    monkeypatch.setattr(
        npu_kv_layout,
        "apply_npu_contiguous_kv_cache_patch",
        lambda: events.append("kv_layout"),
    )
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
    install_filter.assert_called_once_with()
    assert worker_class().init_device() == "ready"
    assert events == ["re_apply", "kv_layout", "original"]

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
    assert events == ["re_apply", "kv_layout", "original"]


def _install_fake_ascend_config(
    monkeypatch: pytest.MonkeyPatch, initialize: Any, *, install: bool = True
) -> tuple[Any, Any]:
    package = ModuleType("vllm_ascend")
    package.__path__ = []
    config_module = ModuleType("vllm_ascend.ascend_config")
    setattr(config_module, "init_ascend_config", initialize)
    setattr(package, "ascend_config", config_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_config", config_module)
    entry = ModuleType("vllm_ascend.patch.platform.patch_engine_core")
    setattr(entry, "init_ascend_config", initialize)
    worker = ModuleType("vllm_ascend.worker.worker")
    setattr(worker, "init_ascend_config", initialize)
    monkeypatch.setitem(sys.modules, entry.__name__, entry)
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    if not install:
        return initialize, entry
    ascend_compat._install_init_ascend_config_sanitizer()
    assert entry.init_ascend_config is config_module.init_ascend_config
    assert worker.init_ascend_config is config_module.init_ascend_config
    ascend_compat._install_init_ascend_config_sanitizer()
    assert entry.init_ascend_config is config_module.init_ascend_config
    return config_module.init_ascend_config, entry


def test_worker_installs_config_filter_before_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.config.parser import parse_gr_config

    config = SimpleNamespace(additional_config={"vllm_gr_beam_max_width": 8})
    seen: list[Any] = []

    def initialize(received: Any) -> None:
        assert received.additional_config == {}
        seen.append(received)

    # A fresh worker process has no filter installed by EngineCore.
    _install_fake_ascend_config(monkeypatch, initialize, install=False)
    worker_module = sys.modules["vllm_ascend.worker.worker"]

    class NPUWorker:
        def __init__(self) -> None:
            worker_module.init_ascend_config(config)

        def init_device(self) -> None:
            raise AssertionError("config validation precedes device initialization")

    setattr(worker_module, "NPUWorker", NPUWorker)
    monkeypatch.setattr(ascend_compat, "_NPU_WORKER_PATCH_APPLIED", False)
    monkeypatch.setattr(ascend_compat, "is_npu_platform", lambda: True)
    assert ascend_compat.patch_npu_worker_init_device() == 1
    NPUWorker()
    assert seen == [config]
    assert parse_gr_config(config).beam_max_width == 8


@pytest.mark.parametrize("canonical", [False, True])  # type: ignore[misc]
def test_ascend_config_filter_preserves_engine_identity_and_gr_config(
    monkeypatch: pytest.MonkeyPatch, canonical: bool
) -> None:
    from copy import deepcopy

    from vllm_gr.config.parser import parse_gr_config

    seen: list[Any] = []

    def initialize(config: Any) -> str:
        assert config.additional_config == {"refresh": True}
        seen.append(config)
        return "native-config"

    initialize_filtered, _ = _install_fake_ascend_config(monkeypatch, initialize)
    configs = []
    for width in (4, 8):
        additional: dict[str, Any] = {"refresh": True}
        if canonical:
            additional["vllm_gr"] = {
                "schema_version": 1,
                "beam": {"execution_mode": "v1", "max_width": width},
                "constraint_table": {"max_top_k": width},
            }
        else:
            additional.update(vllm_gr_beam_execution_mode="v1", vllm_gr_beam_max_width=width)
        config = SimpleNamespace(additional_config=additional)
        before = deepcopy(additional)
        expected = parse_gr_config(config)
        assert initialize_filtered(config) == "native-config"
        assert seen[-1] is config
        assert config.additional_config is additional
        assert additional == before
        assert parse_gr_config(config) == expected
        assert parse_gr_config(deepcopy(config)) == expected
        configs.append(config)

    # Initializing another engine cannot change a previous engine or a
    # completely ordinary config in the same process.
    assert [parse_gr_config(config).beam_max_width for config in configs] == [4, 8]
    assert parse_gr_config(SimpleNamespace(additional_config={})).beam_max_width is None


def test_ascend_config_filter_restores_mapping_on_native_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    additional = {"vllm_gr_beam_max_width": 4, "unknown_vendor_key": True}
    config = SimpleNamespace(additional_config=additional)

    def initialize(received: Any) -> None:
        assert received is config
        # Unknown vendor options still reach native validation.
        assert received.additional_config == {"unknown_vendor_key": True}
        raise ValueError("native validation failed")

    initialize_filtered, _ = _install_fake_ascend_config(monkeypatch, initialize)
    with pytest.raises(ValueError, match="native validation failed"):
        initialize_filtered(config)
    assert config.additional_config is additional
    assert additional["vllm_gr_beam_max_width"] == 4


@pytest.mark.parametrize("additional", [None, {}, {"refresh": True}])  # type: ignore[misc]
def test_ascend_config_filter_leaves_ordinary_config_untouched(
    monkeypatch: pytest.MonkeyPatch, additional: Any
) -> None:
    config = SimpleNamespace(additional_config=additional)

    def initialize(received: Any) -> object:
        assert received is config
        assert received.additional_config is additional
        return config

    initialize_filtered, _ = _install_fake_ascend_config(monkeypatch, initialize)
    assert initialize_filtered(config) is config


def test_ascend_entry_installs_filter_before_delegating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(additional_config={"vllm_gr_beam_max_width": 4})

    def initialize(received: Any) -> None:
        assert received is config
        assert received.additional_config == {}

    _, entry = _install_fake_ascend_config(monkeypatch, initialize)

    def native_entry(*args: Any, **kwargs: Any) -> str:
        assert args == ("socket-address",)
        assert kwargs == {"vllm_config": config, "dp_rank": 2, "local_dp_rank": 1}
        entry.init_ascend_config(config)
        assert config.additional_config == {"vllm_gr_beam_max_width": 4}
        return "ready"

    setattr(entry, "_run_engine_core_patch_func", native_entry)
    assert (
        ascend_compat._sanitize_engine_core_entry(
            "socket-address", vllm_config=config, dp_rank=2, local_dp_rank=1
        )
        == "ready"
    )
