# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from tests.helpers.paths import REPO_ROOT
from vllm_gr.config import GRAttentionBackend
from vllm_gr.v1 import ascend_compat
from vllm_gr.v1.attention import selector_patch


def _install_selector_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, mock.Mock]:
    import vllm.v1.attention.selector as selector

    native_backend = object()
    native_selector = mock.Mock(return_value=native_backend)
    monkeypatch.setattr(selector, "_cached_get_attn_backend", native_selector)
    assert selector_patch.apply_custom_attention_backend_patch() == 1
    return native_backend, native_selector


def test_import_does_not_load_npu_worker() -> None:
    project_root = REPO_ROOT
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
        selector_patch,
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
        selector_patch,
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
        selector_patch,
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
    assert ascend_compat.apply_ascend_attention_backend_patch(force=True) == 0
    assert selector._cached_get_attn_backend is first_wrapper

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


@pytest.mark.l1  # type: ignore[misc]
def test_selector_reinstalls_over_replaced_upstream_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.selector as selector
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    _, first_native = _install_selector_patch(monkeypatch)
    replacement = mock.Mock(return_value="new-native")
    monkeypatch.setattr(selector, "_cached_get_attn_backend", replacement)
    assert selector_patch.apply_custom_attention_backend_patch(force=True) == 1
    assert selector_patch.apply_custom_attention_backend_patch() == 0
    assert (
        selector._cached_get_attn_backend(AttentionBackendEnum.FLASH_ATTN, "config") == "new-native"
    )
    replacement.assert_called_once_with(AttentionBackendEnum.FLASH_ATTN, "config", num_heads=None)
    first_native.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("install_wrapper", [False, True])  # type: ignore[misc]
def test_selector_cache_reset_preserves_installed_callable(
    monkeypatch: pytest.MonkeyPatch, install_wrapper: bool
) -> None:
    import vllm.v1.attention.selector as selector
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    calls: list[object] = []

    @lru_cache
    def native_selector(backend: Any, config: Any, num_heads: int | None = None) -> object:
        result = object()
        calls.append(result)
        return result

    monkeypatch.setattr(selector, "_cached_get_attn_backend", native_selector)
    if install_wrapper:
        assert selector_patch.apply_custom_attention_backend_patch() == 1
    installed = selector._cached_get_attn_backend
    args = (AttentionBackendEnum.FLASH_ATTN, "config")
    first = installed(*args)
    assert installed(*args) is first
    assert len(calls) == 1

    selector_patch.clear_attention_backend_cache()
    assert selector._cached_get_attn_backend is installed
    assert installed(*args) is not first
    assert len(calls) == 2


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
    assert install_filter.call_args_list == [mock.call(), mock.call()]
    assert worker_class().init_device() == "ready"
    assert events == ["re_apply", "kv_layout", "original"]

    from vllm_gr.v1.worker.gr_context import (
        get_beam_graph_runtime_stats,
    )
    from vllm_gr.v1.worker.gr_gc import (
        freeze_gc_after_warmup,
    )

    assert worker_class.__dict__["get_gr_beam_graph_runtime_stats"] is get_beam_graph_runtime_stats
    assert worker_class.__dict__["gr_freeze_gc_after_warmup"] is freeze_gc_after_warmup
    from vllm_gr.v1.worker.hot_update import reload_trie

    assert worker_class.__dict__["reload_trie"] is reload_trie

    events.clear()
    raise_from_original = True
    with pytest.raises(RuntimeError, match="init failed"):
        worker_class().init_device()
    assert events == ["re_apply", "kv_layout", "original"]
    assert worker_class.__dict__["reload_trie"] is reload_trie


@pytest.mark.l1  # type: ignore[misc]
def test_npu_worker_reinstalls_over_replaced_upstream_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ascend_compat, "is_npu_platform", lambda: True)
    monkeypatch.setattr(ascend_compat, "_install_init_ascend_config_sanitizer", lambda: None)
    worker = _install_fake_npu_worker(monkeypatch, lambda _: "native")
    original = worker.init_device
    assert ascend_compat.patch_npu_worker_init_device() == 1
    wrapper = worker.init_device
    assert getattr(wrapper, ascend_compat._NPU_WORKER_ORIGINAL_ATTR) is original
    assert ascend_compat.patch_npu_worker_init_device() == 0
    assert worker.init_device is wrapper

    def replacement(self: object) -> str:
        return "replacement"

    monkeypatch.setattr(worker, "init_device", replacement)
    assert ascend_compat.patch_npu_worker_init_device() == 1
    repaired = worker.init_device
    assert getattr(repaired, ascend_compat._NPU_WORKER_ORIGINAL_ATTR) is replacement
    assert ascend_compat.patch_npu_worker_init_device() == 0
    assert worker.init_device is repaired


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

    config = SimpleNamespace(
        additional_config={"vllm_gr": {"gr": {"beam_width": [8], "candidate_width": [8]}}}
    )
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
    monkeypatch.setattr(ascend_compat, "is_npu_platform", lambda: True)
    assert ascend_compat.patch_npu_worker_init_device() == 1
    NPUWorker()
    assert seen == [config]
    assert parse_gr_config(config).beam_max_width == 8


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("caller", ["worker", "engine_core"])  # type: ignore[misc]
def test_worker_rechecks_config_bindings_without_rewrapping_device_hook(
    monkeypatch: pytest.MonkeyPatch, caller: str
) -> None:
    additional = {"vllm_gr": {"gr": {"beam_width": [8], "candidate_width": [8]}}, "refresh": True}
    config = SimpleNamespace(additional_config=additional)
    seen: list[Any] = []

    def initialize(received: Any) -> None:
        assert received.additional_config == {"refresh": True}
        seen.append(received)

    _, entry = _install_fake_ascend_config(monkeypatch, initialize, install=False)
    worker_module = sys.modules["vllm_ascend.worker.worker"]
    config_module = sys.modules["vllm_ascend.ascend_config"]

    class NPUWorker:
        def __init__(self) -> None:
            worker_module.init_ascend_config(config)

        def init_device(self) -> None:
            raise AssertionError("constructor validation must precede device initialization")

    setattr(worker_module, "NPUWorker", NPUWorker)
    monkeypatch.setattr(ascend_compat, "is_npu_platform", lambda: True)
    assert ascend_compat.patch_npu_worker_init_device() == 1
    device_wrapper = NPUWorker.init_device
    sanitizer = config_module.init_ascend_config

    # A native import restores only this alias; both GR wrappers stay installed.
    rebound_module = worker_module if caller == "worker" else entry
    monkeypatch.setattr(rebound_module, "init_ascend_config", initialize)
    assert ascend_compat.patch_npu_worker_init_device() == 0
    assert rebound_module.init_ascend_config is sanitizer
    assert config_module.init_ascend_config is sanitizer
    assert sanitizer.__wrapped__ is initialize
    assert NPUWorker.init_device is device_wrapper
    NPUWorker()
    entry.init_ascend_config(config)
    assert seen == [config, config]
    assert config.additional_config is additional
    assert additional == {
        "vllm_gr": {"gr": {"beam_width": [8], "candidate_width": [8]}},
        "refresh": True,
    }

    assert ascend_compat.patch_npu_worker_init_device() == 0
    assert NPUWorker.init_device is device_wrapper
    assert config_module.init_ascend_config is sanitizer


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("caller", ["worker", "engine_core"])  # type: ignore[misc]
def test_config_binding_recheck_preserves_unrelated_overrides(
    monkeypatch: pytest.MonkeyPatch, caller: str
) -> None:
    sanitizer, entry = _install_fake_ascend_config(monkeypatch, lambda config: None)
    worker_module = sys.modules["vllm_ascend.worker.worker"]
    overridden_module = worker_module if caller == "worker" else entry
    override = mock.Mock()
    monkeypatch.setattr(overridden_module, "init_ascend_config", override)

    ascend_compat._install_init_ascend_config_sanitizer()

    assert overridden_module.init_ascend_config is override
    assert sys.modules["vllm_ascend.ascend_config"].init_ascend_config is sanitizer


def test_ascend_config_filter_preserves_engine_identity_and_gr_config(
    monkeypatch: pytest.MonkeyPatch,
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
        additional["vllm_gr"] = {"gr": {"beam_width": [width] * 3, "candidate_width": [width] * 3}}
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
    additional = {
        "vllm_gr": {"gr": {"beam_width": [4], "candidate_width": [4]}},
        "unknown_vendor_key": True,
    }
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
    assert additional["vllm_gr"] == {"gr": {"beam_width": [4], "candidate_width": [4]}}


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
    import vllm_gr.patch as runtime_patch

    config = SimpleNamespace(
        additional_config={"vllm_gr": {"gr": {"beam_width": [4], "candidate_width": [4]}}}
    )
    initialize_child = mock.Mock()
    monkeypatch.setattr(runtime_patch, "initialize_engine_core_beam_patches", initialize_child)

    def initialize(received: Any) -> None:
        assert received is config
        assert received.additional_config == {}

    _, entry = _install_fake_ascend_config(monkeypatch, initialize)

    def native_entry(*args: Any, **kwargs: Any) -> str:
        initialize_child.assert_called_once_with()
        assert args == ("socket-address",)
        assert kwargs == {"vllm_config": config, "dp_rank": 2, "local_dp_rank": 1}
        entry.init_ascend_config(config)
        assert config.additional_config == {
            "vllm_gr": {"gr": {"beam_width": [4], "candidate_width": [4]}}
        }
        return "ready"

    setattr(entry, "_run_engine_core_patch_func", native_entry)
    assert (
        ascend_compat._sanitize_engine_core_entry(
            "socket-address", vllm_config=config, dp_rank=2, local_dp_rank=1
        )
        == "ready"
    )


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("trie_count", [1, 3])  # type: ignore[misc]
@pytest.mark.parametrize("beam", [False, True])  # type: ignore[misc]
def test_ascend_child_activates_live_trie_capacity_before_first_core_initialization(
    monkeypatch: pytest.MonkeyPatch, trie_count: int, beam: bool
) -> None:
    """Exercise the real child plan/installer, including an incomplete prior patch."""
    from queue import Queue

    import msgspec
    import vllm.v1.engine.core as native_core

    import vllm_gr.patch as runtime_patch
    from tests.helpers.beam_requests import make_multi_batch_request
    from vllm_gr.v1.beam import token_buffer
    from vllm_gr.v1.beam.batch_contracts import GRAdmissionError
    from vllm_gr.v1.engine import engine_core_patch
    from vllm_gr.v1.engine.core import loaded_constraint_trie_count

    config = SimpleNamespace(
        device_config=SimpleNamespace(device="npu"),
        additional_config={
            "vllm_gr": {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
        }
        if beam
        else {},
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            constraint_backend=SimpleNamespace(
                info=SimpleNamespace(table_metadata=SimpleNamespace(trie_count=trie_count))
            )
        )
    )

    class Core:
        def __init__(self, vllm_config: Any) -> None:
            self.vllm_config = vllm_config
            self.scheduler = SimpleNamespace()
            self.model_executor = SimpleNamespace(
                collective_rpc=mock.Mock(side_effect=lambda function: [function(worker)])
            )
            self.input_queue: Queue[Any] = Queue()

    class CoreProc(Core):
        def _handle_client_request(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("Native dispatch is outside this startup test")

        def process_input_sockets(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("No socket thread is started by this test")

        def _process_engine_step(self) -> None:
            raise AssertionError("No Worker/model is started by this test")

    class DPCoreProc(CoreProc):
        pass

    monkeypatch.setattr(native_core, "EngineCore", Core)
    monkeypatch.setattr(native_core, "EngineCoreProc", CoreProc)
    monkeypatch.setattr(native_core, "DPEngineCoreProc", DPCoreProc)
    monkeypatch.setattr(token_buffer, "install_numpy_request_factory", mock.Mock())
    monkeypatch.setattr(token_buffer, "install_numpy_scheduler_data", mock.Mock())
    monkeypatch.setattr(engine_core_patch, "UPSTREAM_PROCESS_INPUT_SOCKETS", None)
    monkeypatch.setattr(runtime_patch, "_LAST_BEAM_PATCH_REPORT", None)
    # Keep the real requests applicator and verifier. Other child capabilities
    # do not own startup capacity and would require a device/model fixture.
    requests_plan = runtime_patch._engine_core_beam_patches()[:1]
    monkeypatch.setattr(runtime_patch, "_engine_core_beam_patches", lambda: requests_plan)
    raw_initialize = Core.__init__
    engine_core_patch.apply_engine_core_request_patches()
    Core.__init__ = raw_initialize  # type: ignore[method-assign]
    assert runtime_patch._validate_engine_core_request_patch() is not None

    def initialize_ascend(received: Any) -> None:
        assert received is config and received.additional_config == {}

    _, entry = _install_fake_ascend_config(monkeypatch, initialize_ascend, install=False)

    def native_entry(*args: Any, **kwargs: Any) -> CoreProc:
        entry.init_ascend_config(config)
        assert runtime_patch._validate_engine_core_request_patch() is None
        return CoreProc(kwargs["vllm_config"])

    setattr(entry, "_run_engine_core_patch_func", native_entry)
    initialize = None
    for _ in range(2):
        core = ascend_compat._sanitize_engine_core_entry(vllm_config=config)
        if initialize is None:
            initialize = Core.__init__
        assert Core.__init__ is initialize, "Repeated child activation stacked init wrappers"
        if not beam:
            assert not hasattr(core, "_gr_trie_count")
            core.model_executor.collective_rpc.assert_not_called()
            continue
        assert core._gr_trie_count == core.scheduler._gr_trie_count == trie_count
        core.model_executor.collective_rpc.assert_called_once_with(loaded_constraint_trie_count)
        request = make_multi_batch_request()
        bad = msgspec.structs.replace(
            request,
            items=(request.items[0], msgspec.structs.replace(request.items[1], trie_id=trie_count)),
        )
        with pytest.raises(GRAdmissionError, match="trie_id"):
            core._handle_submit_gr_session(bad)
        assert core.input_queue.empty()
        good = msgspec.structs.replace(
            request,
            items=(
                msgspec.structs.replace(request.items[0], trie_id=trie_count - 1),
                request.items[1],
            ),
        )
        core._handle_submit_gr_session(good)
        assert core.input_queue.get_nowait()[1] is good and core.input_queue.empty()


@pytest.mark.l1  # type: ignore[misc]
def test_ascend_child_does_not_start_native_core_when_required_patching_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    _, entry = _install_fake_ascend_config(monkeypatch, mock.Mock(), install=False)
    native_entry = mock.Mock()
    setattr(entry, "_run_engine_core_patch_func", native_entry)
    monkeypatch.setattr(
        runtime_patch,
        "initialize_engine_core_beam_patches",
        mock.Mock(side_effect=RuntimeError("required child patch failed")),
    )
    with pytest.raises(RuntimeError, match="required child patch failed"):
        ascend_compat._sanitize_engine_core_entry(vllm_config=SimpleNamespace(additional_config={}))
    native_entry.assert_not_called()


pytestmark = pytest.mark.cpu_test
