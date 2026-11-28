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
from vllm_gr.platforms.ascend import config_patch
from vllm_gr.v1.attention import selector_patch


def test_import_does_not_load_npu_worker() -> None:
    project_root = REPO_ROOT
    code = """
import sys
import vllm_gr.platforms.ascend.config_patch
import vllm_gr.v1.worker.npu.startup_patch

for module_name in ("torch_npu", "vllm_ascend", "vllm_ascend.worker.worker"):
    assert module_name not in sys.modules, module_name
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.l1  # type: ignore[misc]
def test_selector_cache_reset_clears_cached_resolutions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.attention import selector

    @lru_cache
    def lookup(key: str) -> object:
        return object()

    monkeypatch.setattr(selector, "_cached_get_attn_backend", lookup)
    cached = lookup("key")
    selector_patch.clear_attention_backend_cache()
    assert selector._cached_get_attn_backend is lookup
    assert lookup("key") is not cached


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
    config_patch.install_ascend_config_filter()
    assert entry.init_ascend_config is config_module.init_ascend_config
    assert worker.init_ascend_config is config_module.init_ascend_config
    config_patch.install_ascend_config_filter()
    assert entry.init_ascend_config is config_module.init_ascend_config
    return config_module.init_ascend_config, entry


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

    config_patch.install_ascend_config_filter()

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
        additional = {
            "refresh": True,
            "vllm_gr": {"gr": {"beam_width": [width] * 3, "candidate_width": [width] * 3}},
            "vllm_gr_limits": {"max_decode_rows": width},
        }
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

    entry._run_engine_core_patch_func = native_entry

    class EngineCoreProc:
        run_engine_core = staticmethod(native_entry)

    monkeypatch.setattr(config_patch, "is_npu_platform", lambda: False)
    assert config_patch.apply_engine_core_config_sanitizer(EngineCoreProc) == 0
    assert EngineCoreProc.run_engine_core is native_entry

    monkeypatch.setattr(config_patch, "is_npu_platform", lambda: True)
    assert config_patch.apply_engine_core_config_sanitizer(EngineCoreProc) == 1
    assert EngineCoreProc.run_engine_core is config_patch._sanitize_engine_core_entry
    assert config_patch.apply_engine_core_config_sanitizer(EngineCoreProc) == 0

    monkeypatch.setattr(EngineCoreProc, "run_engine_core", staticmethod(native_entry))
    assert config_patch.apply_engine_core_config_sanitizer(EngineCoreProc) == 1
    assert (
        EngineCoreProc.run_engine_core(
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
    from vllm_gr.v1 import token_buffer
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
    # This test isolates child capacity activation. Full worker readiness has
    # independent cross-rank coverage in test_worker_readiness.py.
    monkeypatch.setattr(
        "vllm_gr.v1.engine.worker_readiness.verify_engine_backend_readiness", mock.Mock()
    )
    raw_initialize = Core.__init__
    engine_core_patch.apply_engine_core_request_patches()
    Core.__init__ = raw_initialize  # type: ignore[method-assign]
    assert engine_core_patch.validate_engine_core_request_patch() is not None

    def initialize_ascend(received: Any) -> None:
        assert received is config and received.additional_config == {}

    _, entry = _install_fake_ascend_config(monkeypatch, initialize_ascend, install=False)

    def native_entry(*args: Any, **kwargs: Any) -> CoreProc:
        entry.init_ascend_config(config)
        assert engine_core_patch.validate_engine_core_request_patch() is None
        return CoreProc(kwargs["vllm_config"])

    entry._run_engine_core_patch_func = native_entry
    initialize = None
    for _ in range(2):
        core = config_patch._sanitize_engine_core_entry(vllm_config=config)
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
    entry._run_engine_core_patch_func = native_entry
    monkeypatch.setattr(
        runtime_patch,
        "initialize_engine_core_beam_patches",
        mock.Mock(side_effect=RuntimeError("required child patch failed")),
    )
    with pytest.raises(RuntimeError, match="required child patch failed"):
        config_patch._sanitize_engine_core_entry(vllm_config=SimpleNamespace(additional_config={}))
    native_entry.assert_not_called()


pytestmark = pytest.mark.cpu_test
