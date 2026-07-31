# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from enum import Enum
from importlib import import_module
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.patch import (
    BeamPatch,
    BeamPatchInitializationError,
    BeamPatchReport,
    BeamPatchResult,
    BeamPatchStatus,
)


def test_patch_plans_preserve_process_responsibilities() -> None:
    from vllm_gr.patch import _engine_core_beam_patches, _required_beam_patches

    frontend_patches = [patch.name for patch in _required_beam_patches()]
    engine_core_patches = [patch.name for patch in _engine_core_beam_patches()]

    assert frontend_patches == [
        "flat_logprobs",
        "openai_beam_search",
        "chat_sampling_conversion",
        "engine_args",
        "serving_models",
        "batch_and_fork",
    ]
    assert engine_core_patches == [
        "engine_core_requests",
        "scheduler",
        "model_runner",
    ]
    assert set(frontend_patches).isdisjoint(engine_core_patches)


def test_runtime_runs_backend_patches_before_common_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch
    from vllm_gr import platforms

    call_order: list[str] = []

    def apply_backend(*, force: bool = False) -> int:
        assert force is True
        call_order.append("backend")
        return 2

    def apply_common(force: bool = False) -> int:
        assert force is True
        call_order.append("common")
        return 3

    monkeypatch.setattr(
        platforms,
        "get_active_backend",
        lambda: SimpleNamespace(apply_runtime_patches=apply_backend),
    )
    monkeypatch.setattr(
        runtime_patch,
        "_apply_common_runtime_patches",
        apply_common,
    )

    assert runtime_patch._apply_patches(force=True) == 5
    assert call_order == ["backend", "common"]


def test_backend_patch_failure_stops_common_patch_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch
    from vllm_gr import platforms

    def fail_backend(*, force: bool = False) -> int:
        del force
        raise RuntimeError("backend failed")

    common = mock.Mock()
    monkeypatch.setattr(
        platforms,
        "get_active_backend",
        lambda: SimpleNamespace(apply_runtime_patches=fail_backend),
    )
    monkeypatch.setattr(runtime_patch, "_apply_common_runtime_patches", common)

    with pytest.raises(RuntimeError, match="backend failed"):
        runtime_patch._apply_patches(force=True)

    common.assert_not_called()


def test_optional_patch_count_tracks_actual_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    monkeypatch.setattr(
        runtime_patch,
        "initialize_required_beam_patches",
        lambda: BeamPatchReport(()),
    )
    monkeypatch.setattr(
        runtime_patch.sliced_logits_patch,
        "apply_sliced_logits_patch",
        mock.Mock(side_effect=(True, False)),
    )
    monkeypatch.setattr(
        runtime_patch.lmcache_patch,
        "is_lmcache_adapter_requested",
        lambda: True,
    )
    monkeypatch.setattr(
        runtime_patch.lmcache_patch,
        "apply_lmcache_adapter",
        mock.Mock(side_effect=(True, False)),
    )

    assert runtime_patch._apply_common_runtime_patches() == 2
    assert runtime_patch._apply_common_runtime_patches() == 0


def test_initialization_applies_required_patch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    state = {"installed": False}
    apply = mock.Mock(side_effect=lambda: state.update(installed=True))
    patch = BeamPatch(
        "beam",
        apply,
        lambda: None if state["installed"] else "not installed",
    )
    monkeypatch.setattr(runtime_patch, "_required_beam_patches", lambda: (patch,))

    first = runtime_patch.initialize_required_beam_patches()
    second = runtime_patch.initialize_required_beam_patches()

    apply.assert_called_once_with()
    assert state["installed"] is True
    assert first.applied_count == 1
    assert first.results[0].status is BeamPatchStatus.APPLIED
    assert second.results[0].status is BeamPatchStatus.ALREADY_APPLIED


def test_required_failures_are_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    later_apply = mock.Mock()

    def fail() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        runtime_patch,
        "_required_beam_patches",
        lambda: (
            BeamPatch("broken", fail, lambda: "not installed"),
            BeamPatch("later", later_apply, lambda: "not installed"),
        ),
    )

    with pytest.raises(BeamPatchInitializationError, match="broken.*boom") as exc_info:
        runtime_patch.initialize_required_beam_patches()

    later_apply.assert_not_called()
    assert exc_info.value.report.results[-1].status is BeamPatchStatus.FAILED
    assert runtime_patch.get_last_beam_patch_report() is exc_info.value.report

    monkeypatch.setattr(
        runtime_patch,
        "_required_beam_patches",
        lambda: (BeamPatch("broken", lambda: None, lambda: "still missing"),),
    )

    with pytest.raises(
        BeamPatchInitializationError,
        match="broken.*still missing",
    ):
        runtime_patch.initialize_required_beam_patches()


def test_engine_core_child_uses_minimal_authoritative_initialization_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngineCoreRequestType(bytes, Enum):
        ADD = b"\x00"

    class FakeEngineCoreProc:
        @staticmethod
        def run_engine_core(*args: object, **kwargs: object) -> None:
            del args, kwargs

    engine_module = ModuleType("vllm.v1.engine")
    engine_module.EngineCoreRequestType = FakeEngineCoreRequestType  # type: ignore[attr-defined]
    engine_core_module = ModuleType("vllm.v1.engine.core")
    engine_core_module.EngineCoreProc = FakeEngineCoreProc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", engine_core_module)

    import vllm_gr.patch as runtime_patch
    import vllm_gr.plugin as runtime_plugin
    from vllm_gr.v1.engine import engine_core_patch

    calls: list[str] = []

    def initialize() -> BeamPatchReport:
        calls.append("initialize")
        return BeamPatchReport(())

    def initialize_backend() -> SimpleNamespace:
        calls.append("backend")
        return SimpleNamespace(device_type="cuda")

    def original_run(*args: object, **kwargs: object) -> None:
        calls.append("run")

    monkeypatch.setattr(
        runtime_plugin,
        "initialize_process_backend",
        initialize_backend,
    )
    monkeypatch.setattr(runtime_patch, "initialize_engine_core_beam_patches", initialize)
    monkeypatch.setattr(engine_core_patch, "_run_engine_core_active", False)
    monkeypatch.setattr(
        engine_core_patch.run_engine_core,
        "_original_run_engine_core",
        original_run,
        raising=False,
    )

    engine_core_patch.run_engine_core()

    assert calls == ["backend", "initialize", "run"]


def test_engine_core_plan_runs_backend_before_common_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch
    from vllm_gr import platforms

    events: list[str] = []

    def apply_backend(*, force: bool = False) -> int:
        assert force is True
        events.append("backend")
        return 1

    def initialize_common(
        patches: tuple[BeamPatch, ...],
    ) -> BeamPatchReport:
        assert [patch.name for patch in patches] == [
            "engine_core_requests",
            "scheduler",
            "model_runner",
        ]
        events.append("common")
        return BeamPatchReport(())

    monkeypatch.setattr(
        platforms,
        "get_active_backend",
        lambda: SimpleNamespace(apply_runtime_patches=apply_backend),
    )
    monkeypatch.setattr(
        runtime_patch,
        "_initialize_beam_patch_plan",
        initialize_common,
    )

    assert runtime_patch.initialize_engine_core_beam_patches() == BeamPatchReport(())
    assert events == ["backend", "common"]


def test_model_runner_verifier_checks_shared_discovery_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch
    from vllm_gr import platforms
    from vllm_gr.v1.worker import model_runner_patch

    class PatchedRunner:
        pass

    class MissingRunner:
        pass

    monkeypatch.setattr(
        platforms,
        "get_active_backend",
        lambda: SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(
        model_runner_patch,
        "discover_active_worker_runners",
        lambda: (
            ("PatchedRunner", PatchedRunner),
            ("MissingRunner", MissingRunner),
        ),
    )
    monkeypatch.setattr(
        model_runner_patch,
        "is_worker_runner_patched",
        lambda runner: runner is PatchedRunner,
    )

    assert (
        runtime_patch._validate_model_runner_patch()
        == "MissingRunner beam execution hooks are not installed"
    )

    monkeypatch.setattr(model_runner_patch, "discover_active_worker_runners", lambda: ())
    assert (
        runtime_patch._validate_model_runner_patch()
        == "No supported ModelRunner is available for backend 'cuda'"
    )


def test_plugin_exposes_simple_per_patch_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_check = import_module("vllm_gr._env_check")
    platforms = import_module("vllm_gr.platforms")
    plugin = import_module("vllm_gr.plugin")
    runtime_patch = import_module("vllm_gr.patch")

    events: list[str] = []
    report = BeamPatchReport((BeamPatchResult("flat_logprobs", BeamPatchStatus.APPLIED),))

    backend = SimpleNamespace(
        device_type="cuda",
        check_environment=lambda: events.append("check_environment"),
    )

    def activate_backend() -> SimpleNamespace:
        events.append("activate")
        return backend

    def apply_patches() -> int:
        events.append("patch")
        return 1

    configure = mock.Mock(side_effect=lambda: events.append("configure"))
    activate = mock.Mock(side_effect=activate_backend)
    global_check = mock.Mock()
    reapply = mock.Mock(side_effect=apply_patches)

    monkeypatch.setattr(env_check, "configure_model_runner_default", configure)
    monkeypatch.setattr(env_check, "check_environment", global_check)
    monkeypatch.setattr(platforms, "activate_current_backend", activate)
    monkeypatch.setattr(runtime_patch, "re_apply_patches", reapply)
    monkeypatch.setattr(
        runtime_patch,
        "get_last_beam_patch_report",
        mock.Mock(return_value=report),
    )

    state = plugin.initialize_runtime()

    reapply.assert_called_once_with()
    global_check.assert_not_called()
    assert events == ["configure", "activate", "check_environment", "patch"]
    assert state == {
        "backend": "cuda",
        "patch_count": 1,
        "patches": {"flat_logprobs": "applied"},
    }
