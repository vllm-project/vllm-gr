# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from importlib import import_module
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


def test_runtime_patch_order_preserves_ascend_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    call_order: list[str] = []

    def apply_attention(force: bool = False) -> int:
        call_order.append("attention")
        return 1

    def patch_npu_worker() -> int:
        call_order.append("npu_worker")
        return 1

    def apply_beam(force: bool = False) -> int:
        call_order.append("beam")
        return 1

    monkeypatch.setattr(
        runtime_patch.ascend_compat,
        "apply_ascend_attention_backend_patch",
        apply_attention,
    )
    monkeypatch.setattr(
        runtime_patch.ascend_compat,
        "patch_npu_worker_init_device",
        patch_npu_worker,
    )
    monkeypatch.setattr(
        runtime_patch,
        "_apply_gr_beam_patches",
        apply_beam,
    )

    assert runtime_patch._apply_patches(force=True) == 3
    assert call_order == ["attention", "npu_worker", "beam"]


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

    assert runtime_patch._apply_gr_beam_patches() == 2
    assert runtime_patch._apply_gr_beam_patches() == 0


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
    import vllm_gr.patch as runtime_patch
    from vllm_gr.v1.engine import engine_core_patch

    calls: list[str] = []

    def initialize() -> BeamPatchReport:
        calls.append("initialize")
        return BeamPatchReport(())

    def original_run(*args: object, **kwargs: object) -> None:
        calls.append("run")

    monkeypatch.setattr(runtime_patch, "initialize_engine_core_beam_patches", initialize)
    monkeypatch.setattr(engine_core_patch, "_run_engine_core_active", False)
    monkeypatch.setattr(
        engine_core_patch.run_engine_core,
        "_original_run_engine_core",
        original_run,
        raising=False,
    )

    engine_core_patch.run_engine_core()

    assert calls == ["initialize", "run"]


def test_model_runner_verifier_checks_shared_discovery_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch
    from vllm_gr.v1.worker import model_runner_patch

    class PatchedRunner:
        pass

    class MissingRunner:
        pass

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
        == "No supported active-platform ModelRunner is available"
    )


def test_plugin_exposes_simple_per_patch_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_check = import_module("vllm_gr._env_check")
    plugin = import_module("vllm_gr.plugin")
    runtime_patch = import_module("vllm_gr.patch")

    report = BeamPatchReport((BeamPatchResult("flat_logprobs", BeamPatchStatus.APPLIED),))
    monkeypatch.setattr(env_check, "configure_model_runner_default", mock.Mock())
    monkeypatch.setattr(env_check, "check_environment", mock.Mock())
    reapply = mock.Mock(return_value=1)
    monkeypatch.setattr(runtime_patch, "re_apply_patches", reapply)
    monkeypatch.setattr(
        runtime_patch,
        "get_last_beam_patch_report",
        mock.Mock(return_value=report),
    )

    state = plugin.initialize_runtime()

    reapply.assert_called_once_with()
    assert state == {
        "patch_count": 1,
        "patches": {"flat_logprobs": "applied"},
    }
