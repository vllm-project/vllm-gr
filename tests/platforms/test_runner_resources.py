# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Resource ownership survives reload, failed initialization and separate engines."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.config import GRAttentionBackend, GRConfig
from vllm_gr.v1.worker import runner as lifecycle

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def runner() -> Any:
    config = SimpleNamespace(additional_config={}, model_config=SimpleNamespace(max_model_len=4096))
    GRConfig(beam_max_width=16, attention_backend=GRAttentionBackend.CUSTOM).write_to_vllm_config(
        config
    )
    impl = SimpleNamespace(
        validate_execution=Mock(),
        bind_gr_attention_execution=Mock(),
        release_gr_attention_execution=Mock(),
    )
    return SimpleNamespace(
        vllm_config=config, model=SimpleNamespace(modules=lambda: [SimpleNamespace(impl=impl)])
    ), impl


def select(monkeypatch: pytest.MonkeyPatch, *executions: Any) -> Any:
    import vllm_gr.platforms

    platform = SimpleNamespace(get_attention_execution=Mock(side_effect=executions))
    monkeypatch.setattr(vllm_gr.platforms, "current_gr_platform", platform, raising=False)
    return platform


def execution() -> Any:
    return SimpleNamespace(is_available=lambda: True, close=Mock())


def test_engines_own_separate_attention_and_close_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, layer_a = runner()
    second, layer_b = runner()
    a, b = execution(), execution()
    select(monkeypatch, a, b)
    lifecycle.initialize_gr_attention(first)
    lifecycle.initialize_gr_attention(second)
    layer_a.bind_gr_attention_execution.assert_called_once_with(a)
    layer_b.bind_gr_attention_execution.assert_called_once_with(b)
    lifecycle.close_gr_attention(first)
    lifecycle.close_gr_attention(first)
    a.close.assert_called_once_with()
    b.close.assert_not_called()
    assert first._gr_ready is False and second._gr_ready is True
    layer_a.release_gr_attention_execution.assert_called_once_with(a)


def test_reload_closes_previous_resources_then_binds_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _ = runner()
    a, b = execution(), execution()
    select(monkeypatch, a, b)
    native = Mock(return_value="loaded")
    load = lifecycle.load_gr_model(native)
    assert load(instance) == "loaded"
    assert load(instance) == "loaded"
    a.close.assert_called_once_with()
    b.close.assert_not_called()
    assert instance._gr_attention_execution is b
    assert native.call_count == 2


def test_failed_binding_closes_resources_without_hiding_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, layer = runner()
    chosen = execution()
    select(monkeypatch, chosen)
    expected = ValueError("unsupported attention geometry")
    layer.validate_execution.side_effect = expected
    chosen.close.side_effect = RuntimeError("secondary cleanup failure")
    with pytest.raises(ValueError, match="geometry") as caught:
        lifecycle.load_gr_model(lambda self: None)(instance)
    assert caught.value is expected
    chosen.close.assert_called_once_with()
    layer.bind_gr_attention_execution.assert_not_called()
    assert instance._gr_ready is False
    assert instance._gr_attention_execution is None


def test_failed_native_load_closes_partially_created_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _ = runner()
    backend = SimpleNamespace(close=Mock())

    def native(self: Any) -> Any:
        self.constraint_backend = backend
        raise RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        lifecycle.load_gr_model(native)(instance)
    backend.close.assert_called_once_with()
    assert instance.constraint_backend is None and not instance._gr_ready


def test_native_autofit_allowed_but_gr_config_drift_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _ = runner()
    select(monkeypatch, execution())

    def fit(self: Any) -> Any:
        self.vllm_config.model_config.max_model_len = 2048

    lifecycle.load_gr_model(fit)(instance)
    assert instance._gr_ready

    def drift(self: Any) -> Any:
        GRConfig(beam_max_width=32).write_to_vllm_config(self.vllm_config)

    with pytest.raises(RuntimeError, match="changed while loading"):
        lifecycle.load_gr_model(drift)(instance)
    assert not instance._gr_ready


def test_reload_rejects_pending_sample_before_releasing_current_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _ = runner()
    chosen = execution()
    select(monkeypatch, chosen)
    lifecycle.initialize_gr_attention(instance)
    instance._gr_pending_stage_sample = True
    native = Mock()
    with pytest.raises(RuntimeError, match="previous dispatch"):
        lifecycle.load_gr_model(native)(instance)
    chosen.close.assert_not_called()
    native.assert_not_called()
    assert instance._gr_ready


def test_attention_cleanup_failure_still_releases_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker.model_runner_patch import close_constraint_backend

    instance, _ = runner()
    chosen = execution()
    select(monkeypatch, chosen)
    lifecycle.initialize_gr_attention(instance)
    backend = SimpleNamespace(close=Mock())
    instance.constraint_backend = backend
    chosen.close.side_effect = RuntimeError("vendor close failed")
    with pytest.raises(RuntimeError, match="vendor close failed"):
        close_constraint_backend(instance)
    backend.close.assert_called_once_with()
    assert instance.constraint_backend is None and not instance._gr_ready
