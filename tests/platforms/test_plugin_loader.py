# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for external backend entry-point loading."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from vllm_gr.platforms import loader
from vllm_gr.platforms.builtins import register_builtin_backends
from vllm_gr.platforms.errors import (
    DuplicateBackendRegistrationError,
    IncompatibleBackendAPIError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from vllm_gr.platforms.protocol import BACKEND_API_VERSION
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
    activate_backend,
    resolve_backend,
)


@dataclass(frozen=True)
class FakeProvider:
    api_version: int = BACKEND_API_VERSION
    device_type: str = "fake"

    def configure_engine(self, config: Any) -> None:
        del config

    def check_environment(self) -> dict[str, str]:
        return {}

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> Any:
        return object()

    def get_attention_hooks(self) -> Any:
        return object()

    def release_accelerator_memory(self) -> None:
        return None


@dataclass
class FakeEntryPoint:
    name: str
    provider: Any
    load_count: int = 0

    def load(self) -> Any:
        self.load_count += 1
        return self.provider


def setup_function() -> None:
    _reset_backend_registry_for_tests()
    loader._PLUGINS_LOADED = False


def teardown_function() -> None:
    _reset_backend_registry_for_tests()
    loader._PLUGINS_LOADED = False


def _install_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    *fake_entry_points: FakeEntryPoint,
) -> mock.Mock:
    discover = mock.Mock(return_value=fake_entry_points)
    monkeypatch.setattr(loader, "entry_points", discover)
    return discover


def test_external_npu_provider_exposes_complete_runtime_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_builtin_backends()
    with pytest.raises(UnsupportedBackendError):
        resolve_backend("npu")

    provider = FakeProvider(device_type="npu")
    entry_point = FakeEntryPoint("npu", provider)
    discover = _install_entry_points(monkeypatch, entry_point)

    loader.load_backend_plugins()

    assert resolve_backend("npu") is provider
    assert activate_backend("npu") is provider
    provider.configure_engine(SimpleNamespace())
    assert provider.check_environment() == {}
    assert provider.apply_runtime_patches() == 0
    assert provider.get_model_runner_hooks() is not None
    assert provider.get_attention_hooks() is not None
    provider.release_accelerator_memory()
    discover.assert_called_once_with(group=loader.ENTRYPOINT_GROUP)
    assert entry_point.load_count == 1


def test_entry_point_name_must_match_provider_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("other", FakeProvider(device_type="fake")),
    )

    with pytest.raises(InvalidBackendProviderError) as exc_info:
        loader.load_backend_plugins()

    message = str(exc_info.value)
    assert "entrypoint_name='other'" in message
    assert "provider_device_type='fake'" in message


def test_rejects_incompatible_provider_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("fake", FakeProvider(api_version=BACKEND_API_VERSION + 1)),
    )

    with pytest.raises(IncompatibleBackendAPIError) as exc_info:
        loader.load_backend_plugins()

    message = str(exc_info.value)
    assert f"provider_api_version={BACKEND_API_VERSION + 1}" in message
    assert f"core_api_version={BACKEND_API_VERSION}" in message


def test_plugin_loader_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint("fake", FakeProvider())
    discover = _install_entry_points(monkeypatch, entry_point)

    loader.load_backend_plugins()
    loader.load_backend_plugins()

    discover.assert_called_once_with(group=loader.ENTRYPOINT_GROUP)
    assert entry_point.load_count == 1


def test_external_provider_cannot_override_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_builtin_backends()
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("cuda", FakeProvider(device_type="cuda")),
    )

    with pytest.raises(DuplicateBackendRegistrationError) as exc_info:
        loader.load_backend_plugins()

    message = str(exc_info.value)
    assert "existing_source='builtin:cuda'" in message
    assert "attempted_source='entrypoint:cuda'" in message


def test_plugin_batch_failure_does_not_partially_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("fake", FakeProvider()),
        FakeEntryPoint("wrong", FakeProvider(device_type="second")),
    )

    with pytest.raises(InvalidBackendProviderError):
        loader.load_backend_plugins()

    with pytest.raises(UnsupportedBackendError):
        resolve_backend("fake")
    assert loader._PLUGINS_LOADED is False


def test_entry_point_must_return_provider_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("fake", FakeProvider),
    )

    with pytest.raises(InvalidBackendProviderError, match="not a class"):
        loader.load_backend_plugins()


def test_activate_current_backend_rejects_unknown_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_module = ModuleType("vllm")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="cpu")  # type: ignore[attr-defined]
    vllm_module.platforms = platforms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    _install_entry_points(monkeypatch)

    with pytest.raises(UnsupportedBackendError) as exc_info:
        loader.activate_current_backend()

    message = str(exc_info.value)
    assert "requested='cpu'" in message
    assert "available=('cuda',)" in message
