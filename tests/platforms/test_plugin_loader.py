# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
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
    IncompatibleBackendError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from vllm_gr.platforms.identity import PlatformIdentity
from vllm_gr.platforms.protocol import PROVIDER_API_VERSION
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
    resolve_backend,
)


@dataclass(frozen=True)
class FakeProvider:
    backend_id: str = "fake"
    device_type: str = "fake"
    api_version: int = PROVIDER_API_VERSION

    def matches(self, identity: PlatformIdentity) -> bool:
        return identity.device_type == self.device_type

    def check_environment(self) -> dict[str, str]:
        return {}

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> Any:
        return object()

    def get_attention_execution(self) -> Any:
        return object()

    def create_stage_execution_service(self) -> Any:
        return object()

    def select_runner_patch_module_name(self, *, prefer_v2: bool = False) -> str:
        del prefer_v2
        return "tests.platforms.fakes.patch"

    def create_constrained_topk_execution(self, *, device: Any = None) -> None:
        del device
        return None


@dataclass(frozen=True)
class FakeLegacyProvider:
    """An API-v1 provider: no ``api_version``, no ``backend_id``, no ``matches``."""

    device_type: str = "fake"

    def check_environment(self) -> dict[str, str]:
        return {}

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> Any:
        return object()

    def get_attention_execution(self) -> Any:
        return object()

    def create_constrained_topk_execution(self, *, device: Any = None) -> None:
        del device
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


def _fake_current_platform(
    monkeypatch: pytest.MonkeyPatch,
    platform: Any,
) -> None:
    vllm_module = ModuleType("vllm")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = platform  # type: ignore[attr-defined]
    vllm_module.platforms = platforms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)


def _nvidia_platform() -> SimpleNamespace:
    return SimpleNamespace(_enum="CUDA", device_type="cuda", device_name="cuda")


def test_loads_external_provider_from_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider()
    entry_point = FakeEntryPoint("fake", provider)
    discover = _install_entry_points(monkeypatch, entry_point)

    loader.load_backend_plugins()

    assert resolve_backend("fake") is provider
    discover.assert_called_once_with(group=loader.ENTRYPOINT_GROUP)
    assert entry_point.load_count == 1


def test_entry_point_name_must_match_provider_backend_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("other", FakeProvider(backend_id="fake")),
    )

    with pytest.raises(InvalidBackendProviderError) as exc_info:
        loader.load_backend_plugins()

    message = str(exc_info.value)
    assert "entrypoint_name='other'" in message
    assert "provider_backend_id='fake'" in message


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
        FakeEntryPoint("cuda", FakeProvider(backend_id="cuda", device_type="cuda")),
    )

    with pytest.raises(DuplicateBackendRegistrationError) as exc_info:
        loader.load_backend_plugins()

    message = str(exc_info.value)
    assert "existing_origin='builtin:cuda [backend_id=cuda]'" in message
    assert "attempted_origin='entrypoint:cuda [backend_id=cuda]'" in message


def test_disable_switch_lets_an_external_provider_replace_a_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(loader.DISABLE_BUILTIN_ENV, "ascend")
    register_builtin_backends()
    with pytest.raises(UnsupportedBackendError):
        resolve_backend("ascend")

    replacement = FakeProvider(backend_id="ascend", device_type="npu")
    _install_entry_points(monkeypatch, FakeEntryPoint("ascend", replacement))
    loader.load_backend_plugins()

    assert resolve_backend("ascend") is replacement


def test_duplicate_ids_are_still_rejected_after_a_disable_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switch removes one built-in; it never suppresses duplicate checks."""
    monkeypatch.setenv(loader.DISABLE_BUILTIN_ENV, "ascend")
    register_builtin_backends()
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("ascend", FakeProvider(backend_id="ascend", device_type="npu")),
        FakeEntryPoint("ascend_dup", FakeProvider(backend_id="ascend", device_type="npu")),
    )

    with pytest.raises(InvalidBackendProviderError) as exc_info:
        loader.load_backend_plugins()

    message = str(exc_info.value)
    assert "entrypoint_name='ascend_dup'" in message


def test_plugin_batch_failure_does_not_partially_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("fake", FakeProvider()),
        FakeEntryPoint("wrong", FakeProvider(backend_id="second")),
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
    _fake_current_platform(
        monkeypatch,
        SimpleNamespace(_enum="CPU", device_type="cpu", device_name="cpu"),
    )
    _install_entry_points(monkeypatch)

    with pytest.raises(UnsupportedBackendError) as exc_info:
        loader.activate_current_backend()

    message = str(exc_info.value)
    assert "device_type='cpu'" in message
    assert "available=('ascend', 'cuda')" in message


def test_activate_current_backend_requires_an_ascend_identity_for_ascend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device placement alone does not select the Ascend provider."""
    _fake_current_platform(
        monkeypatch,
        SimpleNamespace(_enum="OOT", device_type="npu", device_name="npu"),
    )
    _install_entry_points(monkeypatch)

    with pytest.raises(UnsupportedBackendError):
        loader.activate_current_backend()


def test_activate_current_backend_selects_the_vendor_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vendor_module = ModuleType("tests_vendor_platform")

    class NPUPlatform:
        _enum = "OOT"
        device_type = "npu"
        device_name = "npu"

    NPUPlatform.__module__ = "tests_vendor_platform"
    NPUPlatform.__qualname__ = "NPUPlatform"
    vendor_module.NPUPlatform = NPUPlatform  # type: ignore[attr-defined]

    class DerivedVendorPlatform(NPUPlatform):
        pass

    DerivedVendorPlatform.__module__ = "vllm_ascend.platform"
    DerivedVendorPlatform.__qualname__ = "NPUPlatform"
    _fake_current_platform(monkeypatch, DerivedVendorPlatform())
    _install_entry_points(monkeypatch)

    assert loader.activate_current_backend().backend_id == "ascend"


def test_explicit_request_cannot_switch_the_native_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(loader.REQUESTED_BACKEND_ENV, "ascend")
    _fake_current_platform(monkeypatch, _nvidia_platform())
    _install_entry_points(monkeypatch)

    with pytest.raises(IncompatibleBackendError) as exc_info:
        loader.activate_current_backend()

    message = str(exc_info.value)
    assert "requested_origin='builtin:ascend [backend_id=ascend]'" in message


def test_explicit_request_selects_the_matching_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(loader.REQUESTED_BACKEND_ENV, "cuda")
    _fake_current_platform(monkeypatch, _nvidia_platform())
    _install_entry_points(monkeypatch)

    assert loader.activate_current_backend().backend_id == "cuda"


def test_empty_explicit_request_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(loader.REQUESTED_BACKEND_ENV, "  ")
    _fake_current_platform(monkeypatch, _nvidia_platform())
    _install_entry_points(monkeypatch)

    with pytest.raises(InvalidBackendProviderError, match=loader.REQUESTED_BACKEND_ENV):
        loader.activate_current_backend()


def test_legacy_entry_point_is_adapted_with_a_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeLegacyProvider(device_type="cuda")
    _install_entry_points(monkeypatch, FakeEntryPoint("cuda", provider))
    _fake_current_platform(monkeypatch, _nvidia_platform())
    monkeypatch.setenv(loader.DISABLE_BUILTIN_ENV, "cuda")

    loader.load_backend_plugins()

    registered = resolve_backend("cuda")
    assert registered.backend_id == "cuda"
    assert registered.matches(_nvidia_identity()) is True
    assert registered.get_model_runner_hooks() is not None


def test_legacy_entry_point_for_an_unknown_device_type_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("fake", FakeLegacyProvider(device_type="fake")),
    )

    with pytest.raises(InvalidBackendProviderError) as exc_info:
        loader.load_backend_plugins()

    message = str(exc_info.value)
    assert "API version 1" in message
    assert "device_type='fake'" in message


def _nvidia_identity() -> PlatformIdentity:
    return PlatformIdentity(
        platform_enum="CUDA",
        device_type="cuda",
        device_name="cuda",
        platform_class="vllm.platforms.cuda.NvmlCudaPlatform",
        platform_lineage=("vllm.platforms.cuda.NvmlCudaPlatform",),
    )


pytestmark = pytest.mark.cpu_test
