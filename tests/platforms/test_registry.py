# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the process-local backend registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.errors import (
    BackendAlreadyActiveError,
    BackendNotActiveError,
    DuplicateBackendRegistrationError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from vllm_gr.platforms.protocol import BACKEND_API_VERSION
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
    activate_backend,
    get_active_backend,
    register_backend,
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


def setup_function() -> None:
    _reset_backend_registry_for_tests()


def teardown_function() -> None:
    _reset_backend_registry_for_tests()


def test_registers_cuda_and_fake_providers() -> None:
    fake = FakeProvider()

    register_backend(CUDA_BACKEND, source="test:cuda")
    register_backend(fake, source="test:fake")

    assert resolve_backend("cuda") is CUDA_BACKEND
    assert resolve_backend("fake") is fake


def test_registering_same_provider_object_is_idempotent() -> None:
    provider = FakeProvider()

    register_backend(provider, source="first")
    register_backend(provider, source="second")

    assert resolve_backend("fake") is provider


def test_different_provider_for_same_device_type_is_rejected() -> None:
    register_backend(FakeProvider(), source="first-package")

    with pytest.raises(DuplicateBackendRegistrationError) as exc_info:
        register_backend(FakeProvider(), source="second-package")

    message = str(exc_info.value)
    assert "requested='fake'" in message
    assert "existing_source='first-package'" in message
    assert "attempted_source='second-package'" in message
    assert "active=None" in message


def test_unknown_backend_is_rejected_without_cuda_fallback() -> None:
    register_backend(CUDA_BACKEND)

    with pytest.raises(UnsupportedBackendError) as exc_info:
        resolve_backend("gpu")

    message = str(exc_info.value)
    assert "No vllm-gr backend registered for 'gpu'" in message
    assert "available=('cuda',)" in message


def test_get_active_backend_requires_activation() -> None:
    register_backend(FakeProvider())

    with pytest.raises(BackendNotActiveError, match="active=None"):
        get_active_backend()


def test_activation_and_repeated_activation_are_idempotent() -> None:
    provider = FakeProvider()
    register_backend(provider)

    assert activate_backend("fake") is provider
    assert activate_backend("fake") is provider
    assert get_active_backend() is provider


def test_active_backend_cannot_switch_in_same_process() -> None:
    npu = FakeProvider(device_type="npu")
    register_backend(CUDA_BACKEND, source="builtin:cuda")
    register_backend(npu, source="builtin:npu")
    activate_backend("cuda")

    with pytest.raises(BackendAlreadyActiveError) as exc_info:
        activate_backend("npu")

    message = str(exc_info.value)
    assert "requested='npu'" in message
    assert "active='cuda'" in message
    assert "active_source='builtin:cuda'" in message
    assert "requested_source='builtin:npu'" in message


def test_empty_device_types_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_backend("  ")

    with pytest.raises(InvalidBackendProviderError, match="must not be empty"):
        register_backend(FakeProvider(device_type="  "))


def test_device_type_lookup_is_normalized_without_aliases() -> None:
    provider = FakeProvider(device_type=" CuDa ")
    register_backend(provider)

    assert resolve_backend("  CUDA ") is provider
    assert activate_backend("CuDa") is provider
    with pytest.raises(UnsupportedBackendError):
        resolve_backend("gpu")


def test_provider_surface_validation_does_not_call_getters() -> None:
    calls: list[str] = []

    @dataclass(frozen=True)
    class LazyProvider(FakeProvider):
        def get_model_runner_hooks(self) -> Any:
            calls.append("model-runner")
            return object()

        def get_attention_hooks(self) -> Any:
            calls.append("attention")
            return object()

    register_backend(LazyProvider())

    assert calls == []


def test_provider_class_and_missing_methods_are_rejected() -> None:
    with pytest.raises(InvalidBackendProviderError, match="not a class"):
        register_backend(FakeProvider)

    class IncompleteProvider:
        api_version = BACKEND_API_VERSION
        device_type = "incomplete"

    with pytest.raises(InvalidBackendProviderError, match="required callable"):
        register_backend(IncompleteProvider())
