# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Unit tests for the process-local backend registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.errors import (
    AmbiguousBackendError,
    BackendAlreadyActiveError,
    BackendNotActiveError,
    DuplicateBackendRegistrationError,
    IncompatibleBackendError,
    InvalidBackendProviderError,
    InvalidBackendVersionError,
    UnsupportedBackendError,
)
from vllm_gr.platforms.identity import PlatformIdentity
from vllm_gr.platforms.protocol import PROVIDER_API_VERSION
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
    activate_backend,
    get_active_backend,
    register_backend,
    resolve_backend,
    select_backend,
)


@dataclass(frozen=True)
class FakeProvider:
    backend_id: str = "fake"
    device_type: str = "fake"
    api_version: int = PROVIDER_API_VERSION
    gr_version_range: str | None = None
    vllm_version_range: str | None = None
    vendor_version_range: str | None = None
    vendor_distribution: str | None = None

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

    def create_constrained_topk_execution(self, *, device: Any = None) -> None:
        del device
        return None


def fake_identity(device_type: str = "cuda", **overrides: Any) -> PlatformIdentity:
    """A synthetic identity that never touches a real vLLM platform."""
    fields: dict[str, Any] = {
        "platform_enum": "CUDA",
        "device_type": device_type,
        "device_name": device_type,
        "platform_class": "tests.platforms.fakes.FakePlatform",
        "platform_lineage": ("tests.platforms.fakes.FakePlatform", "builtins.object"),
    }
    fields.update(overrides)
    return PlatformIdentity(**fields)


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


@pytest.mark.parametrize(
    "name,factory",
    [
        ("create_graph_executor", 3),
        ("create_graph_executor", lambda self, plan: None),
        ("create_config_service", lambda self, required: None),
    ],
)
def test_invalid_optional_factory_fails_registration(
    monkeypatch: pytest.MonkeyPatch, name: str, factory: Any
) -> None:
    monkeypatch.setattr(FakeProvider, name, factory, raising=False)
    with pytest.raises(InvalidBackendProviderError, match=name):
        register_backend(FakeProvider(), source="test:malformed")


def test_different_provider_for_same_backend_id_is_rejected() -> None:
    register_backend(FakeProvider(), source="first-package")

    with pytest.raises(DuplicateBackendRegistrationError) as exc_info:
        register_backend(FakeProvider(), source="second-package")

    message = str(exc_info.value)
    assert "requested='fake'" in message
    assert "existing_origin='first-package [backend_id=fake]'" in message
    assert "attempted_origin='second-package [backend_id=fake]'" in message
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
    ascend = FakeProvider(backend_id="ascend", device_type="npu")
    register_backend(CUDA_BACKEND, source="builtin:cuda")
    register_backend(ascend, source="builtin:ascend")
    activate_backend("cuda")

    with pytest.raises(BackendAlreadyActiveError) as exc_info:
        activate_backend("ascend")

    message = str(exc_info.value)
    assert "requested='ascend'" in message
    assert "active='cuda'" in message
    assert "active_origin='builtin:cuda [backend_id=cuda]'" in message
    assert "requested_origin='builtin:ascend [backend_id=ascend]'" in message


def test_empty_backend_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_backend("  ")

    with pytest.raises(InvalidBackendProviderError, match="must not be empty"):
        register_backend(FakeProvider(backend_id="  "))


def test_backend_id_lookup_is_normalized_without_aliases() -> None:
    provider = FakeProvider(backend_id=" CuDa ")
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

        def get_attention_execution(self) -> Any:
            calls.append("attention")
            return object()

        def matches(self, identity: PlatformIdentity) -> bool:
            calls.append("matches")
            return True

    register_backend(LazyProvider())

    assert calls == []


def test_provider_class_and_missing_methods_are_rejected() -> None:
    with pytest.raises(InvalidBackendProviderError, match="not a class"):
        register_backend(FakeProvider)

    class IncompleteProvider:
        backend_id = "incomplete"
        api_version = PROVIDER_API_VERSION
        device_type = "incomplete"

    with pytest.raises(InvalidBackendProviderError, match="required callable"):
        register_backend(IncompleteProvider())


def test_provider_missing_only_constrained_topk_execution_is_rejected() -> None:
    """A provider that has every required method but the new one must be refused.

    Registration-time validation must not let such a provider through and fail
    later with an ``AttributeError`` at the call site.
    """

    class IncompleteProvider:
        backend_id = "incomplete"
        api_version = PROVIDER_API_VERSION
        device_type = "incomplete"

        def matches(self, identity: PlatformIdentity) -> bool:
            return False

        def check_environment(self) -> dict[str, str]:
            return {}

        def apply_runtime_patches(self, *, force: bool = False) -> int:
            del force
            return 0

        def get_model_runner_hooks(self) -> Any:
            return object()

        def get_attention_execution(self) -> Any:
            return object()

    # Deliberately a structurally-invalid provider: typing it as Any keeps
    # strict mypy from rejecting the call site (the point is that the runtime
    # validation — not the type system — must refuse it).
    provider: Any = IncompleteProvider()

    with pytest.raises(
        InvalidBackendProviderError, match="create_constrained_topk_execution"
    ) as exc_info:
        register_backend(provider)

    assert "missing=('create_constrained_topk_execution',)" in str(exc_info.value)


def test_invalid_v2_range_declarations_fail_at_registration() -> None:
    with pytest.raises(InvalidBackendProviderError, match="Invalid gr_version_range.*example"):
        register_backend(FakeProvider(gr_version_range="not-a-range"), source="example")
    with pytest.raises(InvalidBackendProviderError, match="requires vendor_distribution"):
        register_backend(FakeProvider(vendor_version_range=">=1"))


def test_provider_without_matches_is_rejected() -> None:
    @dataclass(frozen=True)
    class NoMatchProvider(FakeProvider):
        matches = None  # type: ignore[assignment]

    with pytest.raises(InvalidBackendProviderError, match="matches"):
        register_backend(NoMatchProvider())


@pytest.mark.parametrize("api_version", [1, 3, "2", None, True])  # type: ignore[misc]
def test_unsupported_api_versions_are_rejected(api_version: object) -> None:
    with pytest.raises(InvalidBackendVersionError) as exc_info:
        register_backend(FakeProvider(api_version=api_version))  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert f"found={api_version!r}" in message
    assert "backend_id=fake" in message


def test_select_requires_exactly_one_match() -> None:
    register_backend(FakeProvider(backend_id="one", device_type="cuda"))
    register_backend(FakeProvider(backend_id="two", device_type="cuda"))

    with pytest.raises(AmbiguousBackendError) as exc_info:
        select_backend(fake_identity("cuda"))

    message = str(exc_info.value)
    assert "backend_id=one" in message
    assert "backend_id=two" in message
    assert "VLLM_GR_BACKEND" in message


def test_select_reports_unsupported_platform_without_fallback() -> None:
    register_backend(CUDA_BACKEND, source="builtin:cuda")

    with pytest.raises(UnsupportedBackendError) as exc_info:
        select_backend(fake_identity("cpu", platform_enum="CPU"))

    message = str(exc_info.value)
    assert "device_type='cpu'" in message
    assert "available=('cuda',)" in message


def test_explicit_selection_resolves_ambiguity_and_must_be_compatible() -> None:
    register_backend(FakeProvider(backend_id="one", device_type="cuda"))
    register_backend(FakeProvider(backend_id="two", device_type="cuda"))

    assert select_backend(fake_identity("cuda"), requested="two").backend_id == "two"

    with pytest.raises(IncompatibleBackendError) as exc_info:
        select_backend(fake_identity("cpu", platform_enum="CPU"), requested="one")

    assert "requested_origin='manual [backend_id=one]'" in str(exc_info.value)


def test_explicit_selection_cannot_bypass_duplicate_ids() -> None:
    register_backend(FakeProvider(), source="first-package")

    with pytest.raises(DuplicateBackendRegistrationError):
        register_backend(FakeProvider(), source="second-package")

    assert select_backend(fake_identity("fake"), requested="fake").backend_id == "fake"


def test_activation_with_identity_rejects_incompatible_request() -> None:
    register_backend(CUDA_BACKEND, source="builtin:cuda")

    with pytest.raises(IncompatibleBackendError):
        activate_backend("cuda", identity=fake_identity("cpu", platform_enum="CPU"))


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
