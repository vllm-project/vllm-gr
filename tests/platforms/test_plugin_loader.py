# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for exact, demand-loaded backend entry-point selection."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from vllm_gr.platforms import loader
from vllm_gr.platforms.errors import (
    DuplicateBackendRegistrationError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from vllm_gr.platforms.protocol import BACKEND_API_VERSION
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
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

    def release_accelerator_memory(self) -> None:
        pass

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> Any:
        return object()

    def get_attention_hooks(self) -> Any:
        return object()


@dataclass
class FakeEntryPoint:
    name: str
    provider: Any = None
    value: str = "tests:PROVIDER"
    distribution_name: str = "vllm-gr"
    error: Exception | None = None
    load_count: int = 0

    @property
    def dist(self) -> SimpleNamespace:
        return SimpleNamespace(name=self.distribution_name)

    def load(self) -> Any:
        self.load_count += 1
        if self.error is not None:
            raise self.error
        return self.provider


def setup_function() -> None:
    _reset_backend_registry_for_tests()
    loader._reset_backend_loader_for_tests()


def teardown_function() -> None:
    _reset_backend_registry_for_tests()
    loader._reset_backend_loader_for_tests()


def _install_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    *fake_entry_points: FakeEntryPoint,
) -> mock.Mock:
    discover = mock.Mock(return_value=fake_entry_points)
    monkeypatch.setattr(loader, "entry_points", discover)
    return discover


def test_distribution_declares_cuda_backend_entry_point() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert pyproject["project"]["entry-points"][loader.ENTRYPOINT_GROUP] == {
        "cuda": "vllm_gr.platforms.cuda:CUDA_BACKEND",
    }


def test_loads_only_selected_provider_after_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_provider = FakeProvider(device_type="cuda")
    cuda_entry_point = FakeEntryPoint("cuda", cuda_provider)
    broken_other = FakeEntryPoint("other", error=ImportError("broken provider module"))
    discover = _install_entry_points(monkeypatch, broken_other, cuda_entry_point)

    assert loader.load_backend_plugin("cuda") is cuda_provider

    assert resolve_backend("cuda") is cuda_provider
    with pytest.raises(UnsupportedBackendError):
        resolve_backend("other")
    discover.assert_called_once_with(group=loader.ENTRYPOINT_GROUP)
    assert cuda_entry_point.load_count == 1
    assert broken_other.load_count == 0


def test_other_selection_does_not_load_broken_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_provider = FakeProvider(device_type="other")
    other_entry_point = FakeEntryPoint("other", other_provider)
    broken_cuda = FakeEntryPoint("cuda", error=ImportError("broken CUDA module"))
    _install_entry_points(monkeypatch, broken_cuda, other_entry_point)

    assert loader.load_backend_plugin("other") is other_provider
    assert other_entry_point.load_count == 1
    assert broken_cuda.load_count == 0


def test_entry_point_name_must_exactly_match_provider_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint("cuda", FakeProvider(device_type="CUDA"))
    _install_entry_points(monkeypatch, entry_point)

    with pytest.raises(InvalidBackendProviderError) as exc_info:
        loader.load_backend_plugin("cuda")

    message = str(exc_info.value)
    assert "entrypoint_name='cuda'" in message
    assert "provider_device_type='CUDA'" in message


def test_selected_plugin_loader_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(device_type="cuda")
    entry_point = FakeEntryPoint("cuda", provider)
    discover = _install_entry_points(monkeypatch, entry_point)

    assert loader.load_backend_plugin("cuda") is provider
    assert loader.load_backend_plugin("cuda") is provider

    discover.assert_called_once_with(group=loader.ENTRYPOINT_GROUP)
    assert entry_point.load_count == 1


def test_duplicate_matching_entry_points_fail_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeEntryPoint("cuda", FakeProvider(device_type="cuda"), distribution_name="a")
    second = FakeEntryPoint("cuda", FakeProvider(device_type="cuda"), distribution_name="b")
    _install_entry_points(monkeypatch, first, second)

    with pytest.raises(DuplicateBackendRegistrationError) as exc_info:
        loader.load_backend_plugin("cuda")

    message = str(exc_info.value)
    assert "entrypoint:a:cuda=tests:PROVIDER" in message
    assert "entrypoint:b:cuda=tests:PROVIDER" in message
    assert first.load_count == 0
    assert second.load_count == 0


def test_selected_entry_point_load_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint("cuda", error=ImportError("broken CUDA module"))
    _install_entry_points(monkeypatch, entry_point)

    with pytest.raises(InvalidBackendProviderError, match="Failed to load the selected"):
        loader.load_backend_plugin("cuda")


def test_api_mismatch_fails_only_when_provider_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_provider = FakeProvider(device_type="cuda")
    cuda_entry_point = FakeEntryPoint("cuda", cuda_provider)
    invalid_other = FakeEntryPoint(
        "other",
        FakeProvider(api_version=BACKEND_API_VERSION + 1, device_type="other"),
    )
    _install_entry_points(monkeypatch, invalid_other, cuda_entry_point)

    assert loader.load_backend_plugin("cuda") is cuda_provider
    assert invalid_other.load_count == 0

    with pytest.raises(InvalidBackendProviderError, match="API version is incompatible"):
        loader.load_backend_plugin("other")
    assert invalid_other.load_count == 1


def test_selected_entry_point_must_return_provider_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("fake", FakeProvider),
    )

    with pytest.raises(InvalidBackendProviderError, match="not a class"):
        loader.load_backend_plugin("fake")


@pytest.mark.parametrize("device_type", ["cpu", "unknown"])  # type: ignore[misc]
def test_missing_backend_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("cuda", FakeProvider(device_type="cuda")),
    )

    with pytest.raises(UnsupportedBackendError) as exc_info:
        loader.load_backend_plugin(device_type)

    assert f"requested='{device_type}'" in str(exc_info.value)


def test_activate_current_backend_loads_exact_platform_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_module = ModuleType("vllm")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="cuda")  # type: ignore[attr-defined]
    vllm_module.platforms = platforms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    cuda_provider = FakeProvider(device_type="cuda")
    cuda_entry_point = FakeEntryPoint("cuda", cuda_provider)
    other_entry_point = FakeEntryPoint("other", error=ImportError("broken provider module"))
    _install_entry_points(monkeypatch, other_entry_point, cuda_entry_point)

    assert loader.activate_current_backend() is cuda_provider
    assert cuda_entry_point.load_count == 1
    assert other_entry_point.load_count == 0
