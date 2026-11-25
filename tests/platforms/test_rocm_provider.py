# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_gr.platforms import compat
from vllm_gr.platforms.identity import platform_identity
from vllm_gr.platforms.rocm import provider as rocm

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _platform(enum: str) -> SimpleNamespace:
    return SimpleNamespace(_enum=enum, device_type="cuda", device_name="cuda")


def test_rocm_provider_matches_only_the_native_rocm_platform() -> None:
    assert rocm.ROCM_BACKEND.backend_id == "rocm"
    assert rocm.ROCM_BACKEND.matches(platform_identity(_platform("ROCM")))
    assert not rocm.ROCM_BACKEND.matches(platform_identity(_platform("CUDA")))


def test_rocm_provider_is_a_builtin_backend() -> None:
    from vllm_gr.platforms import loader
    from vllm_gr.platforms.registry import resolve_backend

    loader.load_backend_plugins()
    assert resolve_backend("rocm") is rocm.ROCM_BACKEND


@pytest.mark.parametrize(  # type: ignore[misc]
    ("installed", "accepted"),
    [
        ("0.30.0rc2+rocm101", True),
        ("0.30.0", True),
        ("0.30.0rc1", False),
        ("0.29.0", False),
        ("0.30.1", False),
    ],
)
def test_rocm_environment_check_accepts_rc2_and_release(
    monkeypatch: pytest.MonkeyPatch, installed: str, accepted: bool
) -> None:
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = _platform("ROCM")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setattr(compat, "installed_version", lambda name: installed)

    if accepted:
        assert rocm.ROCM_BACKEND.check_environment() == {"vllm": installed, "platform": "rocm"}
    else:
        with pytest.raises(RuntimeError, match="Unsupported vllm version"):
            rocm.ROCM_BACKEND.check_environment()


def test_rocm_environment_check_rejects_other_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = _platform("CUDA")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    with pytest.raises(RuntimeError, match="requires the native ROCm platform"):
        rocm.ROCM_BACKEND.check_environment()
