# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_gr._env_check import (
    check_ascend_environment,
    check_environment,
    check_runtime_versions,
)


class _FakeAscendPlatform:
    """Synthetic Ascend identity: the vendor class lineage, no vendor import."""

    _enum = "OOT"
    device_type = "npu"
    device_name = "npu"


_FakeAscendPlatform.__module__ = "vllm_ascend.platform"
_FakeAscendPlatform.__qualname__ = "NPUPlatform"


@pytest.fixture  # type: ignore[misc]
def backend_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from vllm_gr.platforms import loader
    from vllm_gr.platforms.registry import _reset_backend_registry_for_tests

    _reset_backend_registry_for_tests()
    monkeypatch.setattr(loader, "entry_points", lambda **kwargs: ())
    yield
    _reset_backend_registry_for_tests()


def test_environment_checks_only_vllm_on_non_npu(backend_registry: None) -> None:
    from vllm_gr.platforms.cuda import CUDA_BACKEND
    from vllm_gr.platforms.registry import get_active_backend

    platform = SimpleNamespace(_enum="CUDA", device_type="cuda", device_name="cuda")
    with (
        mock.patch("vllm.platforms.current_platform", platform),
        mock.patch(
            "vllm_gr._env_check.check_runtime_versions", return_value={"vllm": "0.30.0"}
        ) as check_versions,
        mock.patch("vllm_gr._env_check.check_ascend_environment") as check_ascend,
    ):
        assert check_environment() == {"vllm": "0.30.0", "platform": "cuda"}
        assert get_active_backend() is CUDA_BACKEND

    check_versions.assert_called_once_with()
    check_ascend.assert_not_called()


def test_environment_checks_ascend_runtime_on_npu(backend_registry: None) -> None:
    from vllm_gr.platforms.ascend import NPU_BACKEND
    from vllm_gr.platforms.registry import get_active_backend

    platform = _FakeAscendPlatform()
    expected = {"vllm": "0.30.0", "vllm-ascend": "0.30.0rc2", "platform": "npu"}
    with (
        mock.patch("vllm.platforms.current_platform", platform),
        mock.patch("vllm_gr._env_check.check_runtime_versions") as check_versions,
        mock.patch(
            "vllm_gr._env_check.check_ascend_environment", return_value=expected
        ) as check_ascend,
    ):
        assert check_environment() is expected
        assert get_active_backend() is NPU_BACKEND

    check_ascend.assert_called_once_with(require_npu_platform=True)
    check_versions.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
def test_environment_rejects_unknown_platform_without_cuda_fallback(backend_registry: None) -> None:
    from vllm_gr.platforms.errors import UnsupportedBackendError

    with (
        mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(_enum="UNSPECIFIED", device_type="unknown"),
        ),
        mock.patch("vllm_gr._env_check.check_runtime_versions") as check_versions,
        pytest.raises(UnsupportedBackendError, match="unknown"),
    ):
        check_environment()
    check_versions.assert_not_called()


def test_supported_vllm_version() -> None:
    with mock.patch(
        "vllm_gr._env_check.importlib.metadata.version",
        side_effect=lambda name: {"vllm": "0.30.0", "vllm-ascend": "0.30.0rc2+npu"}[name],
    ):
        check_runtime_versions(require_ascend=True)


def test_supported_vllm_local_version() -> None:
    with mock.patch(
        "vllm_gr._env_check.importlib.metadata.version",
        return_value="0.30.0+cu130",
    ):
        check_runtime_versions()


def test_rejects_release_candidate_vllm_version() -> None:
    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            return_value="0.30.0rc2",
        ),
        pytest.raises(RuntimeError, match="required 0.30.0"),
    ):
        check_runtime_versions()


def test_rejects_old_vllm_version() -> None:
    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            return_value="0.29.0",
        ),
        pytest.raises(RuntimeError, match="required 0.30.0"),
    ):
        check_runtime_versions()


def test_rejects_wrong_ascend_version() -> None:
    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            side_effect=lambda name: {"vllm": "0.30.0", "vllm-ascend": "0.30.0"}[name],
        ),
        pytest.raises(RuntimeError, match="required 0.30.0rc2"),
    ):
        check_runtime_versions(require_ascend=True)


def test_requires_ascend_distribution_on_npu() -> None:
    def version(name: str) -> str:
        if name == "vllm":
            return "0.30.0"
        raise PackageNotFoundError(name)

    with mock.patch("vllm_gr._env_check.importlib.metadata.version", side_effect=version):
        with pytest.raises(RuntimeError, match="vllm-ascend"):
            check_runtime_versions(require_ascend=True)


def _ascend_distribution_version(name: str) -> str:
    return {
        "vllm": "0.30.0",
        "vllm-ascend": "0.30.0rc2+npu",
        "triton-ascend": "3.6.0+dev20261027",
    }[name]


def test_ascend_environment_checks_active_triton_backend() -> None:
    fake_triton = mock.Mock(__file__="triton_ascend/triton/__init__.py")

    def import_module(name: str) -> mock.Mock:
        if name == "triton":
            return fake_triton
        if name == "triton._C.libtriton.ascend":
            return mock.Mock()
        raise AssertionError(f"unexpected import: {name}")

    def distribution_version(name: str) -> str:
        if name == "triton":
            raise PackageNotFoundError(name)
        return _ascend_distribution_version(name)

    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            side_effect=distribution_version,
        ),
        mock.patch("vllm_gr._env_check.importlib.import_module", side_effect=import_module),
    ):
        report = check_ascend_environment()

    assert report["triton-ascend"] == "3.6.0+dev20261027"
    assert report["triton-ascend-backend"] == "available"
    assert report["triton"] == "not installed"


def test_ascend_environment_rejects_overwritten_triton_namespace() -> None:
    fake_triton = mock.Mock(__file__="triton/triton/__init__.py")

    def import_module(name: str) -> mock.Mock:
        if name == "triton":
            return fake_triton
        raise ModuleNotFoundError(
            "No module named 'triton._C.libtriton.ascend'; 'triton._C.libtriton' is not a package"
        )

    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            side_effect=_ascend_distribution_version,
        ),
        mock.patch("vllm_gr._env_check.importlib.import_module", side_effect=import_module),
    ):
        with pytest.raises(RuntimeError, match="standard Triton wheel overwrote"):
            check_ascend_environment()


@pytest.mark.parametrize("version", ["3.2.1", "3.6.0rc1", "3.7.0"])  # type: ignore[misc]
def test_ascend_environment_rejects_wrong_triton_ascend_version(version: str) -> None:
    def distribution_version(name: str) -> str:
        versions = {
            "vllm": "0.30.0",
            "vllm-ascend": "0.30.0rc2",
            "triton-ascend": version,
        }
        return versions[name]

    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            side_effect=distribution_version,
        ),
        pytest.raises(RuntimeError, match="required 3.6.0"),
    ):
        check_ascend_environment()


pytestmark = pytest.mark.cpu_test
