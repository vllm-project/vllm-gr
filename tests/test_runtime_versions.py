# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.metadata import PackageNotFoundError
from unittest import mock

import pytest

from vllm_gr._env_check import (
    check_ascend_environment,
    check_environment,
    check_runtime_versions,
)


def test_environment_checks_only_vllm_on_non_npu() -> None:
    for device_type in ("cpu", "cuda"):
        platform = mock.Mock(device_type=device_type)
        expected = {"vllm": "0.30.0"}

        with (
            mock.patch("vllm.platforms.current_platform", platform),
            mock.patch(
                "vllm_gr._env_check.check_runtime_versions", return_value=expected
            ) as check_versions,
            mock.patch("vllm_gr._env_check.check_ascend_environment") as check_ascend,
        ):
            report = check_environment()

        assert report is expected
        check_versions.assert_called_once_with()
        check_ascend.assert_not_called()


def test_environment_checks_ascend_runtime_on_npu() -> None:
    platform = mock.Mock(device_type="npu")
    expected = {"vllm": "0.30.0", "vllm-ascend": "0.30.0rc2"}

    with (
        mock.patch("vllm.platforms.current_platform", platform),
        mock.patch("vllm_gr._env_check.check_runtime_versions") as check_versions,
        mock.patch(
            "vllm_gr._env_check.check_ascend_environment", return_value=expected
        ) as check_ascend,
    ):
        report = check_environment()

    assert report is expected
    check_ascend.assert_called_once_with()
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
        "triton-ascend": "3.2.1",
    }[name]


def test_ascend_environment_checks_active_triton_backend(configured_ascend_runtime: None) -> None:
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

    assert report["triton-ascend"] == "3.2.1"
    assert report["triton-ascend-backend"] == "available"
    assert report["triton"] == "not installed"


def test_ascend_environment_rejects_overwritten_triton_namespace(
    configured_ascend_runtime: None,
) -> None:
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


def test_ascend_environment_rejects_wrong_triton_ascend_version(
    configured_ascend_runtime: None,
) -> None:
    def distribution_version(name: str) -> str:
        versions = {
            "vllm": "0.30.0",
            "vllm-ascend": "0.30.0rc2",
            "triton-ascend": "3.3.0",
        }
        return versions[name]

    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            side_effect=distribution_version,
        ),
        pytest.raises(RuntimeError, match="required 3.2.1"),
    ):
        check_ascend_environment()


@pytest.fixture  # type: ignore[misc]
def configured_ascend_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin a mismatched Triton-Ascend version so guard tests do not depend on
    # the shipped pin.
    monkeypatch.setattr("vllm_gr._env_check._SUPPORTED_TRITON_ASCEND_VERSION", "3.2.1")


def test_unset_triton_ascend_pin_fails_before_importing_vendor_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shipped default pins the validated runtime, so construct the
    # unset-pin state explicitly instead of relying on the module constant.
    monkeypatch.setattr("vllm_gr._env_check._SUPPORTED_TRITON_ASCEND_VERSION", None)
    with mock.patch("vllm_gr._env_check.importlib.import_module") as importer:
        with pytest.raises(RuntimeError, match="Ascend support is not enabled.*unset"):
            check_ascend_environment()
    importer.assert_not_called()
