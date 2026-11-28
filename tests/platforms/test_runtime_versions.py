# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from argparse import Namespace
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from unittest import mock

import pytest

from tests.helpers.ascend_platform import AscendGRPlatform
from tests.helpers.native_platform import native_platform
from vllm_gr.platforms import get_current_gr_platform
from vllm_gr.platforms.ascend.environment import check_ascend_environment
from vllm_gr.platforms.compat import check_vllm_runtime, require_runtime_version


class _FakeAscendPlatform(AscendGRPlatform.native_platform_type):  # type: ignore[misc]
    pass


def test_environment_checks_only_vllm_on_non_npu() -> None:
    from tests.helpers.cuda_platform import CudaGRPlatform

    platform = native_platform("CUDA", "cuda")
    with (
        mock.patch("vllm.platforms.current_platform", platform),
        mock.patch(
            "vllm_gr.platforms.compat.check_vllm_runtime", return_value={"vllm": "0.30.0"}
        ) as check_versions,
        mock.patch("vllm_gr.platforms.ascend.environment.check_ascend_environment") as check_ascend,
    ):
        assert get_current_gr_platform().check_environment() == {
            "vllm": "0.30.0",
            "platform": "cuda",
        }
        assert isinstance(get_current_gr_platform(), CudaGRPlatform)

    check_versions.assert_called_once_with()
    check_ascend.assert_not_called()


def test_environment_checks_ascend_runtime_on_npu() -> None:
    from tests.helpers.ascend_platform import AscendGRPlatform

    platform = _FakeAscendPlatform()
    expected = {"vllm": "0.30.0", "vllm-ascend": "0.30.0rc2", "platform": "npu"}
    with (
        mock.patch("vllm.platforms.current_platform", platform),
        mock.patch("vllm_gr.platforms.compat.check_vllm_runtime") as check_versions,
        mock.patch(
            "vllm_gr.platforms.ascend.environment.check_ascend_environment", return_value=expected
        ) as check_ascend,
    ):
        assert get_current_gr_platform().check_environment() is expected
        assert isinstance(get_current_gr_platform(), AscendGRPlatform)

    check_ascend.assert_called_once_with(require_npu_platform=True)
    check_versions.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
def test_environment_rejects_unknown_platform_without_cuda_fallback() -> None:
    from vllm_gr.platforms.errors import UnsupportedBackendError

    with (
        mock.patch(
            "vllm.platforms.current_platform",
            native_platform("UNSPECIFIED", "unknown"),
        ),
        mock.patch("vllm_gr.platforms.compat.check_vllm_runtime") as check_versions,
        pytest.raises(UnsupportedBackendError, match="unknown"),
    ):
        get_current_gr_platform().check_environment()
    check_versions.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("versions", "check", "expected"),
    [
        (
            {"vllm": "0.30.0", "vllm-ascend": "0.30.0rc2+npu"},
            lambda: require_runtime_version("vllm-ascend", "0.30.0rc2"),
            "0.30.0rc2+npu",
        ),
        ({"vllm": "0.30.0+cu130"}, check_vllm_runtime, {"vllm": "0.30.0+cu130"}),
    ],
    ids=["vendor-local-suffix", "core-local-suffix"],
)
def test_supported_runtime_version_keeps_the_local_suffix(
    versions: dict[str, str], check: Callable[[], object], expected: object
) -> None:
    with mock.patch(
        "vllm_gr.platforms.compat.metadata.version",
        side_effect=lambda name: versions[name],
    ):
        assert check() == expected


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "installed", ["0.29.0", "0.30.0rc2", "0.30+vendor"], ids=["older", "rc", "shortened"]
)
def test_rejects_unsupported_vllm_versions(installed: str) -> None:
    with (
        mock.patch("vllm_gr.platforms.compat.metadata.version", return_value=installed),
        pytest.raises(RuntimeError, match="required 0.30.0"),
    ):
        check_vllm_runtime()


def test_rejects_wrong_ascend_version() -> None:
    with (
        mock.patch(
            "vllm_gr.platforms.compat.metadata.version",
            side_effect=lambda name: {"vllm": "0.30.0", "vllm-ascend": "0.30.0"}[name],
        ),
        pytest.raises(RuntimeError, match="required 0.30.0rc2"),
    ):
        require_runtime_version("vllm-ascend", "0.30.0rc2")


def test_requires_ascend_distribution_on_npu() -> None:
    def version(name: str) -> str:
        if name == "vllm":
            return "0.30.0"
        raise PackageNotFoundError(name)

    with mock.patch("vllm_gr.platforms.compat.metadata.version", side_effect=version):
        with pytest.raises(RuntimeError, match="vllm-ascend"):
            require_runtime_version("vllm-ascend", "0.30.0rc2")


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
            "vllm_gr.platforms.compat.metadata.version",
            side_effect=distribution_version,
        ),
        mock.patch("vllm_gr.platforms.ascend.environment.import_module", side_effect=import_module),
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
            "vllm_gr.platforms.compat.metadata.version",
            side_effect=_ascend_distribution_version,
        ),
        mock.patch("vllm_gr.platforms.ascend.environment.import_module", side_effect=import_module),
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
            "vllm_gr.platforms.compat.metadata.version",
            side_effect=distribution_version,
        ),
        pytest.raises(RuntimeError, match="required 3.6.0"),
    ):
        check_ascend_environment()


@pytest.mark.l1  # type: ignore[misc]
def test_require_npu_accepts_external_platform_on_native_ascend() -> None:
    from vllm_gr.entrypoints.cli.check_env import CheckEnvSubcommand

    external = mock.Mock(backend_id="vendor-ascend")
    external.check_environment.return_value = {"platform": "vendor-ascend"}
    with (
        mock.patch("vllm.platforms.current_platform", _FakeAscendPlatform()),
        mock.patch("vllm_gr.platforms.get_current_gr_platform", return_value=external),
    ):
        CheckEnvSubcommand.cmd(Namespace(npu=True))
    external.check_environment.assert_called_once_with()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
def test_check_env_cli_uses_shared_runtime_checks(
    device: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from vllm_gr.entrypoints.cli.check_env import CheckEnvSubcommand

    versions = {"vllm": "0.30.0+local"}
    if device == "npu":
        versions.update({"vllm-ascend": "0.30.0rc2+npu", "triton-ascend": "3.6.0"})
    platform = _FakeAscendPlatform() if device == "npu" else native_platform("CUDA", "cuda")

    def version(name: str) -> str:
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    with (
        mock.patch("vllm.platforms.current_platform", platform),
        mock.patch("vllm_gr.platforms.compat.metadata.version", side_effect=version),
        mock.patch("vllm_gr.platforms.ascend.environment.import_module") as import_vendor,
    ):
        CheckEnvSubcommand.cmd(Namespace(npu=False))

    report = capsys.readouterr().out
    assert "environment check passed" in report
    assert f"platform: {device}" in report
    for name, installed in versions.items():
        assert f"{name}: {installed}" in report
    if device == "npu":
        import_vendor.assert_any_call("triton._C.libtriton.ascend")
    else:
        import_vendor.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("platform_enum", "placement", "require_npu", "message"),
    [
        ("ROCM", "cuda", False, "ROCM"),
        ("OOT", "npu", False, "OOT"),
        ("CUDA", "cuda", True, "--npu requires the native Ascend platform"),
    ],
)
def test_check_env_cli_rejects_unsupported_native_identity(
    monkeypatch: pytest.MonkeyPatch,
    platform_enum: str,
    placement: str,
    require_npu: bool,
    message: str,
) -> None:
    import vllm.platforms

    from vllm_gr.entrypoints.cli.check_env import CheckEnvSubcommand

    native = native_platform(platform_enum, "foreign" if platform_enum == "OOT" else placement)
    native.device_type = placement
    monkeypatch.setattr(vllm.platforms, "current_platform", native)
    with pytest.raises(RuntimeError, match=message):
        CheckEnvSubcommand.cmd(Namespace(npu=require_npu))


pytestmark = pytest.mark.cpu_test
