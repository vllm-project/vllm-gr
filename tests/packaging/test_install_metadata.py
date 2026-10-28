# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for platform-aware install metadata routing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_FILE = REPO_ROOT / "requirements" / "common.txt"


def _load_setup_module() -> ModuleType:
    # setup.py imports setuptools_scm at the top level, which may not be
    # installed in lightweight test environments.  Inject a stub so the
    # module can be loaded without the real package.
    if "setuptools_scm" not in sys.modules:
        try:
            __import__("setuptools_scm")
        except ModuleNotFoundError as exc:
            if exc.name != "setuptools_scm":
                raise
            setuptools_scm_stub = ModuleType("setuptools_scm")
            setattr(setuptools_scm_stub, "get_version", MagicMock())
            sys.modules["setuptools_scm"] = setuptools_scm_stub

    setup_path = REPO_ROOT / "setup.py"
    spec = importlib.util.spec_from_file_location("vllm_gr_setup", setup_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm_gr_setup"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(  # type: ignore[misc]
    ("device", "expected_subset", "excluded"),
    [
        (
            "cpu",
            {"msgspec", "safetensors>=0.4.5"},
            {"vllm"},
        ),
        (
            "cuda",
            {"msgspec", "safetensors>=0.4.5"},
            {"vllm"},
        ),
        (
            "npu",
            {"msgspec", "safetensors>=0.4.5"},
            {"vllm"},
        ),
    ],
)
def test_get_install_requires_by_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_subset: set[str],
    excluded: set[str],
) -> None:
    gr_setup = _load_setup_module()
    # Reset the module-level cache so the monkeypatched function is used.
    monkeypatch.setattr(gr_setup, "_cached_device", None)
    monkeypatch.setattr(gr_setup, "detect_target_device", lambda: device)
    monkeypatch.setattr(gr_setup, "warn_deprecated_extra_install", lambda: None)

    requirements = set(gr_setup.get_install_requires())

    assert expected_subset.issubset(requirements)
    assert requirements.isdisjoint(excluded)


def test_common_requirements_exclude_vllm_and_dev_tools() -> None:
    content = COMMON_FILE.read_text()
    lines = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert "pytest" not in lines
    assert "ruff" not in lines


def test_release_version_by_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_versions = {
        "cuda": "0.22.1",
        "npu": "0.22.1+npu",
        "cpu": "0.22.1+cpu",
    }

    for device, expected in expected_versions.items():
        gr_setup = _load_setup_module()
        monkeypatch.delenv("VLLM_GR_VERSION_OVERRIDE", raising=False)
        monkeypatch.setattr(gr_setup, "detect_target_device", lambda device=device: device)
        write_version = MagicMock(return_value=expected)
        monkeypatch.setattr(gr_setup, "get_version", write_version)

        assert gr_setup.get_vllm_gr_version() == expected
        assert gr_setup.VLLM_GR_BASE_VERSION == "0.22.1"
        write_version.assert_called_once_with(write_to="vllm_gr/_version.py")


def test_release_version_override(monkeypatch: pytest.MonkeyPatch) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setenv("VLLM_GR_VERSION_OVERRIDE", "0.22.1.dev1")
    write_version = MagicMock(return_value="0.22.1.dev1")
    monkeypatch.setattr(gr_setup, "get_version", write_version)

    assert gr_setup.get_vllm_gr_version() == "0.22.1.dev1"
    write_version.assert_called_once_with(write_to="vllm_gr/_version.py")


def test_warn_deprecated_extra_install_emits_warning() -> None:
    import warnings

    gr_setup = _load_setup_module()
    original_argv = sys.argv
    try:
        sys.argv = ["pip", "install", "-e", ".[gpu]"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gr_setup.warn_deprecated_extra_install()
        assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    finally:
        sys.argv = original_argv


def test_build_rust_sidecar_packages_release_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gr_setup = _load_setup_module()
    build_lib = tmp_path / "build-lib"
    target_dir = tmp_path / "rust-target"
    cargo = tmp_path / "cargo"
    cargo.write_text("", encoding="utf-8")
    monkeypatch.delenv(gr_setup.SKIP_RUST_BUILD_ENV, raising=False)
    monkeypatch.setattr(gr_setup.shutil, "which", lambda name: str(cargo))

    def run(command: list[str], *, cwd: Path, check: bool) -> None:
        assert command[:4] == [str(cargo), "build", "--release", "--locked"]
        assert command[command.index("--manifest-path") + 1] == str(gr_setup.RUST_MANIFEST)
        assert command[command.index("--target-dir") + 1] == str(target_dir)
        assert cwd == gr_setup.REPO_ROOT
        assert check is True
        compiled = target_dir / "release" / gr_setup.RUST_BINARY_NAME
        compiled.parent.mkdir(parents=True)
        compiled.write_bytes(b"beam-proc")
        assert "--lib" in command
        assert command[command.index("--features") + 1] == "iceoryx"
        (compiled.parent / gr_setup.RUST_LIBRARY_NAME).write_bytes(b"ce-transport")

    monkeypatch.setattr(gr_setup.subprocess, "run", run)

    packaged = gr_setup.build_rust_sidecar(build_lib, target_dir)

    assert packaged is not None
    assert packaged == build_lib / "vllm_gr" / "bin" / gr_setup.RUST_BINARY_NAME
    assert packaged.read_bytes() == b"beam-proc"
    assert (packaged.parent / gr_setup.RUST_LIBRARY_NAME).read_bytes() == b"ce-transport"
    assert packaged.stat().st_mode & 0o111


def test_build_py_uses_runtime_source_binary_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setattr(gr_setup.build_py, "run", lambda self: None)
    build_sidecar = MagicMock()
    monkeypatch.setattr(gr_setup, "build_rust_sidecar", build_sidecar)

    command = gr_setup.BuildPyWithRust(gr_setup.BinaryDistribution())
    command.build_lib = str(tmp_path / "build-lib")
    command.run()

    build_sidecar.assert_called_once_with(
        tmp_path / "build-lib",
        gr_setup.REPO_ROOT / "rust" / "target",
    )


def test_binary_wheel_is_python_independent_and_platform_specific() -> None:
    gr_setup = _load_setup_module()
    distribution = gr_setup.BinaryDistribution()
    command = gr_setup.BdistWheelWithRust(distribution)
    command.ensure_finalized()

    assert distribution.has_ext_modules() is True
    assert command.root_is_pure is False
    python_tag, abi_tag, platform_tag = command.get_tag()

    assert python_tag == "py3"
    assert abi_tag == "none"
    assert platform_tag != "any"
