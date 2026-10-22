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
    setup_path = REPO_ROOT / "setup.py"
    spec = importlib.util.spec_from_file_location("vllm_gr_setup", setup_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Lightweight test environments may not have setuptools_scm. Keep its
    # stub local to this load so later setuptools entry points can import the
    # real package. Restore both module entries even if setup.py raises.
    with pytest.MonkeyPatch.context() as patcher:
        if "setuptools_scm" not in sys.modules:
            patcher.setitem(sys.modules, "setuptools_scm", MagicMock())
        patcher.setitem(sys.modules, "vllm_gr_setup", module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("preloaded", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("load_error", [False, True])  # type: ignore[misc]
def test_setup_module_load_restores_import_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, preloaded: bool, load_error: bool
) -> None:
    names = ("setuptools_scm", "vllm_gr_setup")
    existing = {name: ModuleType(name) for name in names}
    get_version = MagicMock(return_value="0.22.1")
    existing["setuptools_scm"].get_version = get_version  # type: ignore[attr-defined]
    for name in names:
        if preloaded:
            monkeypatch.setitem(sys.modules, name, existing[name])
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)

    if load_error:
        (tmp_path / "setup.py").write_text(
            "from setuptools_scm import get_version\nraise RuntimeError('setup load failed')\n"
        )
        monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
        with pytest.raises(RuntimeError, match="setup load failed"):
            _load_setup_module()
    else:
        module = _load_setup_module()
        assert callable(module.get_version)
        if preloaded:
            assert module.get_version is get_version
            assert module.get_version() == "0.22.1"
            get_version.assert_called_once_with()

    for name in names:
        if preloaded:
            assert sys.modules[name] is existing[name]
        else:
            assert name not in sys.modules


@pytest.mark.parametrize(  # type: ignore[misc]
    ("device", "expected_subset", "excluded"),
    [
        (
            "cpu",
            {"msgspec", "safetensors>=0.4.5", "PyYAML>=6.0"},
            {"vllm"},
        ),
        (
            "cuda",
            {"msgspec", "safetensors>=0.4.5", "PyYAML>=6.0"},
            {"vllm"},
        ),
        (
            "npu",
            {"msgspec", "safetensors>=0.4.5", "PyYAML>=6.0"},
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
