# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for platform-aware install metadata routing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_FILE = REPO_ROOT / "requirements" / "common.txt"


def _load_setup_module() -> ModuleType:
    # setup.py imports setuptools_scm at the top level, which may not be
    # installed in lightweight test environments.  Inject a stub so the
    # module can be loaded without the real package.
    if "setuptools_scm" not in sys.modules:
        sys.modules["setuptools_scm"] = MagicMock()

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


def test_missing_selected_backend_requirements_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setattr(gr_setup, "detect_target_device", lambda: "missing")
    monkeypatch.setattr(gr_setup, "warn_deprecated_extra_install", lambda: None)
    monkeypatch.setattr(gr_setup, "REQUIREMENTS_DIR", tmp_path)

    with pytest.raises(
        RuntimeError,
        match="Detected target device 'missing'.*does not include that backend",
    ):
        gr_setup.get_install_requires()


def test_missing_recursive_requirements_file_is_not_ignored(tmp_path: Path) -> None:
    gr_setup = _load_setup_module()
    requirements_file = tmp_path / "cuda.txt"
    requirements_file.write_text("-r missing.txt\n")

    with pytest.raises(RuntimeError, match="Requirements file not found"):
        gr_setup.load_requirements(requirements_file)


def test_empty_selected_backend_requirements_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gr_setup = _load_setup_module()
    (tmp_path / "empty.txt").write_text("# intentionally empty\n")
    monkeypatch.setattr(gr_setup, "detect_target_device", lambda: "empty")
    monkeypatch.setattr(gr_setup, "warn_deprecated_extra_install", lambda: None)
    monkeypatch.setattr(gr_setup, "REQUIREMENTS_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="does not define any install dependencies"):
        gr_setup.get_install_requires()


def test_target_device_override_has_highest_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setattr(gr_setup, "_cached_device", None)
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", "CUDA")
    monkeypatch.setenv("READTHEDOCS", "1")

    assert gr_setup.detect_target_device() == "cuda"


def test_readthedocs_selects_cpu_before_torch_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setattr(gr_setup, "_cached_device", None)
    monkeypatch.delenv("VLLM_GR_TARGET_DEVICE", raising=False)
    monkeypatch.setenv("READTHEDOCS", "1")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(cuda="12.8")),
    )

    assert gr_setup.detect_target_device() == "cpu"


@pytest.mark.parametrize(  # type: ignore[misc]
    ("torch_module", "expected"),
    [
        (SimpleNamespace(version=SimpleNamespace(cuda="12.8")), "cuda"),
        (SimpleNamespace(version=SimpleNamespace(cuda=None)), "cpu"),
    ],
)
def test_torch_backend_auto_detection(
    monkeypatch: pytest.MonkeyPatch,
    torch_module: object,
    expected: str,
) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setattr(gr_setup, "_cached_device", None)
    monkeypatch.delenv("VLLM_GR_TARGET_DEVICE", raising=False)
    monkeypatch.delenv("READTHEDOCS", raising=False)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    assert gr_setup.detect_target_device() == expected


def test_missing_torch_defaults_to_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setattr(gr_setup, "_cached_device", None)
    monkeypatch.delenv("VLLM_GR_TARGET_DEVICE", raising=False)
    monkeypatch.delenv("READTHEDOCS", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    assert gr_setup.detect_target_device() == "cuda"


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
