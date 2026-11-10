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
from packaging.requirements import Requirement

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
    ("device", "expected_subset", "expected_packages", "excluded_packages"),
    [
        (
            "cpu",
            {"msgspec", "regex", "safetensors>=0.6.2", "PyYAML>=6.0"},
            {"msgspec", "regex", "safetensors", "pyyaml"},
            {"vllm"},
        ),
        (
            "cuda",
            {
                "msgspec",
                "regex",
                "safetensors>=0.6.2",
                "PyYAML>=6.0",
                "vllm==0.30.0",
            },
            {"msgspec", "regex", "safetensors", "pyyaml", "vllm"},
            set(),
        ),
        (
            "npu",
            {"msgspec", "regex", "safetensors>=0.6.2", "PyYAML>=6.0"},
            {"msgspec", "regex", "safetensors", "pyyaml"},
            {"vllm"},
        ),
    ],
)
def test_get_install_requires_by_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_subset: set[str],
    expected_packages: set[str],
    excluded_packages: set[str],
) -> None:
    gr_setup = _load_setup_module()
    # Reset the module-level cache so the monkeypatched function is used.
    monkeypatch.setattr(gr_setup, "_cached_device", None)
    monkeypatch.setattr(gr_setup, "detect_target_device", lambda: device)

    requirements = set(gr_setup.get_install_requires())
    package_names = {
        Requirement(requirement).name.lower().replace("_", "-") for requirement in requirements
    }

    assert expected_subset.issubset(requirements)
    assert expected_packages.issubset(package_names)
    assert package_names.isdisjoint(excluded_packages)


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
        "cuda": "0.30.0",
        "npu": "0.30.0+npu",
        "cpu": "0.30.0+cpu",
    }

    for device, expected in expected_versions.items():
        gr_setup = _load_setup_module()
        monkeypatch.delenv("VLLM_GR_VERSION_OVERRIDE", raising=False)
        monkeypatch.setattr(gr_setup, "detect_target_device", lambda device=device: device)
        write_version = MagicMock(return_value=expected)
        monkeypatch.setattr(gr_setup, "get_version", write_version)

        assert gr_setup.get_vllm_gr_version() == expected
        assert gr_setup.VLLM_GR_BASE_VERSION == "0.30.0"
        write_version.assert_called_once_with(write_to="vllm_gr/_version.py")


def test_release_version_override(monkeypatch: pytest.MonkeyPatch) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setenv("VLLM_GR_VERSION_OVERRIDE", "0.30.0.dev1")
    write_version = MagicMock(return_value="0.30.0.dev1")
    monkeypatch.setattr(gr_setup, "get_version", write_version)

    assert gr_setup.get_vllm_gr_version() == "0.30.0.dev1"
    write_version.assert_called_once_with(write_to="vllm_gr/_version.py")


pytestmark = pytest.mark.cpu_test
