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
    assert not any(line.startswith("vllm") for line in lines)
    assert "pytest" not in lines
    assert "ruff" not in lines


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
