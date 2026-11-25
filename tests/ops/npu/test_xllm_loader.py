# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host checks for the Python binding contract, without Ascend load claims."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]

REQUIRED = ("cache_unshared_kv", "x_attention", "select_unshared_kv", "rec_constrained_topk")


class DynamicNamespaces:
    """Model namespace creation independently of operator registration."""

    def __getattr__(self, name: str) -> Any:
        namespace = SimpleNamespace()
        setattr(self, name, namespace)
        return namespace


@pytest.fixture(params=["empty-namespace", "failed-preload"])
def loader(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    # Check binding availability both when the current namespace guard returns
    # early and when the optional extension load is attempted but fails.
    torch = ModuleType("torch")
    torch.ops = DynamicNamespaces() if request.param == "empty-namespace" else object()
    utils = ModuleType("torch.utils")
    preload = Mock(side_effect=OSError("library unavailable"))
    utils.cpp_extension = SimpleNamespace(load=preload)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "custom_ops_lib", None)
    spec = importlib.util.spec_from_file_location(
        "_xllm_loader_contract", REPO_ROOT / "vllm_gr/ops/npu/_xllm_loader.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if request.param == "failed-preload":
        assert preload.called


def test_missing_binding_is_unavailable(loader: Any) -> None:
    assert loader.custom_ops() is None
    assert all(not loader.has_op(name) for name in REQUIRED)


def test_import_failure_is_unavailable(
    loader: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delitem(sys.modules, "custom_ops_lib")
    (tmp_path / "custom_ops_lib.py").write_text("raise ImportError('unresolved vendor symbol')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert loader.custom_ops() is None
    assert all(not loader.has_op(name) for name in REQUIRED)


@pytest.mark.parametrize("name", REQUIRED)
@pytest.mark.parametrize("present", [False, True])
def test_missing_or_noncallable_entry_is_unavailable(
    loader: Any, monkeypatch: pytest.MonkeyPatch, name: str, present: bool
) -> None:
    binding = ModuleType("custom_ops_lib")
    for entry in REQUIRED:
        setattr(binding, entry, Mock())
    if present:
        setattr(binding, name, None)
    else:
        delattr(binding, name)
    monkeypatch.setitem(sys.modules, "custom_ops_lib", binding)
    assert loader.custom_ops() is binding
    assert not loader.has_op(name)
    assert all(loader.has_op(entry) for entry in REQUIRED if entry != name)


def test_complete_binding_and_repeated_checks(loader: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    binding = ModuleType("custom_ops_lib")
    for name in REQUIRED:
        setattr(binding, name, Mock())
    monkeypatch.setitem(sys.modules, "custom_ops_lib", binding)
    for _ in range(2):
        assert loader.custom_ops() is binding
        assert all(loader.has_op(name) for name in REQUIRED)
    monkeypatch.setitem(sys.modules, "custom_ops_lib", None)
    assert loader.custom_ops() is None
    assert all(not loader.has_op(name) for name in REQUIRED)
