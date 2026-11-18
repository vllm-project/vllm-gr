# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Load the optional XAttention output binding after checking its build receipt."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, cast

_ADDON_MODULE = "vllm_gr_xattention_out_ext"
_RUNTIME_DISTRIBUTIONS = ("torch", "torch-npu", "vllm", "vllm-ascend")
_PROVIDER_SYMBOLS = {"aclnnXAttention", "aclnnXAttentionGetWorkspaceSize"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_file_identity(path: str, identity: dict[str, str]) -> bool:
    actual = Path(path).resolve(strict=True)
    return (
        actual == Path(identity["path"]).resolve(strict=True)
        and _sha256(actual) == identity["sha256"]
    )


def _binding_is_compatible(module: ModuleType) -> bool:
    """Check runtime and artifact consistency once before attention dispatch."""
    if module.__file__ is None:
        return False
    addon_path = Path(module.__file__).resolve(strict=True)
    receipt = json.loads(
        addon_path.with_name(f"{_ADDON_MODULE}.receipt.json").read_text(encoding="utf-8")
    )
    if (
        receipt["status"] != "built"
        or receipt["schema_version"] != 1
        or receipt["abi_version"] != 1
        or receipt["python_abi"] != list(sys.version_info[:2])
    ):
        return False
    if receipt["runtime"] != {
        name: importlib.metadata.version(name) for name in _RUNTIME_DISTRIBUTIONS
    }:
        return False
    if not _same_file_identity(str(addon_path), receipt["addon"]):
        return False
    custom_ops = import_module("custom_ops_lib")
    if custom_ops.__file__ is None or not _same_file_identity(
        custom_ops.__file__, receipt["custom_ops"]
    ):
        return False
    info = module.binding_info()
    if info["abi_version"] != 1:
        return False
    if (
        set(info["providers"]) != _PROVIDER_SYMBOLS
        or set(receipt["providers"]) != _PROVIDER_SYMBOLS
    ):
        return False
    return all(
        _same_file_identity(path, receipt["providers"][symbol])
        for symbol, path in info["providers"].items()
    )


@lru_cache(maxsize=1)
def _load_binding() -> Callable[..., Any] | None:
    try:
        module = import_module(_ADDON_MODULE)
        operation = getattr(module, "x_attention_out", None)
        if callable(operation) and _binding_is_compatible(module):
            return cast(Callable[..., Any], operation)
    except (ImportError, OSError, AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        # All checks precede native attention dispatch. A missing/stale addon
        # preserves the original allocating kernel and output copy.
        return None
    return None


def x_attention_out_available() -> bool:
    """Probe once, before constructing a view or checking physical NPU layout."""
    return _load_binding() is not None


def get_x_attention_out(output: Any) -> Callable[..., Any] | None:
    """Decline unsupported physical output layouts before native dispatch."""
    if output.device.type != "npu":
        return None
    try:
        torch_npu = import_module("torch_npu")
    except (ImportError, OSError):
        return None
    if torch_npu.get_npu_format(output) != 2:
        return None
    return _load_binding()
