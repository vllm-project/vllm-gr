# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""Host contracts for the optional output binding and its explicit build."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "tools/npu/xattention_out/build.py"


def load_source(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def binding() -> Any:
    return load_source(ROOT / "vllm_gr/ops/npu/_xattention_out.py", "_binding_contract")


@pytest.fixture
def addon(binding: Any, tmp_path: Path, monkeypatch: Any) -> Any:
    extension = tmp_path / "vllm_gr_xattention_out_ext.so"
    custom_ops = tmp_path / "custom_ops_lib.so"
    provider = tmp_path / "libcust_opapi.so"
    for path in (extension, custom_ops, provider):
        path.write_bytes(path.name.encode())
    versions = {name: f"version-{name}" for name in binding._RUNTIME_DISTRIBUTIONS}
    info = {
        "abi_version": 1,
        "source_pins": dict(binding._SOURCE_PINS),
        "providers": {name: str(provider) for name in binding._PROVIDER_SYMBOLS},
    }
    module = ModuleType("vllm_gr_xattention_out_ext")
    module.__file__ = str(extension)
    operation = Mock()
    setattr(module, "x_attention_out", operation)
    describe = Mock(return_value=info)
    setattr(module, "binding_info", describe)

    def identity(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": binding._sha256(path)}

    receipt = {
        "status": "built",
        "schema_version": 1,
        "abi_version": 1,
        "python_abi": list(sys.version_info[:2]),
        "source_pins": dict(binding._SOURCE_PINS),
        "runtime": versions,
        "addon": identity(extension),
        "custom_ops": identity(custom_ops),
        "providers": {name: identity(provider) for name in binding._PROVIDER_SYMBOLS},
    }
    receipt_path = tmp_path / "vllm_gr_xattention_out_ext.receipt.json"
    receipt_path.write_text(json.dumps(receipt))

    def importer(name: str) -> Any:
        if name == "vllm_gr_xattention_out_ext":
            return module
        if name == "custom_ops_lib":
            return SimpleNamespace(__file__=str(custom_ops))
        if name == "torch_npu":
            return SimpleNamespace(get_npu_format=lambda _: 2)
        raise AssertionError(name)

    monkeypatch.setattr(binding, "import_module", Mock(side_effect=importer))
    monkeypatch.setattr(binding.importlib.metadata, "version", versions.__getitem__)
    return SimpleNamespace(
        module=module,
        operation=operation,
        describe=describe,
        info=info,
        receipt=receipt,
        receipt_path=receipt_path,
        extension=extension,
        custom_ops=custom_ops,
        provider=provider,
    )


def test_compatible_addon_is_loaded_once_before_attention(binding: Any, addon: Any) -> None:
    assert binding.x_attention_out_available()
    output = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert binding.get_x_attention_out(output) is addon.operation
    assert binding.x_attention_out_available()
    addon.describe.assert_called_once()
    addon.operation.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "missing_receipt",
        "invalid_json",
        "incomplete_receipt",
        "failed_build",
        "schema",
        "abi",
        "python",
        "sources",
        "runtime",
        "extension_changed",
        "custom_ops_changed",
        "provider_changed",
        "provider_path",
        "provider_symbol",
        "metadata_missing",
        "metadata_abi",
        "metadata_sources",
        "metadata_failure",
        "operation_missing",
    ],
)
def test_unrecognized_or_stale_addon_falls_back_before_write(
    binding: Any, addon: Any, tmp_path: Path, change: str
) -> None:
    receipt = addon.receipt
    if change == "missing_receipt":
        addon.receipt_path.unlink()
    elif change == "invalid_json":
        addon.receipt_path.write_text("{")
    elif change == "incomplete_receipt":
        receipt.pop("providers")
    elif change == "failed_build":
        receipt["status"] = "failed"
    elif change == "schema":
        receipt["schema_version"] = 2
    elif change == "abi":
        receipt["abi_version"] = 2
    elif change == "python":
        receipt["python_abi"] = [0, 0]
    elif change == "sources":
        receipt["source_pins"] = {}
    elif change == "runtime":
        receipt["runtime"] = {}
    elif change.endswith("_changed"):
        getattr(addon, change.removesuffix("_changed")).write_bytes(b"changed")
    elif change == "provider_path":
        other = tmp_path / "other-provider.so"
        other.write_bytes(addon.provider.read_bytes())
        addon.info["providers"]["aclnnXAttention"] = str(other)
    elif change == "provider_symbol":
        addon.info["providers"].pop("aclnnXAttentionGetWorkspaceSize")
    elif change == "metadata_missing":
        delattr(addon.module, "binding_info")
    elif change == "metadata_abi":
        addon.info["abi_version"] = 2
    elif change == "metadata_sources":
        addon.info["source_pins"] = {}
    elif change == "metadata_failure":
        addon.describe.side_effect = RuntimeError("cannot resolve operator")
    elif change == "operation_missing":
        delattr(addon.module, "x_attention_out")
    else:
        raise AssertionError(change)
    if change not in ("missing_receipt", "invalid_json"):
        addon.receipt_path.write_text(json.dumps(receipt))
    assert not binding.x_attention_out_available()
    assert binding._load_binding() is None
    addon.operation.assert_not_called()


@pytest.mark.parametrize("error", [ImportError("missing"), OSError("binary ABI mismatch")])
def test_missing_addon_is_cached_without_build_or_retry(
    binding: Any, monkeypatch: Any, error: Exception
) -> None:
    importer = Mock(side_effect=error)
    monkeypatch.setattr(binding, "import_module", importer)
    assert not binding.x_attention_out_available()
    assert not binding.x_attention_out_available()
    importer.assert_called_once_with("vllm_gr_xattention_out_ext")


@pytest.mark.parametrize("distribution", ["torch", "torch-npu", "vllm", "vllm-ascend"])
def test_changed_runtime_distribution_declines_before_write(
    binding: Any, addon: Any, monkeypatch: Any, distribution: str
) -> None:
    recorded = addon.receipt["runtime"]
    monkeypatch.setattr(
        binding.importlib.metadata,
        "version",
        lambda name: "changed" if name == distribution else recorded[name],
    )
    assert not binding.x_attention_out_available()
    addon.operation.assert_not_called()


def test_cuda_output_declines_without_importing_vendor(binding: Any, monkeypatch: Any) -> None:
    importer = Mock(side_effect=AssertionError("must not import vendor runtime"))
    monkeypatch.setattr(binding, "import_module", importer)
    output = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    assert binding.get_x_attention_out(output) is None
    importer.assert_not_called()


def test_non_nd_output_declines_before_loading_addon(binding: Any, monkeypatch: Any) -> None:
    importer = Mock(return_value=SimpleNamespace(get_npu_format=lambda _: 29))
    monkeypatch.setattr(binding, "import_module", importer)
    output = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert binding.get_x_attention_out(output) is None
    importer.assert_called_once_with("torch_npu")


def test_build_plan_needs_no_site_packages_and_creates_nothing(tmp_path: Path) -> None:
    output = tmp_path / "new-build"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(BUILD),
            "--xllm-ops-dir",
            str(tmp_path / "source"),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "planned" and not report["execute"]
    assert not report["installs_packages"] and not report["kernel_rebuilt"]
    assert not output.exists()


@pytest.mark.parametrize("exists", [False, True])
def test_build_rejects_missing_or_unrecognized_vendor_source(tmp_path: Path, exists: bool) -> None:
    build = load_source(BUILD, "_out_build_source")
    if exists:
        source = tmp_path / "test/python_test"
        source.mkdir(parents=True)
        (source / "RegisterOps.cpp").write_text("different signature")
    with pytest.raises(ValueError, match="Unsupported xllm-ops source"):
        build.validate_sources(tmp_path)


def test_failed_build_retains_receipt_and_logs(tmp_path: Path, monkeypatch: Any) -> None:
    build = load_source(BUILD, "_out_build_failure")
    output = tmp_path / "failed-build"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(BUILD), "--execute", "--xllm-ops-dir", str(tmp_path), "--output-dir", str(output)],
    )
    monkeypatch.setattr(build, "validate_sources", lambda _: {})
    monkeypatch.setattr(build, "runtime_versions", lambda: {})
    monkeypatch.setattr(
        build.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 1, "compiler stdout", "compiler stderr")),
    )
    with pytest.raises(RuntimeError, match="Binding build failed"):
        build.main()
    receipt = json.loads((output / "build-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["returncode"] == 1
    assert (output / "build.stdout.log").read_text() == "compiler stdout"
    assert (output / "build.stderr.log").read_text() == "compiler stderr"
    assert not (output / "lib/vllm_gr_xattention_out_ext.receipt.json").exists()


def test_existing_build_directory_is_not_modified(tmp_path: Path, monkeypatch: Any) -> None:
    build = load_source(BUILD, "_out_build_existing")
    marker = tmp_path / "previous-result"
    marker.write_text("keep")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(BUILD), "--execute", "--xllm-ops-dir", str(tmp_path), "--output-dir", str(tmp_path)],
    )
    with pytest.raises(ValueError, match="fresh output directory"):
        build.main()
    assert marker.read_text() == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["previous-result"]
