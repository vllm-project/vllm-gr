# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Explicit, isolated build of the optional XAttention output binding."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

MODULE_NAME = "vllm_gr_xattention_out_ext"
RECEIPT_NAME = f"{MODULE_NAME}.receipt.json"
RUNTIME_BASE_VERSIONS = {"torch": "2.10.0", "vllm": "0.30.0"}
RUNTIME_EXACT_VERSIONS = {"torch-npu": "2.10.0.post7", "vllm-ascend": "0.30.0rc2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(name)
        for name in (*RUNTIME_BASE_VERSIONS, *RUNTIME_EXACT_VERSIONS)
    }
    for name, expected in RUNTIME_BASE_VERSIONS.items():
        if versions[name].split("+", 1)[0] != expected:
            raise ValueError(f"Unsupported {name}: {versions[name]}; expected {expected}")
    for name, expected in RUNTIME_EXACT_VERSIONS.items():
        if versions[name] != expected:
            raise ValueError(f"Unsupported {name}: {versions[name]}; expected {expected}")
    return versions


def file_identity(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Compile; otherwise only print a plan"
    )
    parser.add_argument("--xllm-ops-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source_dir = args.xllm_ops_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; previous build artifacts must be retained")
    report: dict[str, Any] = {
        "status": "planned",
        "execute": args.execute,
        "source_dir": str(source_dir),
        "output_dir": str(output),
        "installs_packages": False,
        "kernel_rebuilt": False,
        "command": [
            sys.executable,
            str(here / "setup.py"),
            "build_ext",
            "--build-temp",
            str(output / "temp"),
            "--build-lib",
            str(output / "lib"),
        ],
    }
    if not args.execute:
        print(json.dumps(report, indent=2))
        return

    output.mkdir(parents=True, exist_ok=False)
    try:
        report["runtime"] = runtime_versions()
        report["python_abi"] = list(sys.version_info[:2])
        report["build_sources"] = {
            name: sha256(here / name) for name in ("build.py", "setup.py", "xattention_out.cpp")
        }
        environment = os.environ.copy()
        environment["VLLM_GR_XATTENTION_SOURCE_DIR"] = str(source_dir)
        completed = subprocess.run(
            report["command"], cwd=here, env=environment, capture_output=True, text=True
        )
        (output / "build.stdout.log").write_text(completed.stdout)
        (output / "build.stderr.log").write_text(completed.stderr)
        report["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError("Binding build failed; see build.stdout.log and build.stderr.log")
        binaries = list((output / "lib").glob(f"{MODULE_NAME}*.so"))
        if len(binaries) != 1:
            raise RuntimeError("The build must produce exactly one extension")
        # Import only after an explicit build. Resolving symbols does not run attention.
        importlib.import_module("torch_npu")
        custom_ops = importlib.import_module("custom_ops_lib")
        if custom_ops.__file__ is None:
            raise RuntimeError("Cannot identify the installed custom_ops_lib binary")
        sys.path.insert(0, str(output / "lib"))
        addon = importlib.import_module(MODULE_NAME)
        info = addon.binding_info()
        if info["abi_version"] != 1:
            raise RuntimeError("The built addon has an unsupported binding contract")
        report["addon"] = file_identity(binaries[0])
        report["custom_ops"] = file_identity(custom_ops.__file__)
        report["providers"] = {
            name: file_identity(path) for name, path in info["providers"].items()
        }
        if set(report["providers"]) != {"aclnnXAttention", "aclnnXAttentionGetWorkspaceSize"}:
            raise RuntimeError("Both scale-enabled XAttention API symbols must resolve")
        report["schema_version"] = 1
        report["abi_version"] = 1
        report["status"] = "built"
        (output / "lib" / RECEIPT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "build-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
