#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve candidate runtime declarations before creating the CI environment.

Only the standard library is used: resolving versions must not import setup.py,
GR, torch, or vLLM from a previously cached environment. NPU runners must supply
the vendor runtime in the selected base Python; this tool does not install it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GUARDS = {
    "vllm": "_SUPPORTED_VLLM_VERSION",
    "vllm-ascend": "_SUPPORTED_VLLM_ASCEND_VERSION",
    "triton-ascend": "_SUPPORTED_TRITON_ASCEND_VERSION",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def literal(path: Path, name: str) -> str:
    values = [
        node.value
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(values) != 1 or not isinstance(values[0], ast.Constant):
        raise ValueError(f"{path}: expected one literal assignment to {name}")
    value = values[0].value
    if not isinstance(value, str):
        raise TypeError(f"{path}: {name} must be a version string")
    release, separator, rc = value.partition("rc")
    parts = release.split(".")
    if (
        len(parts) != 3
        or not all(part.isascii() and part.isdecimal() for part in parts)
        or (separator and not (rc.isascii() and rc.isdecimal()))
    ):
        raise ValueError(f"{path}: {name} must be a release or rc version literal")
    return value


def declarations(repo: Path, device: str) -> dict[str, str]:
    packages = tuple(GUARDS) if device == "npu" else ("vllm",)
    guards = {
        package: literal(repo / "vllm_gr/_env_check.py", GUARDS[package]) for package in packages
    }
    base = literal(repo / "setup.py", "VLLM_GR_BASE_VERSION")
    config = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    cuda_lines = [
        line.split("#", 1)[0].strip()
        for line in (repo / "requirements/cuda.txt").read_text(encoding="utf-8").splitlines()
    ]
    pins = [line for line in cuda_lines if line.lower().startswith("vllm")]
    if (
        base != guards["vllm"]
        or config["tool"]["setuptools_scm"]["fallback_version"] != base
        or pins != [f"vllm=={base}"]
    ):
        raise ValueError("vLLM guard, CUDA pin, GR base and SCM fallback versions must agree")
    expected = {"vllm": base, "vllm-gr": base}
    if device == "npu":
        expected.update({key: guards[key] for key in ("vllm-ascend", "triton-ascend")})
    return expected


def source_digest(repo: Path) -> str:
    names = (
        "setup.py",
        "pyproject.toml",
        "vllm_gr/_env_check.py",
        "tools/ci/resolve_runtime.py",
        ".pre-commit-config.yaml",
    )
    paths = {
        name: (
            Path(__file__).resolve()
            if name == "tools/ci/resolve_runtime.py" and not (repo / name).is_file()
            else repo / name
        )
        for name in names
    }
    paths.update(
        {
            str(path.relative_to(repo)): path
            for path in (repo / "requirements").rglob("*")
            if path.is_file()
        }
    )
    return digest(
        {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    )


def inventory() -> dict[str, str]:
    return {
        str(dist.metadata["Name"]).lower().replace("_", "-"): metadata.version(
            dist.metadata["Name"]
        )
        for dist in metadata.distributions()
    }


def check_versions(expected: dict[str, str], installed: dict[str, str]) -> list[str]:
    return [
        f"{name}: expected {version}, found {installed.get(name, 'not installed')}"
        for name, version in expected.items()
        if installed.get(name, "").partition("+")[0] != version
    ]


def resolve(repo: Path, device: str) -> dict[str, Any]:
    device = "cuda" if device == "gpu" else device
    spec: dict[str, Any] = {
        "schema": 1,
        "device": device,
        "expected": declarations(repo, device),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "dependency_sha256": source_digest(repo),
    }
    if device == "npu":
        # Inherited packages are part of the cache identity, not an implicit
        # dependency on whichever vendor image happens to host a later job.
        spec["base_packages"] = inventory()
        required = {
            name: version for name, version in spec["expected"].items() if name != "vllm-gr"
        }
        errors = check_versions(required, spec["base_packages"])
        if "torch-npu" not in spec["base_packages"]:
            errors.append("the NPU base Python must provide torch-npu")
        if errors:
            raise ValueError("NPU runner image is not ready: " + "; ".join(errors))
    spec["cache_key"] = (
        f"gr-runtime-v1-{device}-{spec['os']}-{spec['architecture']}"
        f"-py{spec['python']}-vllm{spec['expected']['vllm']}-{digest(spec)}"
    )
    return spec


def check_origin(module: str, origin: Path, repo: Path, prefixes: list[Path]) -> str | None:
    if module == "vllm_gr":
        if origin != repo / "vllm_gr/__init__.py":
            return f"vllm_gr must import from {repo}, found {origin}"
    elif not any(origin.is_relative_to(prefix) for prefix in prefixes):
        return f"{module} imports outside the runtime prefixes: {origin}"
    return None


def verify(repo: Path, spec: dict[str, Any], venv: Path) -> dict[str, Any]:
    installed = inventory()
    report: dict[str, Any] = {
        "spec": spec,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "repo": str(repo),
        "packages": installed,
        "modules": {},
        "module_versions": {},
        "errors": [],
    }
    errors: list[str] = report["errors"]
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    report["revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    if Path(sys.prefix).resolve() != venv:
        errors.append(f"expected Python prefix {venv}, found {sys.prefix}")
    for key, value in {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "architecture": platform.machine(),
    }.items():
        if value != spec[key]:
            errors.append(f"{key} differs from the environment cache identity")
    if source_digest(repo) != spec["dependency_sha256"]:
        errors.append("runtime declarations changed after cache resolution")
    errors.extend(check_versions(spec["expected"], installed))
    report["distributions"] = {
        name: str(metadata.distribution(name).locate_file(""))
        for name in (*spec["expected"], "torch", "triton", "torch-npu")
        if name in installed
    }
    modules = ["vllm_gr", "vllm", "torch", "triton"]
    prefixes = [venv]
    if spec["device"] == "npu":
        prefixes.append(Path(sys.base_prefix).resolve())
        modules.extend(["vllm_ascend", "torch_npu"])
    # Fail before importing an unsupported runtime, while still saving versions.
    if not errors:
        for name in modules:
            try:
                module = importlib.import_module(name)
                origin = Path(module.__file__ or "").resolve()
                report["modules"][name] = str(origin)
                module_version = str(getattr(module, "__version__", "unknown"))
                report["module_versions"][name] = module_version
                if name in ("vllm", "vllm_gr"):
                    errors.extend(
                        check_versions(
                            {name: spec["expected"][name.replace("_", "-")]},
                            {name: module_version},
                        )
                    )
                error = check_origin(name, origin, repo, prefixes)
                if error:
                    errors.append(error)
            except Exception as exc:  # noqa: BLE001 - archive native runtime import failures
                errors.append(f"cannot import {name}: {exc}")
        if not errors:
            try:
                if spec["device"] == "npu":
                    guard = importlib.import_module("vllm_gr._env_check")
                    report["ascend"] = guard.check_ascend_environment(require_npu_platform=True)
                else:
                    platforms = importlib.import_module("vllm.platforms")
                    report["platform"] = str(platforms.current_platform.device_type)
                    if report["platform"] != "cuda":
                        errors.append(f"expected CUDA platform, found {report['platform']}")
                torch = importlib.import_module("torch")
                report["torch_cuda"] = torch.version.cuda
            except Exception as exc:  # noqa: BLE001 - archive vendor platform initialization failures
                errors.append(f"runtime platform check failed: {exc}")
    report["installed_sha256"] = digest(installed)
    report["ok"] = not errors
    return report


def append_values(path: Path | None, values: dict[str, str]) -> None:
    if path is not None:
        with path.open("a", encoding="utf-8") as stream:
            for key, value in values.items():
                if "\n" in value or "\r" in value:
                    raise ValueError(f"multiline value for {key}")
                stream.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("resolve", "verify"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("gpu", "cuda", "npu"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--github-env", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    report: dict[str, Any]
    try:
        if args.command == "resolve":
            if not args.device:
                raise ValueError("resolve requires --device")
            spec = resolve(args.repo.resolve(), args.device)
            report = {"ok": True, **spec}
            append_values(args.github_output, {"cache-key": spec["cache_key"]})
            append_values(
                args.github_env,
                {
                    "VLLM_GR_TARGET_DEVICE": spec["device"],
                    "CI_VLLM_REQUIREMENT": f"vllm=={spec['expected']['vllm']}",
                },
            )
        else:
            if not args.spec or not args.venv:
                raise ValueError("verify requires --spec and --venv")
            spec = json.loads(args.spec.read_text(encoding="utf-8"))
            report = verify(args.repo.resolve(), spec, args.venv.resolve())
            if report["ok"] and args.cache_root:
                cache = args.cache_root.resolve() / spec["cache_key"] / report["installed_sha256"]
                append_values(
                    args.github_env,
                    {
                        name: str(cache / subdir)
                        for name, subdir in {
                            "VLLM_CACHE_ROOT": "vllm",
                            "TORCHINDUCTOR_CACHE_DIR": "inductor",
                            "TRITON_CACHE_DIR": "triton",
                        }.items()
                    },
                )
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        report = {"ok": False, "errors": [str(exc)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["ok"]:
        print("; ".join(report["errors"]), file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
