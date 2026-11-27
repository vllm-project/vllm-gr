# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Rebuild and validate the MLU component wheels from a committed revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[3]
HOST_TESTS = (
    "tests/constraints/test_loader.py",
    "tests/constraints/test_mlu_constraint.py",
    "tests/constraints/test_scoring.py",
    "tests/platforms/test_mlu_provider.py",
    "tests/platforms/test_plugin_loader.py",
    "tests/platforms/test_registry.py",
    "tests/platforms/test_provider_compat.py",
    "tests/config/test_schema.py",
    "tests/config/test_validate_config.py",
    "tests/packaging/test_install_metadata.py",
)

SMOKE = """\
import sys
from importlib.metadata import requires, version
from pathlib import Path
from types import SimpleNamespace

import vllm_gr
from packaging.requirements import Requirement
from vllm_gr.platforms import platform_identity, select_backend
from vllm_gr.platforms.loader import load_backend_plugins

installed = Path(sys.argv[1])
load_backend_plugins()
entry = select_backend(platform_identity(SimpleNamespace(device_type='mlu')))
assert Path(vllm_gr.__file__).resolve().is_relative_to(installed)
assert Path(sys.modules[type(entry.provider).__module__].__file__).resolve().is_relative_to(installed)
assert entry.backend_id == entry.provider.device_type == 'mlu'
assert vllm_gr.__version__ == version('vllm-gr') and vllm_gr.__version__.endswith('mlu')
assert not (requires('vllm-gr-mlu') or [])
actual = {Requirement(r) for r in requires('vllm-gr') or [] if Requirement(r).marker is None}
expected = {Requirement(line) for raw in Path(sys.argv[2]).read_text().splitlines()
            if (line := raw.split('#', 1)[0].strip())}
assert actual == expected, (actual, expected)
assert not {'torch', 'torch_mlu', 'torch_npu', 'vllm', 'triton'}.intersection(sys.modules)
print(vllm_gr.__version__, entry.origin)
"""


def test_counts(path: Path) -> dict[str, int]:
    cases = tuple(ElementTree.parse(path).iter("testcase"))
    return {
        "tests": len(cases),
        **{
            name: sum(case.find(name) is not None for case in cases)
            for name in ("failure", "error", "skipped")
        },
    }


def installed_versions() -> dict[str, str | None]:
    names = ("torch", "torch-mlu", "byted-torch-mlu", "vllm", "vllm-mlu", "pytest", "setuptools")
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def validate(mode: str, commit: str, output: Path, receipt: dict[str, Any]) -> None:
    py = sys.executable
    checkout = output / "source"
    wheels = output / "wheels"
    installed = output / "installed"
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    for name in (
        "VLLM_GR_TARGET_DEVICE",
        "VLLM_GR_BACKEND",
        "VLLM_GR_DISABLE_BUILTIN_BACKENDS",
        "VLLM_GR_VERSION_OVERRIDE",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR",
        "PYTEST_ADDOPTS",
        "PYTHONOPTIMIZE",
    ):
        env.pop(name, None)

    def run(name: str, *command: str, cwd: Path = output, **overrides: str) -> None:
        log = output / f"{name}.log"
        print(f"{name}: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=cwd,
                env={**env, **overrides},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        receipt["steps"].append({"name": name, "returncode": result.returncode, "command": command})
        result.check_returncode()

    run(
        "clone",
        "git",
        "clone",
        "--quiet",
        "--no-checkout",
        "--no-hardlinks",
        str(ROOT),
        str(checkout),
    )
    run("checkout", "git", "switch", "--quiet", "--detach", commit, cwd=checkout)
    if mode == "host":
        run(
            "host-tests",
            py,
            "-m",
            "pytest",
            "-q",
            "--strict-test-levels",
            "-m",
            "cpu_test",
            "--junitxml",
            str(output / "host.xml"),
            *HOST_TESTS,
            cwd=checkout,
        )
        receipt["tests"] = test_counts(output / "host.xml")
        if receipt["tests"]["tests"] <= receipt["tests"]["skipped"]:
            raise RuntimeError("host gate did not execute a test")

    build = (py, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(wheels))
    run("core-wheel", *build, ".", cwd=checkout, VLLM_GR_TARGET_DEVICE="mlu")
    run("plugin-wheel", *build, "plugins/mlu", cwd=checkout)
    artifacts = sorted(wheels.glob("*.whl"))
    if len(artifacts) != 2:
        raise RuntimeError(f"expected two wheels, found {artifacts}")
    receipt["wheels"] = {
        wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest() for wheel in artifacts
    }
    run(
        "install",
        py,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "--no-index",
        "--no-deps",
        "--target",
        str(installed),
        *(str(wheel) for wheel in artifacts),
    )
    runtime_path = os.pathsep.join([str(installed), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    run(
        "installed-smoke",
        py,
        "-c",
        SMOKE,
        str(installed),
        str(checkout / "requirements/common.txt"),
        PYTHONPATH=runtime_path,
    )

    if mode == "device":
        run("cnmon", "cnmon", "info")
        report = output / "device.xml"
        device_code = """\
import sys
from pathlib import Path
import pytest
import vllm_gr
import vllm_gr_mlu
assert all(Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]))
           for module in (vllm_gr, vllm_gr_mlu))
raise SystemExit(pytest.main(sys.argv[2:]))
"""
        run(
            "device-tests",
            py,
            "-c",
            device_code,
            str(installed),
            "-q",
            "--import-mode=importlib",
            "--junitxml",
            str(report),
            str(checkout / "plugins/mlu/tests"),
            PYTHONPATH=os.pathsep.join([runtime_path, str(checkout)]),
        )
        counts = receipt["tests"] = test_counts(report)
        if not counts["tests"] or counts["skipped"] or counts["failure"] or counts["error"]:
            raise RuntimeError(f"device gate requires every test to run and pass: {counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("host", "device"))
    parser.add_argument("--ref", default="HEAD", help="Git revision to validate")
    parser.add_argument("--output", type=Path, required=True, help="New receipt directory")
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{args.ref}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    output = args.output.resolve()
    output.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "commit": commit,
        "mode": args.mode,
        "platform": platform.platform(),
        "python": sys.version,
        "dependencies": installed_versions(),
        "steps": [],
        "status": "FAIL",
    }
    try:
        validate(args.mode, commit, output, receipt)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        receipt["error"] = str(error)
    else:
        receipt["status"] = "PASS"
    finally:
        path = output / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"{receipt['status']}: {path}")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
