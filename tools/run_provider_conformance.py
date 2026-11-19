# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise two separately installed wheels outside the source checkout."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/platforms/fixtures/provider"
BOOTSTRAP = """
import sys
from importlib.metadata import distribution
from pathlib import Path
core, provider, checkout = map(Path, sys.argv[1:4])
sys.path[:0] = [str(core), str(provider)]
assert not any(Path(p).resolve().is_relative_to(checkout) for p in sys.path)
import vllm_gr, gr_example
assert Path(vllm_gr.__file__).resolve().is_relative_to(core)
assert Path(gr_example.__file__).resolve().is_relative_to(provider)
bundled = distribution("vllm-gr").entry_points.select(group="vllm_gr.platform_backends")
assert {"cuda", "ascend"} <= {entry.name for entry in bundled}, bundled
from gr_example.probe import run
run(*sys.argv[4:])
"""


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(core_wheel: Path, expected: str, *, native_platform: bool = False) -> None:
    if digest(core_wheel) != expected:
        raise ValueError("core wheel SHA-256 differs from the supplied digest")
    with TemporaryDirectory(prefix="gr-provider-") as name:
        root = Path(name).resolve()
        if root.is_relative_to(ROOT):
            raise ValueError("set TMPDIR or TEMP to a directory outside the source checkout")
        frozen = root / core_wheel.name
        shutil.copyfile(core_wheel, frozen)
        if digest(frozen) != expected:
            raise ValueError("core wheel changed while freezing the artifact")
        core, provider = root / "core", root / "provider"
        pip = [sys.executable, "-m", "pip"]
        install = ["install", "--no-index", "--no-deps", "--no-compile", "--target"]
        subprocess.run([*pip, *install, str(core), str(frozen)], cwd=root, check=True)
        core_files = {p.relative_to(core): digest(p) for p in core.rglob("*") if p.is_file()}
        # Build from a copy so the checkout never receives provider build artifacts.
        project = shutil.copytree(FIXTURE, root / "project")
        subprocess.run(
            [
                *pip,
                "wheel",
                "--no-deps",
                "--no-index",
                "--no-build-isolation",
                "--wheel-dir",
                str(root),
                str(project),
            ],
            cwd=root,
            check=True,
        )
        (plugin,) = root.glob("vllm_gr_conformance_provider-*.whl")
        subprocess.run([*pip, *install, str(provider), str(plugin)], cwd=root, check=True)
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("VLLM_GR_", "GR_CONFORMANCE_"))
        }
        env.update(VLLM_PLUGINS="gr_conformance", VLLM_GR_STRICT_PROVIDER_RANGES="1")
        # A parent build can target CPU while this host fixture provides an OOT platform.
        env.pop("VLLM_TARGET_DEVICE", None)
        scenarios = [
            "engine",
            "worker-0",
            "worker-1",
            "agreement",
            "mismatch",
            "incomplete",
            "unsupported",
            "ambiguous",
            "explicit",
            "version",
            "patch",
        ]
        if native_platform:
            scenarios.append("bootstrap")
        for scenario in scenarios:
            if scenario == "bootstrap":
                env["VLLM_PLUGINS"] = "gr_conformance,vllm_gr"
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    BOOTSTRAP,
                    str(core),
                    str(provider),
                    str(ROOT),
                    scenario,
                    "native" if native_platform else "fixture",
                ],
                cwd=root,
                env=env,
                check=True,
                timeout=120,
            )
        if digest(frozen) != expected or digest(core_wheel) != expected:
            raise RuntimeError("core wheel changed during external provider conformance")
        installed = {p.relative_to(core): digest(p) for p in core.rglob("*") if p.is_file()}
        if installed != core_files:
            raise RuntimeError("the installed core changed during external provider conformance")
        print(
            f"Provider conformance passed. Core SHA-256 {expected}. Provider SHA-256 {digest(plugin)}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core_wheel", type=Path)
    parser.add_argument(
        "--sha256", required=True, help="expected digest of the prebuilt core wheel"
    )
    parser.add_argument(
        "--native-platform",
        action="store_true",
        help="obtain the host fixture identity through vLLM's platform entry point",
    )
    args = parser.parse_args()
    run(args.core_wheel.resolve(), args.sha256.lower(), native_platform=args.native_platform)
