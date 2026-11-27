# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compare an sdist-built wheel with a wheel built from an export of Git HEAD."""

import os
import subprocess
import sys
import tarfile
import tempfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.utils import parse_sdist_filename, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = Path("vllm_gr/entrypoints/grpc/proto")


def package_files(wheel: Path) -> set[str]:
    with ZipFile(wheel) as archive:
        return {
            name
            for name in archive.namelist()
            if name.startswith("vllm_gr/") and not name.endswith("/")
        }


def main() -> None:
    if os.environ.get("VLLM_GR_TARGET_DEVICE") not in {"cpu", "cuda", "mlu", "npu"}:
        raise SystemExit("Set VLLM_GR_TARGET_DEVICE to cpu, cuda, mlu or npu for both builds.")
    (wheel,) = (ROOT / "dist").glob("*.whl")
    (sdist,) = (ROOT / "dist").glob("*.tar.gz")
    name, version, _, _ = parse_wheel_filename(wheel.name)
    if parse_sdist_filename(sdist.name) != (name, version):
        raise SystemExit(f"Distribution names or versions differ: {sdist.name}, {wheel.name}")
    with tarfile.open(sdist) as source:
        member = source.extractfile(f"{sdist.name.removesuffix('.tar.gz')}/PKG-INFO")
        if member is None or BytesParser().parsebytes(member.read())["Version"] != str(version):
            raise SystemExit(f"Unexpected PKG-INFO version in {sdist.name}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        export = work / "source"
        export.mkdir()
        tar = work / "source.tar"
        subprocess.run(["git", "archive", "--output", str(tar), "HEAD"], cwd=ROOT, check=True)
        with tarfile.open(tar) as archive:
            archive.extractall(export, filter="data")
        # The export has no Git metadata. Compare the same version and device.
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel"],
            cwd=export,
            check=True,
            env={**os.environ, "VLLM_GR_VERSION_OVERRIDE": str(version)},
        )
        archive_wheel = export / "dist" / wheel.name
        from_sdist = package_files(wheel)
        from_git = package_files(archive_wheel)
        if from_sdist != from_git:
            raise SystemExit(
                f"Only in sdist-built wheel: {sorted(from_sdist - from_git)}\n"
                f"Only in archive-built wheel: {sorted(from_git - from_sdist)}"
            )
        installed = work / "installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(installed),
                str(archive_wheel),
            ],
            cwd=work,
            check=True,
        )
        for proto in ("common.proto", "vllm_engine.proto"):
            relative = PROTO_DIR / proto
            if (installed / relative).read_bytes() != (export / relative).read_bytes():
                raise SystemExit(f"Installed protocol file differs from the source: {relative}")
    print(f"Verified {len(from_sdist)} package files and installed gRPC protocols in {wheel.name}")


if __name__ == "__main__":
    main()
