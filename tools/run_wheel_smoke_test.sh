#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# Build a wheel, install it into an isolated target directory, and smoke-test
# the installed artifact without importing the source checkout.

set -euo pipefail

PYTHON="python3"

usage() {
    echo "Usage: $0 [--python PYTHON]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            [[ $# -ge 2 ]] || {
                echo "Error: --python requires a value" >&2
                exit 1
            }
            PYTHON="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE_ROOT="$(mktemp -d)"
INSTALL_DIR="$SMOKE_ROOT/site-packages"

cleanup() {
    rm -rf "$SMOKE_ROOT"
}
trap cleanup EXIT

cd "$ROOT_DIR"

echo "Building wheel for ${VLLM_GR_TARGET_DEVICE:-auto-detected device}"
bash scripts/build_wheel.sh --python "$PYTHON"

shopt -s nullglob
wheels=(dist/*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "Error: expected exactly one wheel in dist/, found ${#wheels[@]}" >&2
    exit 1
fi

echo "Installing wheel into isolated directory: ${wheels[0]}"
"$PYTHON" -m pip install --no-deps --target "$INSTALL_DIR" "${wheels[0]}"

echo "Importing installed wheel outside the source checkout"
(
    cd "$SMOKE_ROOT"
    export PYTHONPATH="$INSTALL_DIR"
    export INSTALL_DIR
    export SOURCE_DIR="$ROOT_DIR"
    export VLLM_GR_EXPECTED_VERSION="${VLLM_GR_EXPECTED_VERSION:-}"
    export VLLM_GR_TARGET_DEVICE="${VLLM_GR_TARGET_DEVICE:-}"
    "$PYTHON" - <<'PY'
import os
import tarfile
from email.parser import BytesParser, Parser
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

import vllm_gr

install_dir = Path(os.environ["INSTALL_DIR"]).resolve()
origin = Path(vllm_gr.__file__).resolve()
if not origin.is_relative_to(install_dir):
    raise SystemExit(f"vllm_gr imported outside wheel target: {origin}")

# Probe installed module locations without importing hardware implementations.
worker_package = "vllm_gr.v1.worker"
for module in (
    "model_runner_patch",
    "cuda.gpu_stage_hooks",
    "cuda.v1.gpu_model_runner_patch",
    "cuda.v2.gpu_model_runner_v2_patch",
    "cuda.v2.gpu_model_runner_v2_resources",
    "npu.npu_model_runner_patch",
):
    spec = find_spec(f"{worker_package}.{module}")
    if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(install_dir):
        raise SystemExit(f"worker module missing from installed wheel: {module}")

version = metadata.version("vllm-gr")
(sdist,) = (Path(os.environ["SOURCE_DIR"]) / "dist").glob("*.tar.gz")
with tarfile.open(sdist) as archive:
    info = archive.extractfile(f"{sdist.name.removesuffix('.tar.gz')}/PKG-INFO")
    if info is None or BytesParser().parsebytes(info.read())["Version"] != version:
        raise SystemExit(f"{sdist.name} PKG-INFO disagrees with installed version {version}")
expected = os.environ["VLLM_GR_EXPECTED_VERSION"]
if expected and version != expected:
    raise SystemExit(f"expected vllm-gr {expected}, found {version}")
if vllm_gr.__version__ != version:
    raise SystemExit(
        f"module/distribution version mismatch: {vllm_gr.__version__} != {version}"
    )

requirements = {
    Requirement(value).name.lower().replace("_", "-"): str(Requirement(value).specifier)
    for value in metadata.requires("vllm-gr") or []
    if Requirement(value).marker is None
}
target = os.environ["VLLM_GR_TARGET_DEVICE"]
overridden = any(os.environ.get(name) for name in (
    "VLLM_GR_VERSION_OVERRIDE",
    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR",
    "SETUPTOOLS_SCM_PRETEND_VERSION",
))
sdist_metadata = Path(os.environ["SOURCE_DIR"]) / "PKG-INFO"
if not overridden and sdist_metadata.is_file():
    source_version = Parser().parsestr(sdist_metadata.read_text(encoding="utf-8"))["Version"]
    if version != source_version:
        raise SystemExit(f"sdist/wheel version mismatch: {source_version} != {version}")
elif not overridden:
    local = Version(version).local
    suffix = local.split(".")[-1] if local else None
    if target in {"cpu", "npu"} and suffix != target:
        raise SystemExit(f"{target} wheel has wrong version suffix: {version}")
    if target == "cuda" and suffix in {"cpu", "npu"}:
        raise SystemExit(f"CUDA wheel has wrong version suffix: {version}")
if "regex" not in requirements:
    raise SystemExit("wheel is missing its direct regex dependency")
if target == "cuda" and requirements.get("vllm") != "==0.30.0":
    raise SystemExit(f"CUDA wheel has wrong vLLM requirement: {requirements.get('vllm')}")
if target in {"cpu", "npu"} and "vllm" in requirements:
    raise SystemExit(f"{target} wheel unexpectedly depends on vLLM")
print(f"vllm-gr {version} from {origin}")
PY
)

echo "Running packaging metadata tests"
"$PYTHON" -m pytest \
    --confcutdir=tests/packaging \
    tests/packaging/test_install_metadata.py \
    -q

core_sha256="$("$PYTHON" -c 'import hashlib, sys; print(hashlib.file_digest(open(sys.argv[1], "rb"), "sha256").hexdigest())' "${wheels[0]}")"
"$PYTHON" tools/run_provider_conformance.py "${wheels[0]}" --sha256 "$core_sha256"

echo "Wheel smoke test completed successfully."
