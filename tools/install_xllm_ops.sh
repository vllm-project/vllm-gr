#!/usr/bin/env bash

# Ensure the xllm-ops custom operator library is available on an Ascend NPU
# server. If it is missing, clone the upstream repository, initialize its
# submodules, source the Ascend Toolkit environment, build the operators, and
# run the Python operator checks. Custom build overrides remain supported.

set -Eeuo pipefail

log() {
    printf '[xllm-ops] %s\n' "$*"
}

die() {
    printf '[xllm-ops] ERROR: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
XLLM_OPS_REPOSITORY="${XLLM_OPS_REPOSITORY:-https://github.com/xLLM-AI/xllm-ops.git}"
XLLM_OPS_DIR="${XLLM_OPS_DIR:-${REPO_ROOT}/third_party/xllm-ops}"

# Set XLLM_OPS_SKIP_NPU_CHECK=1 when the script is invoked by a scheduler that
# has already selected an Ascend device but does not expose npu-smi in PATH.
is_npu_server() {
    if [[ "${XLLM_OPS_SKIP_NPU_CHECK:-0}" == "1" ]]; then
        return 0
    fi

    if command -v npu-smi >/dev/null 2>&1; then
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 - <<'PY' >/dev/null 2>&1
try:
    import torch
    import torch_npu  # noqa: F401
    raise SystemExit(0 if hasattr(torch, "npu") and torch.npu.is_available() else 1)
except Exception:
    raise SystemExit(1)
PY
        return $?
    fi

    return 1
}

probe_xllm_ops() {
    command -v python3 >/dev/null 2>&1 || return 1

    python3 - <<'PY'
import ctypes.util
import importlib.util
import os
import subprocess
import sys

# A Python package/module is a valid installation signal.
module_present = any(
    importlib.util.find_spec(name) is not None
    for name in ("xllm_ops", "custom_ops_lib")
)

# The vendored wrapper loads the shared object through -lxllm_ops.  Check both
# the dynamic linker and common explicit library paths.
library_present = ctypes.util.find_library("xllm_ops") is not None
for root in filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":")):
    for filename in ("libxllm_ops.so", "xllm_ops.so"):
        if os.path.isfile(os.path.join(root, filename)):
            library_present = True

try:
    output = subprocess.run(
        ["ldconfig", "-p"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    library_present = library_present or "xllm_ops" in output
except (FileNotFoundError, OSError):
    pass

if not (module_present or library_present):
    raise SystemExit(1)

# If the project wrapper is importable, require at least one registered op.
# Import errors are treated as a failed probe so a broken library is rebuilt.
try:
    spec = importlib.util.find_spec("vllm_ascend.ops.xllm_ops")
    if spec is not None:
        module = __import__("vllm_ascend.ops.xllm_ops", fromlist=["*"])
        if hasattr(module, "x_attention"):
            raise SystemExit(0)
except Exception:
    raise SystemExit(1)

# A discoverable shared object/package is sufficient when the wrapper is not
# installed in this environment yet.
raise SystemExit(0)
PY
}

run_build() {
    if [[ -n "${XLLM_OPS_BUILD_SCRIPT:-}" ]]; then
        [[ -f "${XLLM_OPS_BUILD_SCRIPT}" ]] || die "build script not found: ${XLLM_OPS_BUILD_SCRIPT}"
        log "Running build script: ${XLLM_OPS_BUILD_SCRIPT}"
        bash "${XLLM_OPS_BUILD_SCRIPT}"
        return
    fi

    command -v git >/dev/null 2>&1 || die "git is required to clone xllm-ops"

    if [[ -e "${XLLM_OPS_DIR}" && ! -d "${XLLM_OPS_DIR}" ]]; then
        die "xllm-ops target path exists but is not a directory: ${XLLM_OPS_DIR}"
    fi

    if [[ ! -d "${XLLM_OPS_DIR}/.git" ]]; then
        if [[ -e "${XLLM_OPS_DIR}" && -n "$(find "${XLLM_OPS_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            die "xllm-ops directory is not an empty Git checkout: ${XLLM_OPS_DIR}"
        fi
        mkdir -p "$(dirname "${XLLM_OPS_DIR}")"
        log "Cloning ${XLLM_OPS_REPOSITORY} into ${XLLM_OPS_DIR}"
        git clone "${XLLM_OPS_REPOSITORY}" "${XLLM_OPS_DIR}"
    else
        log "Using existing xllm-ops checkout: ${XLLM_OPS_DIR}"
    fi

    log "Initializing xllm-ops submodules"
    (cd "${XLLM_OPS_DIR}" && git submodule init && git submodule update)

    local ascend_toolkit_env="${ASCEND_TOOLKIT_ENV:-}"
    if [[ -z "${ascend_toolkit_env}" ]]; then
        [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]] || die \
            "ASCEND_TOOLKIT_HOME is not set; cannot locate Ascend Toolkit set_env.sh"
        ascend_toolkit_env="${ASCEND_TOOLKIT_HOME}/set_env.sh"
    fi
    [[ -f "${ascend_toolkit_env}" ]] || die \
        "Ascend Toolkit environment script not found: ${ascend_toolkit_env}"
    [[ -f "${XLLM_OPS_DIR}/build.sh" ]] || die \
        "xllm-ops build script not found: ${XLLM_OPS_DIR}/build.sh"
    [[ -f "${XLLM_OPS_DIR}/test/python_test/build_and_run.sh" ]] || die \
        "xllm-ops test script not found: ${XLLM_OPS_DIR}/test/python_test/build_and_run.sh"

    log "Sourcing Ascend Toolkit environment: ${ascend_toolkit_env}"
    log "Building xllm-ops"
    log "Running xllm-ops Python operator tests"
    (
        cd "${XLLM_OPS_DIR}"
        # shellcheck disable=SC1090
        source "${ascend_toolkit_env}"
        bash build.sh
        cd test/python_test
        bash build_and_run.sh
    )
}

is_npu_server || {
    log "No Ascend NPU detected; skipping xllm-ops installation."
    exit 0
}

if probe_xllm_ops; then
    log "xllm-ops is already installed; nothing to do."
    exit 0
fi

log "xllm-ops was not found on this NPU server."
run_build

probe_xllm_ops || die \
    "the build command completed, but xllm-ops could not be detected afterwards"

log "xllm-ops installation verified successfully."
