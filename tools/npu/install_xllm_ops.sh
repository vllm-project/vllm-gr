#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


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
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
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

# shellcheck source=tools/npu/xllm_env.sh
source "${SCRIPT_DIR}/xllm_env.sh"

probe_xllm_ops() (
    command -v python3 >/dev/null 2>&1 || return 1
    if [[ -n ${ASCEND_TOOLKIT_ENV:-} ]]; then
        # shellcheck disable=SC1090
        source "${ASCEND_TOOLKIT_ENV}" || return 1
    fi
    load_xllm_env --if-installed || return 1
    python3 - <<'PY'
import sys

# Import Torch first, as the runtime does, to load the binding's dependencies.
import torch  # noqa: F401
import custom_ops_lib

required = ("cache_unshared_kv", "x_attention", "select_unshared_kv", "rec_constrained_topk")
missing = [name for name in required if not callable(getattr(custom_ops_lib, name, None))]
if missing:
    sys.exit(f"custom_ops_lib is missing callable operators: {', '.join(missing)}")
PY
)

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
        bash build.sh \
            -n 'beam_search_group;cache_unshared_kv;select_unshared_kv;x_attention;onerec_final_beam_select;rec_constrained_top_k' \
            --no-build-transformer --no-build-flash
        load_xllm_env
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
