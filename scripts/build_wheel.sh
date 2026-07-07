#!/bin/bash
# Script to build wheel for vllm-gr
# Usage: ./build_wheel.sh [--python PYTHON] [--skip-clean]

set -euo pipefail

PYTHON="python3"
SKIP_CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            [[ $# -ge 2 ]] || { echo "Error: --python requires a value" >&2; exit 1; }
            PYTHON="$2"
            shift 2
            ;;
        --skip-clean)
            SKIP_CLEAN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--python PYTHON] [--skip-clean]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--python PYTHON] [--skip-clean]" >&2
            exit 1
            ;;
    esac
done

echo "Building wheel with Python: $PYTHON"

cd "$(dirname "$0")/.."

if [[ "${SKIP_CLEAN}" == "true" ]]; then
    echo "Skipping cleanup as requested"
else
    echo "Cleaning previous build artifacts"
    # setuptools may produce either vllm-gr.egg-info or vllm_gr.egg-info
    # depending on version; clean both.
    rm -rf dist/ build/ vllm-gr.egg-info vllm_gr.egg-info
fi

"$PYTHON" -m pip install --quiet build wheel setuptools-scm

echo "Building wheel with VLLM_GR_TARGET_DEVICE=${VLLM_GR_TARGET_DEVICE:-<auto-detected by setup.py>}"
"$PYTHON" -m build --wheel

echo "Wheel built successfully!"
ls -lh dist/*.whl

if command -v unzip >/dev/null 2>&1; then
    echo "Requires-Dist from wheel metadata:"
    for whl in dist/*.whl; do
        if [[ -f "$whl" ]]; then
            metadata=$(unzip -p "$whl" "*/METADATA") || {
                echo "ERROR: failed to read METADATA from wheel $whl" >&2
                exit 1
            }
            echo "$metadata" | grep -E '^Requires-Dist:' || echo "(no Requires-Dist found in $whl)"
        fi
    done
fi
