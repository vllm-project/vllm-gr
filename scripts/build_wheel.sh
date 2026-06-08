#!/bin/bash
# Script to build wheel for vllm-gr
# Usage: ./build_wheel.sh [--python PYTHON] [--skip-clean]

set -e

PYTHON="python3"
SKIP_CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --skip-clean)
            SKIP_CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--python PYTHON] [--skip-clean]"
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
    rm -rf dist/ build/ vllm-gr.egg-info vllm_gr.egg-info
fi

export VLLM_GR_TARGET_DEVICE="${VLLM_GR_TARGET_DEVICE:-cpu}"

"$PYTHON" -m pip install --quiet build wheel setuptools-scm

echo "Building wheel with VLLM_GR_TARGET_DEVICE=${VLLM_GR_TARGET_DEVICE}"
"$PYTHON" -m build --wheel

echo "Wheel built successfully!"
ls -lh dist/*.whl

if command -v unzip >/dev/null 2>&1; then
    echo "Requires-Dist from wheel metadata:"
    unzip -p dist/*.whl "*/METADATA" | grep -E '^Requires-Dist:' || true
fi
