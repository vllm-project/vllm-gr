#!/bin/bash
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
    PYTHONPATH="$INSTALL_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -c 'import vllm_gr; print(vllm_gr.__version__)'
)

echo "Running packaging metadata tests"
"$PYTHON" -m pytest tests/packaging/test_install_metadata.py -q

echo "Wheel smoke test completed successfully."
