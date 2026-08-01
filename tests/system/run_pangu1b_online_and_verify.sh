#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${PANGU1B_MODEL_PATH:-${PANGU_MODEL_PATH:-}}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: PANGU1B_MODEL_PATH or PANGU_MODEL_PATH must point to openPangu-Embedded-1B-V1.1" >&2
  exit 2
fi

echo "Running Pangu 1B online serve ST via unittest"
echo "  model: $MODEL_PATH"
echo "  min_accuracy: ${PANGU1B_MIN_ACCURACY:-0.5}"
echo "  block_size: ${PANGU1B_BLOCK_SIZE:-vLLM default}"

python -m unittest tests.st.test_pangu1b_online_st

echo "Pangu 1B online serve ST passed."
