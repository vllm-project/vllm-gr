#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${PANGU1B_MODEL_PATH:-${PANGU_MODEL_PATH:-}}"
DATASET_PATH="${PANGU1B_DATASET_PATH:-tests/resources/pangu1b/gsm8k_mini.json}"
NUM_PROMPTS="${PANGU1B_NUM_PROMPTS:-8}"
MIN_ACCURACY="${PANGU1B_MIN_ACCURACY:-0.5}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: PANGU1B_MODEL_PATH or PANGU_MODEL_PATH must point to openPangu-Embedded-1B-V1.1" >&2
  exit 2
fi

echo "Running Pangu 1B GSM8K mini accuracy ST"
echo "  model: $MODEL_PATH"
echo "  dataset: $DATASET_PATH"
echo "  num_prompts: $NUM_PROMPTS"
echo "  min_accuracy: $MIN_ACCURACY"

python examples/offline_inference/pangu1b/pangu_generate.py \
  --model "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --min-accuracy "$MIN_ACCURACY"

echo "Pangu 1B accuracy ST passed."
