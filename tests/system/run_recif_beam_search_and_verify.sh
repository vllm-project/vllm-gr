#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${RECIF_MODEL_PATH:-../checkpoint}"
OUTPUT_FILE="$(mktemp)"
trap 'rm -f "$OUTPUT_FILE"' EXIT

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Error: RECIF_MODEL_PATH is not a directory: $MODEL_PATH" >&2
  exit 2
fi
for required_file in config.json _model_rank0.pt external_rank0.pt; do
  if [[ ! -f "$MODEL_PATH/$required_file" ]]; then
    echo "Error: RECIF checkpoint is missing $required_file in $MODEL_PATH" >&2
    exit 2
  fi
done

echo "Running RECIF offline beam-search guard"
echo "  model: $MODEL_PATH"

python ./examples/offline_inference/beam_search/offline_recif_beam_search.py \
  --model_path "$MODEL_PATH" \
  --history 598080194427,628177754964,755993681678 \
  --bf 8,8,8 \
  --beam 64 | tee "$OUTPUT_FILE"

python ./tests/system/verify_recif_beam_search.py "$OUTPUT_FILE" \
  --max-rank-drift 2
