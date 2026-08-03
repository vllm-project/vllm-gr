#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${RECIF_MODEL_PATH:-../checkpoint}"
PORT="${RECIF_SERVER_PORT:-8011}"
OUTPUT_FILE="$(mktemp)"
SERVER_PID=""

cleanup() {
  rm -f "$OUTPUT_FILE"
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -P "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting RECIF online server on port $PORT"
vllm-gr serve "$MODEL_PATH" \
  --port "$PORT" \
  --skip-tokenizer-init \
  --logprobs-mode processed_logprobs \
  --max-logprobs 8 \
  --max-num-seqs 64 &
SERVER_PID=$!

if ! timeout 600 bash -c "
  until curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null; do
    sleep 2
  done
"; then
  echo "RECIF server failed to start within 600 seconds" >&2
  exit 1
fi

curl -sf "http://127.0.0.1:${PORT}/v1/recif/beam_search" \
  -H "Content-Type: application/json" \
  -d '{
    "history_sids": [598080194427, 628177754964, 755993681678],
    "branch_factors": [8, 8, 8],
    "beam_width": 64
  }' > "$OUTPUT_FILE"

python ./tests/system/verify_recif_beam_search.py "$OUTPUT_FILE" \
  --max-rank-drift 2
