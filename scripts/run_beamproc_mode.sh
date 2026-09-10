#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "python" && "$MODE" != "rust" ]]; then
  echo "Usage: DATASET_PATH=/path/to/OpenOneRec/benchmark_data $0 python|rust"
  echo
  echo "Examples:"
  echo "  DATASET_PATH=/path/to/OpenOneRec/benchmark_data $0 python"
  echo "  DATASET_PATH=/path/to/OpenOneRec/benchmark_data $0 rust"
  exit 2
fi

PORT="${PORT:-7000}"
MODEL="${MODEL:-OpenOneRec/OneRec-1.7B}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
BEAM_WIDTH="${BEAM_WIDTH:-128}"
MAX_LOGPROBS="${MAX_LOGPROBS:-$BEAM_WIDTH}"
REQUEST_RATE="${REQUEST_RATE:-1}"
TASK_TYPE="${TASK_TYPE:-interactive}"
RESULT_DIR="${RESULT_DIR:-results/acc_test_results}"
SERVER_LOG="${SERVER_LOG:-/tmp/vllm-gr-${MODE}-server.log}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

if [[ -z "${DATASET_PATH:-}" ]]; then
  echo "Error: DATASET_PATH must point to the OpenOneRec benchmark data root or a parquet sample." >&2
  exit 2
fi
if [[ "$DATASET_PATH" == *.parquet ]]; then
  DATASET_PATH="$(dirname "$(dirname "$DATASET_PATH")")"
fi
if [[ "$(basename "$DATASET_PATH")" == "$TASK_TYPE" && -f "$(dirname "$DATASET_PATH")/sid2pid.json" ]]; then
  DATASET_PATH="$(dirname "$DATASET_PATH")"
fi
if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Error: DATASET_PATH is not a directory: $DATASET_PATH" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$MODE" == "rust" ]]; then
  export VLLM_GR_BEAM_PROC_ENABLED=1
  export VLLM_GR_BEAM_PROC_NUM_WORKERS="${VLLM_GR_BEAM_PROC_NUM_WORKERS:-4}"
  export VLLM_GR_BEAM_PROC_BINARY="${VLLM_GR_BEAM_PROC_BINARY:-$ROOT_DIR/rust/target/release/beam-proc}"
  export VLLM_GR_BEAM_PROC_LOG_LEVEL="${VLLM_GR_BEAM_PROC_LOG_LEVEL:-off}"
  export VLLM_GR_BEAM_PROC_LOG_FILE="${VLLM_GR_BEAM_PROC_LOG_FILE:-/tmp/vllm-gr-beam-proc.log}"

  if [[ ! -x "$VLLM_GR_BEAM_PROC_BINARY" ]]; then
    echo "BeamProc binary not found at $VLLM_GR_BEAM_PROC_BINARY; building release binary..."
    cargo build --release --manifest-path "$ROOT_DIR/rust/Cargo.toml" --bin beam-proc
  fi
else
  unset VLLM_GR_BEAM_PROC_ENABLED
  unset VLLM_GR_BEAM_PROC_FINAL_TIMEOUT_S
  unset VLLM_GR_BEAM_PROC_COLLECT_APISERVER_OUTPUT
  unset VLLM_GR_BEAM_PROC_BINARY
  unset VLLM_GR_BEAM_PROC_CONFIG
  unset VLLM_GR_BEAM_PROC_APISERVER_OUTPUT_ADDR
  unset VLLM_GR_BEAM_PROC_ENGINECORE_ADDR
  unset VLLM_GR_BEAM_PROC_STARTED
  unset VLLM_GR_BEAM_PROC_NUM_WORKERS
  unset VLLM_GR_BEAM_PROC_LOG_LEVEL
  unset VLLM_GR_BEAM_PROC_LOG_FILE
fi

echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Dataset root: $DATASET_PATH"
echo "Port: $PORT"
echo "Max logprobs: $MAX_LOGPROBS"
echo "Server log: $SERVER_LOG"
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
  echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
fi

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    pkill -P "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

serve_args=(
  serve
  "$MODEL"
  --port "$PORT"
  --max-logprobs "$MAX_LOGPROBS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens 16384
  --scheduling-policy priority
  --attention-backend CUSTOM
  --disable-uvicorn-access-log
  --disable-log-stats
  --default-chat-template-kwargs '{"enable_thinking": false}'
)
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
  serve_args+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  serve_args+=(--enforce-eager)
fi
vllm-gr "${serve_args[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for server on http://localhost:${PORT}/v1/models ..."
deadline=$((SECONDS + ${SERVER_START_TIMEOUT:-600}))
while ! curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process exited before port ${PORT} became ready."
    echo "Last server log lines from ${SERVER_LOG}:"
    tail -120 "$SERVER_LOG" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Server did not become ready on port ${PORT} before timeout."
    echo "Last server log lines from ${SERVER_LOG}:"
    tail -120 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 2
done

echo "Warmup..."
python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model "$MODEL" \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type "$TASK_TYPE" \
  --num-prompts 10 \
  --use-beam-search \
  --port "$PORT" \
  --request-rate "$REQUEST_RATE" \
  --n "$BEAM_WIDTH"

echo "Accuracy benchmark..."
acc_args=(
  -m benchmarks.open_one_rec.one_rec_acc_test
  --endpoint /v1/chat/completions
  --backend openai-chat
  --model "$MODEL"
  --dataset-name onerec
  --dataset-path "$DATASET_PATH"
  --task-type "$TASK_TYPE"
  --num-prompts "$NUM_PROMPTS"
  --result-dir "$RESULT_DIR"
  --port "$PORT"
  --n "$BEAM_WIDTH"
)
if [[ -n "${CONCURRENCY_LIMIT:-}" ]]; then
  acc_args+=(--max-concurrency "$CONCURRENCY_LIMIT")
fi
python "${acc_args[@]}"
