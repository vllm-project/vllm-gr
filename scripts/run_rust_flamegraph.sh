#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NUM_PROMPTS="${NUM_PROMPTS:-100}"
BEAM_WIDTH="${BEAM_WIDTH:-32}"
TASK_TYPE="${TASK_TYPE:-interactive}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-8000}"
REQUEST_RATE="${REQUEST_RATE:-1}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"
VLLM_GR_BEAM_PROC_LOG_LEVEL="${VLLM_GR_BEAM_PROC_LOG_LEVEL:-error}"
TQDM_DISABLE="${TQDM_DISABLE:-1}"
DATASET_PATH="${DATASET_PATH:-}"

if [[ -z "$DATASET_PATH" ]]; then
  echo "Error: DATASET_PATH must point to the OpenOneRec benchmark data root or a parquet sample." >&2
  exit 2
fi

RESULT_DIR="${RESULT_DIR:-results/flamegraph_rust${NUM_PROMPTS}}"
SERVER_LOG="${SERVER_LOG:-/tmp/vllm-gr-flame-rust${NUM_PROMPTS}-server.log}"
RUN_LOG="${RUN_LOG:-/tmp/vllm-gr-flame-rust${NUM_PROMPTS}-run.log}"
PID_FILE="${PID_FILE:-$RESULT_DIR/beam-proc.pid}"
FLAMEGRAPH_OUT="${FLAMEGRAPH_OUT:-$RESULT_DIR/beam-proc-rust${NUM_PROMPTS}.svg}"

PERF="${PERF:-}"
FLAMEGRAPH_BIN="${FLAMEGRAPH_BIN:-flamegraph}"
FLAMEGRAPH_FREQ="${FLAMEGRAPH_FREQ:-199}"
FLAMEGRAPH_TITLE="${FLAMEGRAPH_TITLE:-beam-proc rust ${NUM_PROMPTS} prompts}"
BEAMPROC_PID_TIMEOUT_S="${BEAMPROC_PID_TIMEOUT_S:-600}"
if [[ -z "$PERF" ]]; then
  PERF="$(command -v perf || true)"
fi
if [[ -z "$PERF" ]]; then
  echo "perf was not found. Set PERF=/path/to/perf and retry." >&2
  exit 1
fi
if ! command -v "$FLAMEGRAPH_BIN" >/dev/null 2>&1; then
  echo "flamegraph was not found in PATH. Set FLAMEGRAPH_BIN=/path/to/flamegraph and retry." >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"
: >"$SERVER_LOG"
: >"$RUN_LOG"

BENCH_PID=""
FLAMEGRAPH_PID=""
# shellcheck disable=SC2329  # Invoked indirectly by the trap below.
cleanup() {
  if [[ -n "$FLAMEGRAPH_PID" ]] && kill -0 "$FLAMEGRAPH_PID" 2>/dev/null; then
    kill "$FLAMEGRAPH_PID" 2>/dev/null || true
  fi
  if [[ -n "$BENCH_PID" ]] && kill -0 "$BENCH_PID" 2>/dev/null; then
    kill "$BENCH_PID" 2>/dev/null || true
    wait "$BENCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting Rust benchmark"
echo "  result dir: $RESULT_DIR"
echo "  server log: $SERVER_LOG"
echo "  run log:    $RUN_LOG"
echo "  gpu:        $CUDA_VISIBLE_DEVICES"
echo "  dataset:    $DATASET_PATH"
echo "  perf:       $PERF"

(
  export TASK_TYPE
  export NUM_PROMPTS
  export BEAM_WIDTH
  export REQUEST_RATE
  export VLLM_LOGGING_LEVEL
  export VLLM_GR_BEAM_PROC_LOG_LEVEL
  export TQDM_DISABLE
  export CUDA_VISIBLE_DEVICES
  export DATASET_PATH
  export PORT
  export SERVER_LOG
  export RESULT_DIR

  ./scripts/run_beamproc_mode.sh rust
) >"$RUN_LOG" 2>&1 &
BENCH_PID=$!

echo "Waiting for beam-proc pid in $SERVER_LOG ..."
deadline=$((SECONDS + BEAMPROC_PID_TIMEOUT_S))
BEAMPROC_PID=""
while [[ -z "$BEAMPROC_PID" ]]; do
  if [[ -f "$SERVER_LOG" ]]; then
    BEAMPROC_PID="$(sed -n 's/.*Started beam-proc pid=\([0-9][0-9]*\).*/\1/p' "$SERVER_LOG" | tail -n 1)"
  fi

  if [[ -n "$BEAMPROC_PID" ]] && ! kill -0 "$BEAMPROC_PID" 2>/dev/null; then
    echo "Found beam-proc pid $BEAMPROC_PID, but it is not running yet/anymore." >&2
    BEAMPROC_PID=""
  fi

  if ! kill -0 "$BENCH_PID" 2>/dev/null; then
    echo "Benchmark exited before beam-proc started." >&2
    echo "Last run log lines:" >&2
    tail -80 "$RUN_LOG" >&2 || true
    echo "Last server log lines:" >&2
    tail -80 "$SERVER_LOG" >&2 || true
    wait "$BENCH_PID"
  fi

  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for beam-proc pid." >&2
    echo "Last server log lines:" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi

  sleep 0.25
done

echo "$BEAMPROC_PID" >"$PID_FILE"
echo "Attaching flamegraph to beam-proc pid $BEAMPROC_PID"
echo "  pid file:   $PID_FILE"
echo "  svg output: $FLAMEGRAPH_OUT"

PERF="$PERF" "$FLAMEGRAPH_BIN" \
  -p "$BEAMPROC_PID" \
  -F "$FLAMEGRAPH_FREQ" \
  --ignore-status \
  --no-inline \
  --palette rust \
  --title "$FLAMEGRAPH_TITLE" \
  -o "$FLAMEGRAPH_OUT" &
FLAMEGRAPH_PID=$!

BENCH_STATUS=0
wait "$BENCH_PID" || BENCH_STATUS=$?
BENCH_PID=""

if [[ -n "$FLAMEGRAPH_PID" ]]; then
  wait "$FLAMEGRAPH_PID" || true
  FLAMEGRAPH_PID=""
fi

echo "Benchmark exit status: $BENCH_STATUS"
echo "Flamegraph: $FLAMEGRAPH_OUT"
echo "Result dir: $RESULT_DIR"
exit "$BENCH_STATUS"
