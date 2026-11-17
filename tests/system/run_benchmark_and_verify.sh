#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/ci/benchmark_gate_helpers.sh
source "$SCRIPT_DIR/../../tools/ci/benchmark_gate_helpers.sh"

MODEL_PATH="${VLLM_GR_TEST_MODEL:-OpenOneRec/OneRec-1.7B}"

if [[ -n "${VLLM_GR_TEST_MODEL:-}" ]]; then
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: VLLM_GR_TEST_MODEL is not a directory: $MODEL_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Error: config.json was not found in VLLM_GR_TEST_MODEL: $MODEL_PATH" >&2
    exit 1
  fi
fi

if [[ -z "${DATASET_PATH:-}" || ! -d "$DATASET_PATH" ]]; then
  echo "Error: DATASET_PATH must point to local OpenOneRec benchmark data." >&2
  exit 1
fi
if [[ -z "${CATALOG_PATH:-}" || ! -f "$CATALOG_PATH" ]]; then
  echo "Error: CATALOG_PATH must point to the local video catalog JSON." >&2
  exit 1
fi

required_policy_variables=(
  BENCHMARK_POLICY_NAME
  BENCHMARK_DEVICE
  ACCURACY_FAILURE_MODE
  PASS_AT_1_THRESHOLD
  PASS_AT_32_THRESHOLD
  RECALL_AT_32_THRESHOLD
  ELAPSED_TIME_ONLINE_THRESHOLD
  ELAPSED_TIME_OFFLINE_THRESHOLD
  P99_TTFT_THRESHOLD
  P99_TPOT_THRESHOLD
)
for variable_name in "${required_policy_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Error: $variable_name must be set by the benchmark policy resolver." >&2
    exit 1
  fi
done

p99_ttft_failure_mode="${P99_TTFT_FAILURE_MODE:-${PERFORMANCE_FAILURE_MODE:-}}"
p99_tpot_failure_mode="${P99_TPOT_FAILURE_MODE:-${PERFORMANCE_FAILURE_MODE:-}}"
online_elapsed_failure_mode="${ONLINE_ELAPSED_FAILURE_MODE:-${PERFORMANCE_FAILURE_MODE:-}}"
offline_elapsed_failure_mode="${OFFLINE_ELAPSED_FAILURE_MODE:-${PERFORMANCE_FAILURE_MODE:-}}"
for mode_name in \
  ACCURACY_FAILURE_MODE \
  p99_ttft_failure_mode \
  p99_tpot_failure_mode \
  online_elapsed_failure_mode \
  offline_elapsed_failure_mode; do
  mode_value="${!mode_name:-}"
  if [[ -z "$mode_value" ]]; then
    echo "Error: $mode_name is missing and no v1 performance fallback is available." >&2
    exit 1
  fi
  validate_failure_mode "$mode_name" "$mode_value" || exit $?
done

run_port="${VLLM_GR_BENCHMARK_PORT:-7000}"
server_timeout="${VLLM_GR_BENCHMARK_SERVE_TIMEOUT:-600}"
policy_name="$BENCHMARK_POLICY_NAME"
device="$BENCHMARK_DEVICE"
pass_1_th="$PASS_AT_1_THRESHOLD"
pass_32_th="$PASS_AT_32_THRESHOLD"
recall_32_th="$RECALL_AT_32_THRESHOLD"
elapsed_time_online_th="$ELAPSED_TIME_ONLINE_THRESHOLD"
elapsed_time_offline_th="$ELAPSED_TIME_OFFLINE_THRESHOLD"
p99_ttft_th="$P99_TTFT_THRESHOLD"
p99_tpot_th="$P99_TPOT_THRESHOLD"

# Constraint Tables are available on both CUDA and NPU.  Exercise that worker
# path whenever the resolved policy or local hardware is an accelerator;
# leaving CUDA on the Trie catalog would let this system test pass without
# covering the CUDA constrained-top-k kernels it is meant to protect.
#
# The triples document follows the video_catalog.json generation flow: CI
# pre-generates it in the "Build catalog" step (tools/build_constraint_triples.py)
# and passes TRIPLES_PATH, mirroring CATALOG_PATH.  When it is absent (local
# runs) it is derived from the catalog location and built on demand.
constraint_device=cuda
if [[ "$device" == "npu" || "$device" == "gpu" || "$device" == "cuda" ]] \
    || command -v npu-smi > /dev/null 2>&1 \
    || nvidia-smi -L > /dev/null 2>&1; then
  triples_path="${TRIPLES_PATH:-$(dirname "$CATALOG_PATH")/video_constraint_triples.json}"
  if [[ ! -f "$triples_path" ]]; then
    echo "Building Constraint Tables triples: $triples_path"
    if ! python tools/build_constraint_triples.py \
        --catalog "$CATALOG_PATH" \
        --output "$triples_path" \
        --begin "<|sid_begin|>" \
        --end "<|sid_end|>"; then
      echo "Error: failed to build video_constraint_triples.json from $CATALOG_PATH" >&2
      exit 1
    fi
  else
    echo "Using pre-built Constraint Tables triples: $triples_path"
  fi
  if [[ "$device" == "npu" ]]; then
    constraint_device=npu
  fi
fi

echo "Using model: $MODEL_PATH"
echo "Using dataset: $DATASET_PATH"
echo "Using catalog: $CATALOG_PATH"
echo "Benchmark policy: $policy_name"
echo "Device: $device"
echo "Accuracy failures: $ACCURACY_FAILURE_MODE"
echo "P99 TTFT failures: $p99_ttft_failure_mode"
echo "P99 TPOT failures: $p99_tpot_failure_mode"
echo "Online elapsed failures: $online_elapsed_failure_mode"
echo "Offline elapsed failures: $offline_elapsed_failure_mode"
echo "Execution, transport, timeout, and parsing failures: blocking"

extract_numeric_value() {
    local input_lines="$1"
    local string_key="$2"
    local matched_line
    local value

    matched_line=$(printf '%s\n' "$input_lines" | grep -F -m 1 -- "$string_key")
    if [[ -z "$matched_line" ]]; then
      echo "Error: key $string_key not found" >&2
      return 1
    fi
    value=$(
      printf '%s\n' "$matched_line" |
        grep -oE '[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?' |
        tail -n 1
    )
    if [[ -z "$value" ]]; then
      echo "Error: key $string_key did not contain a numeric value" >&2
      return 1
    fi
    if ! awk -v val="$value" 'BEGIN {exit !(val >= 0)}'; then
      echo "Error: key $string_key contained an invalid negative value: $value" >&2
      return 1
    fi
    printf '%s\n' "$value"
}

echo "Running OneRec offline benchmark"
# Single source of truth for the beam width driven by every client below
# (--n).  The beam KV pool / captured graph width must cover it, so both are
# derived from the same variable instead of an external environment override
# (a missing override silently disabled the pool and corrupted beam decode).
beam_width=128
config_dir="${VLLM_GR_BENCHMARK_CONFIG_DIR:-$SCRIPT_DIR/../../generated_data/benchmark_v1}"
mkdir -p "$config_dir" || exit 1
config_dir="$(cd "$config_dir" && pwd)"
gr_overrides=$(python - "$MODEL_PATH" "${triples_path:?Beam V1 requires CUDA or NPU constraint triples}" \
    "$config_dir/constraint_table.safetensors" "$constraint_device" "$beam_width" <<'PY_CONFIG'
import json
import sys
from pathlib import Path
from contextlib import redirect_stdout

model, triples, artifact_path, backend, width = sys.argv[1:]
artifact = Path(artifact_path)
with redirect_stdout(sys.stderr):
    from examples.offline_inference.beam_search.offline_beam_search import build_artifact

    digest, tokenizer_digest = build_artifact(Path(model), Path(triples), artifact)
print(json.dumps({
    "beam_width": [int(width)] * 3,
    "beam_graph_enabled": backend == "cuda",
    "worker_decision": True,
    "constraint_table": {
        "backend": backend,
        "max_top_k": int(width),
        "path": str(artifact.resolve()),
        "digests": {"artifact": digest, "tokenizer": tokenizer_digest},
    },
    "candidate_width": [int(width)] * 3,
}))
PY_CONFIG
) || exit 1
output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
  --model "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --dataset-name onerec \
  --num-prompts 100 \
  --batch-size 1 \
  --use-beam-search \
  --task-type video \
  --gr-config "$gr_overrides" \
  --n "$beam_width")
offline_status=$?
printf '%s\n' "$output"
if [[ "$offline_status" -ne 0 ]]; then
  echo "OneRec offline benchmark execution failed with exit code $offline_status." >&2
  exit "$offline_status"
fi

pass_1=$(extract_numeric_value "$output" "pass@1") || exit 1
pass_32=$(extract_numeric_value "$output" "pass@32") || exit 1
recall_32=$(extract_numeric_value "$output" "recall@32") || exit 1
elapsed_time=$(extract_numeric_value "$output" "elapsed_time") || exit 1

validate_accuracy_metric "offline pass@1" "$pass_1" "$pass_1_th" \
  "$ACCURACY_FAILURE_MODE" "$device" 86 || exit $?
validate_accuracy_metric "offline pass@32" "$pass_32" "$pass_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" 86 || exit $?
validate_accuracy_metric "offline recall@32" "$recall_32" "$recall_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" 86 || exit $?
validate_performance_metric "offline elapsed_time" "$elapsed_time" \
  "$elapsed_time_offline_th" "$offline_elapsed_failure_mode" "$device" 89 || exit $?

SERVER_PID=""
stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "Stopping OneRec server PID $SERVER_PID..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

wait_for_server() {
  local deadline=$((SECONDS + server_timeout))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      wait "$SERVER_PID" 2>/dev/null
      local server_status=$?
      echo "OneRec server exited before readiness (exit=$server_status)." >&2
      return 1
    fi
    if curl --fail --silent --show-error \
      "http://127.0.0.1:${run_port}/v1/models" > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "OneRec server did not become ready within ${server_timeout}s." >&2
  return 1
}

echo "Starting OneRec online server"
python -m vllm_gr.entrypoints.cli.main serve --gr "$MODEL_PATH" \
  --attention-config '{"backend":"CUSTOM"}' \
  --gr-config "$gr_overrides" \
  --async-scheduling \
  --no-enable-chunked-prefill \
  --port "$run_port" \
  --seed 123 \
  --chat-template "$SCRIPT_DIR/../../benchmarks/open_one_rec/qwen3_soft_switch.jinja2" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-logprobs 512 \
  --max_num_seqs 2048 \
  --max_num_batched_tokens 16384 \
  --scheduling-policy priority &
SERVER_PID=$!

wait_for_server || exit 1
echo "OneRec server is ready; running warmup."
warmup_output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model "$MODEL_PATH" \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video \
  --num-prompts 5 \
  --use-beam-search \
  --port "$run_port" \
  --request-rate 1 \
  --max-concurrency 1 \
  --n "$beam_width")
warmup_status=$?
printf '%s\n' "$warmup_output"
if [[ "$warmup_status" -ne 0 ]]; then
  echo "OneRec online warmup failed with exit code $warmup_status." >&2
  exit "$warmup_status"
fi

output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model "$MODEL_PATH" \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video \
  --num-prompts 10 \
  --use-beam-search \
  --port "$run_port" \
  --request-rate 0.5 \
  --max-concurrency 1 \
  --n "$beam_width")
online_performance_status=$?
printf '%s\n' "$output"
if [[ "$online_performance_status" -ne 0 ]]; then
  echo "OneRec online performance run failed with exit code $online_performance_status." >&2
  exit "$online_performance_status"
fi

failed=$(extract_numeric_value "$output" "Failed requests") || exit 1
if ! awk -v val="$failed" 'BEGIN {exit !(val == 0)}'; then
  echo "FAILURE: There were $failed failed requests." >&2
  exit 1
fi
p99_ttft=$(extract_numeric_value "$output" "P99 TTFT") || exit 1
p99_tpot=$(extract_numeric_value "$output" "P99 TPOT") || exit 1
validate_performance_metric "P99 TTFT" "$p99_ttft" "$p99_ttft_th" \
  "$p99_ttft_failure_mode" "$device" 87 || exit $?
validate_performance_metric "P99 TPOT" "$p99_tpot" "$p99_tpot_th" \
  "$p99_tpot_failure_mode" "$device" 88 || exit $?

output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
  --model "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --dataset-name onerec \
  --num-prompts 100 \
  --max-concurrency 1 \
  --use-beam-search \
  --backend openai-chat \
  --task-type video \
  --endpoint /v1/chat/completions \
  --port "$run_port" \
  --n "$beam_width")
online_accuracy_status=$?
printf '%s\n' "$output"
if [[ "$online_accuracy_status" -ne 0 ]]; then
  echo "OneRec online accuracy run failed with exit code $online_accuracy_status." >&2
  exit "$online_accuracy_status"
fi

pass_1=$(extract_numeric_value "$output" "pass@1") || exit 1
pass_32=$(extract_numeric_value "$output" "pass@32") || exit 1
recall_32=$(extract_numeric_value "$output" "recall@32") || exit 1
elapsed_time_online=$(extract_numeric_value "$output" "elapsed_time") || exit 1
validate_accuracy_metric "online pass@1" "$pass_1" "$pass_1_th" \
  "$ACCURACY_FAILURE_MODE" "$device" 86 || exit $?
validate_accuracy_metric "online pass@32" "$pass_32" "$pass_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" 86 || exit $?
validate_accuracy_metric "online recall@32" "$recall_32" "$recall_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" 86 || exit $?
validate_performance_metric "online elapsed_time" "$elapsed_time_online" \
  "$elapsed_time_online_th" "$online_elapsed_failure_mode" "$device" 89 || exit $?

stop_server
trap - EXIT
exit 0
