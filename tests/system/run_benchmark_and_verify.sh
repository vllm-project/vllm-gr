#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
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

if [[ -z "${DATASET_PATH:-}" ]]; then
  echo "Error: DATASET_PATH must point to the local OpenOneRec benchmark data." >&2
  exit 1
fi
if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Error: DATASET_PATH is not a directory: $DATASET_PATH" >&2
  exit 1
fi

if [[ -z "${CATALOG_PATH:-}" ]]; then
  echo "Error: CATALOG_PATH must point to the local video catalog JSON." >&2
  exit 1
fi
if [[ ! -f "$CATALOG_PATH" ]]; then
  echo "Error: CATALOG_PATH is not a file: $CATALOG_PATH" >&2
  exit 1
fi

echo "Using model: $MODEL_PATH"
echo "Using dataset: $DATASET_PATH"
echo "Using catalog: $CATALOG_PATH"

required_policy_variables=(
  BENCHMARK_POLICY_NAME
  BENCHMARK_DEVICE
  ACCURACY_FAILURE_MODE
  PERFORMANCE_FAILURE_MODE
  ONLINE_NON_ACCURACY_FAILURE_MODE
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

for failure_mode_name in \
  ACCURACY_FAILURE_MODE \
  PERFORMANCE_FAILURE_MODE \
  ONLINE_NON_ACCURACY_FAILURE_MODE; do
  case "${!failure_mode_name}" in
    blocking|warning)
      ;;
    *)
      echo "Error: $failure_mode_name must be blocking or warning, got: ${!failure_mode_name}" >&2
      exit 1
      ;;
  esac
done

run_port="${VLLM_GR_BENCHMARK_PORT:-7000}"
unset VLLM_GR_BENCHMARK_PORT
timeout=600 # time for waiting server to go up
policy_name="$BENCHMARK_POLICY_NAME"
device="$BENCHMARK_DEVICE"
pass_1_th="$PASS_AT_1_THRESHOLD"
pass_32_th="$PASS_AT_32_THRESHOLD"
recall_32_th="$RECALL_AT_32_THRESHOLD"
elapsed_time_online_th="$ELAPSED_TIME_ONLINE_THRESHOLD"
elapsed_time_offline_th="$ELAPSED_TIME_OFFLINE_THRESHOLD"
p99_ttft_th="$P99_TTFT_THRESHOLD"
p99_tpot_th="$P99_TPOT_THRESHOLD"
rust_p99_ttft_th=250
avg_decode_time_th=120
pass32_th_bs=0.21 #Beam star is measured without catalog.
online_accuracy_failure_status=86
online_performance_failure_status=87

echo "Benchmark policy: $policy_name"
echo "Device: $device"
echo "Accuracy failure mode: $ACCURACY_FAILURE_MODE"
echo "Performance failure mode: $PERFORMANCE_FAILURE_MODE"
echo "OneRec online non-accuracy failure mode: $ONLINE_NON_ACCURACY_FAILURE_MODE"
echo "Online elapsed-time threshold: $elapsed_time_online_th seconds"
echo "Offline elapsed-time threshold: $elapsed_time_offline_th seconds"

# Function to extract numeric value based on string key from multi-line input
extract_numeric_value() {
    local input_lines="$1"
    local string_key="$2"

    # Use grep to find the line containing the string key
    local matched_line=$(echo "$input_lines" | grep -m 1 "$string_key")

    if [[ -n "$matched_line" ]]; then
        # Use regex to extract the numeric value from the matched line
        local value=$(echo "$matched_line" | grep -oP "$string_key.*:\s*\K[0-9.]+")
        if [[ -z "$value" ]]; then
            echo "Error: Key $string_key not found, or formatting did not match" >&2
            return  1
        else
            echo "$value"
        fi
    else
      echo "key $string_key not found" >&2
      return 1
    fi
}
echo "offline test"
echo "Using dataset root: $DATASET_PATH"

output=$( python -m benchmarks.open_one_rec.one_rec_acc_test \
--gr \
--model "$MODEL_PATH" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--max-concurrency 16 \
--use-beam-search \
--task-type video \
--catalog-path "$CATALOG_PATH" \
--backend CUSTOM \
--n 128 )


echo "$output"

pass_1=$(extract_numeric_value "$output" "pass@1") || exit 1
pass_32=$(extract_numeric_value "$output" "pass@32") || exit 1
recall_32=$(extract_numeric_value "$output" "recall@32") || exit 1
elapsed_time=$(extract_numeric_value "$output" "elapsed_time") || exit 1


echo "Parsed pass@1: $pass_1"
echo "Parsed pass@32: $pass_32"
echo "Parsed recall@32: $recall_32"
echo "Parsed elapsed_time: $elapsed_time"


# Validate pass@1
validate_accuracy_metric "pass@1" "$pass_1" "$pass_1_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?

# Validate pass@32
validate_accuracy_metric "pass@32" "$pass_32" "$pass_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?

# Validate recall@32
validate_accuracy_metric "recall@32" "$recall_32" "$recall_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?

# Validate elapsed_time
validate_performance_metric "elapsed_time" "$elapsed_time" "$elapsed_time_offline_th" \
  "$PERFORMANCE_FAILURE_MODE" "$device" "$online_performance_failure_status" || exit $?


# Online tests run in a subshell so accuracy, performance, and execution
# failures can be classified before applying their independent policy modes.
run_online_tests() (


# Ensure we clean up the background server process if the script exits early
stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Stopping server PID $SERVER_PID..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 3
    fi
    if [ -n "${BEAMPROC_LOG_TAIL_PID:-}" ]; then
        kill "$BEAMPROC_LOG_TAIL_PID" 2>/dev/null || true
        wait "$BEAMPROC_LOG_TAIL_PID" 2>/dev/null || true
        BEAMPROC_LOG_TAIL_PID=""
    fi
}

trap stop_server EXIT

# Start a vllm-gr server in the background, wait until it is ready, and
# optionally run a warmup benchmark against it.
# Usage: start_server <name> <warmup_prompts> [extra vllm-gr serve args...]
#   warmup_prompts: number of warmup prompts, 0 to skip warmup
start_server() {
    local name="$1"
    local warmup_prompts="$2"
    shift 2

    echo "Starting $name server..."
    vllm-gr serve \
        --port "$run_port" \
        --seed 123 \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --max-logprobs 512 \
        --scheduling-policy priority \
        --attention-backend CUSTOM \
        "$@" \
        "$MODEL_PATH" &

    SERVER_PID=$!

    echo "Waiting for $name server to be ready on port $run_port..."
    if ! timeout $timeout bash -c "
        until curl -s http://localhost:${run_port}/v1/models > /dev/null; do
            sleep 2
        done
    "; then
        echo "$name server failed to start within the timeout limit of $timeout seconds."
        exit 1
    fi
    echo "$name server is ready!"

    if [[ "$warmup_prompts" -gt 0 ]]; then
        echo "Running warmup with $warmup_prompts prompts..."
        output=$(python -m benchmarks.open_one_rec.one_rec_main \
            bench serve \
            --endpoint /v1/chat/completions \
            --backend openai-chat \
            --model "$MODEL_PATH" \
            --dataset-name onerec \
            --dataset-path "$DATASET_PATH" \
            --task-type video \
            --num-prompts "$warmup_prompts" \
            --use-beam-search \
            --port "$run_port" \
            --request-rate 1 \
            --n 128)
        echo "Warmup complete."
    fi
}

# Run the online latency workload and validate its request failures, P99 TTFT,
# and P99 TPOT. Both Python and Rust use this helper so their benchmark
# parameters remain identical.
run_latency_benchmark() {
    local implementation="$1"
    local ttft_threshold="$2"
    local metric_prefix=""
    local output
    local failed
    local p99_ttft
    local p99_tpot

    if [[ -n "$implementation" ]]; then
        metric_prefix="$implementation "
    fi

    echo "Running ${metric_prefix}latency benchmark..."
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
      --n 128)

    echo "$output"

    failed=$(extract_numeric_value "$output" "Failed requests") || return 1
    if [[ "$failed" -ne 0 ]]; then
        echo "FAILURE: ${metric_prefix}latency benchmark had $failed failed requests."
        return 1
    fi

    p99_ttft=$(extract_numeric_value "$output" "P99 TTFT") || return 1
    p99_tpot=$(extract_numeric_value "$output" "P99 TPOT") || return 1

    echo "Parsed ${metric_prefix}P99 TTFT: ${p99_ttft} ms"
    validate_performance_metric "${metric_prefix}P99 TTFT" "$p99_ttft" "$ttft_threshold" \
      "$PERFORMANCE_FAILURE_MODE" "$device" "$online_performance_failure_status" || return $?

    echo "Parsed ${metric_prefix}P99 TPOT: ${p99_tpot} ms"
    validate_performance_metric "${metric_prefix}P99 TPOT" "$p99_tpot" "$p99_tpot_th" \
      "$PERFORMANCE_FAILURE_MODE" "$device" "$online_performance_failure_status" || return $?
}

# Run the online accuracy workload and validate request success, recommendation
# accuracy, and elapsed time. Rust reuses the exact Python workload and gates.
run_online_accuracy_benchmark() {
    local implementation="$1"
    local metric_prefix=""
    local output
    local request_failed
    local pass_1
    local pass_32
    local recall_32
    local elapsed_time_online

    if [[ -n "$implementation" ]]; then
        metric_prefix="$implementation "
    fi

    echo "Running ${metric_prefix}online accuracy benchmark..."
    output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
      --model "$MODEL_PATH" \
      --dataset-path "$DATASET_PATH" \
      --dataset-name onerec \
      --num-prompts 100 \
      --use-beam-search \
      --backend openai-chat \
      --task-type video \
      --endpoint /v1/chat/completions \
      --port "$run_port" \
      --n 128)

    echo "$output"

    request_failed=$(extract_numeric_value "$output" "request_failed") || return 1
    if [[ "$request_failed" -ne 0 ]]; then
        echo "FAILURE: ${metric_prefix}online accuracy benchmark had $request_failed failed requests."
        return 1
    fi

    pass_1=$(extract_numeric_value "$output" "pass@1") || return 1
    pass_32=$(extract_numeric_value "$output" "pass@32") || return 1
    recall_32=$(extract_numeric_value "$output" "recall@32") || return 1
    elapsed_time_online=$(extract_numeric_value "$output" "elapsed_time") || return 1

    echo "Parsed ${metric_prefix}pass@1: $pass_1"
    echo "Parsed ${metric_prefix}pass@32: $pass_32"
    echo "Parsed ${metric_prefix}recall@32: $recall_32"
    echo "Parsed ${metric_prefix}elapsed_time: $elapsed_time_online"

    validate_accuracy_metric "${metric_prefix}pass@1" "$pass_1" "$pass_1_th" \
      "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || return $?
    validate_accuracy_metric "${metric_prefix}pass@32" "$pass_32" "$pass_32_th" \
      "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || return $?
    validate_accuracy_metric "${metric_prefix}recall@32" "$recall_32" "$recall_32_th" \
      "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || return $?
    validate_performance_metric "${metric_prefix}elapsed_time" "$elapsed_time_online" \
      "$elapsed_time_online_th" "$PERFORMANCE_FAILURE_MODE" "$device" \
      "$online_performance_failure_status" || return $?
}

start_server "primary" 5 \
    --max_num_seqs 128 \
    --max_num_batched_tokens 16384 \
    --catalog-path "$CATALOG_PATH"

run_latency_benchmark "" "$p99_ttft_th" || exit $?
run_online_accuracy_benchmark "" || exit $?


# Stop first server before starting chunked prefill server test
stop_server


# Online test 2: Chunked Prefill Server (--max_num_seqs 100 --max_num_batched_tokens 100)
echo "Starting chunked prefill server test (--max_num_seqs 100 --max_num_batched_tokens 100)..."

start_server "chunked prefill" 0 \
    --max_num_seqs 100 \
    --max_num_batched_tokens 100 \
    --catalog-path "$CATALOG_PATH"

# Run 1: Cold KV Cache (first execution)
echo "Running Run 1 (Cold KV Cache)..."
output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
--model "$MODEL_PATH" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--use-beam-search \
--backend openai-chat \
--task-type video \
--endpoint /v1/chat/completions \
--port "$run_port" \
--n 128)

echo "$output"
pass_1_r1=$(extract_numeric_value "$output" "pass@1") || exit 1
pass_32_r1=$(extract_numeric_value "$output" "pass@32") || exit 1
recall_32_r1=$(extract_numeric_value "$output" "recall@32") || exit 1
elapsed_time_r1=$(extract_numeric_value "$output" "elapsed_time") || exit 1

echo "Run 1 (Cold Cache) - Parsed pass@1: $pass_1_r1"
echo "Run 1 (Cold Cache) - Parsed pass@32: $pass_32_r1"
echo "Run 1 (Cold Cache) - Parsed recall@32: $recall_32_r1"
echo "Run 1 (Cold Cache) - Parsed elapsed_time: $elapsed_time_r1"

validate_accuracy_metric "Chunked Prefill Run 1 pass@1" "$pass_1_r1" "$pass_1_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?
validate_accuracy_metric "Chunked Prefill Run 1 pass@32" "$pass_32_r1" "$pass_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?
validate_accuracy_metric "Chunked Prefill Run 1 recall@32" "$recall_32_r1" "$recall_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?


# Run 2: Warm KV Cache (second execution)
echo "Running Run 2 (Warm KV Cache)..."
output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
--model "$MODEL_PATH" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--use-beam-search \
--backend openai-chat \
--task-type video \
--endpoint /v1/chat/completions \
--port "$run_port" \
--n 128)

echo "$output"
pass_1_r2=$(extract_numeric_value "$output" "pass@1") || exit 1
pass_32_r2=$(extract_numeric_value "$output" "pass@32") || exit 1
recall_32_r2=$(extract_numeric_value "$output" "recall@32") || exit 1
elapsed_time_r2=$(extract_numeric_value "$output" "elapsed_time") || exit 1

echo "Run 2 (Warm Cache) - Parsed pass@1: $pass_1_r2"
echo "Run 2 (Warm Cache) - Parsed pass@32: $pass_32_r2"
echo "Run 2 (Warm Cache) - Parsed recall@32: $recall_32_r2"
echo "Run 2 (Warm Cache) - Parsed elapsed_time: $elapsed_time_r2"

validate_accuracy_metric "Chunked Prefill Run 2 pass@1" "$pass_1_r2" "$pass_1_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?
validate_accuracy_metric "Chunked Prefill Run 2 pass@32" "$pass_32_r2" "$pass_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?
validate_accuracy_metric "Chunked Prefill Run 2 recall@32" "$recall_32_r2" "$recall_32_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?

stop_server

# Run beam star check
start_server "beam-star" 5 \
    --max_num_seqs 256 \
    --gpu-memory-utilization 0.8 \
    --max_num_batched_tokens 16384

output=$(python -m benchmarks.open_one_rec.one_rec_main \
      bench serve \
      --endpoint /v1/chat/completions \
      --backend openai-chat \
      --model "$MODEL_PATH" \
      --dataset-name onerec \
      --dataset-path "$DATASET_PATH" \
      --task-type video \
      --num-prompts 100 \
      --use-beam-search \
      --port "$run_port" \
      --max-concurrency 1 \
      --n 128 \
      --beam-star 2.2)
echo "Beam Star"
echo "$output"
avg_decode_time=$(extract_numeric_value "$output" "Avg Decode Time") || exit 1
validate_performance_metric "Avg Decode Time" "$avg_decode_time" "$avg_decode_time_th" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?

output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
--model "$MODEL_PATH" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--use-beam-search \
--backend openai-chat \
--task-type video \
--endpoint /v1/chat/completions \
--port "$run_port" \
--max-concurrency 16 \
--beam-star 2.2 \
--n 128)

pass32=$(extract_numeric_value "$output" "pass@32") || exit 1
validate_accuracy_metric "Beam Star pass@32" "$pass32" "$pass32_th_bs" \
  "$ACCURACY_FAILURE_MODE" "$device" "$online_accuracy_failure_status" || exit $?

stop_server

# Rust BeamProc latency and accuracy gates.
if [[ "$device" == "gpu" || "$device" == "npu" ]]; then
    rust_binary="$REPO_ROOT/rust/target/release/beam-proc"

    echo "Building Rust BeamProc release binary for the latency and accuracy gates..."
    rustc --version
    cargo --version
    if ! cargo build \
      --release \
      --locked \
      --manifest-path "$REPO_ROOT/rust/Cargo.toml" \
      --target-dir "$REPO_ROOT/rust/target" \
      --bin beam-proc; then
        echo "Rust BeamProc release build failed." >&2
        exit 1
    fi
    if [[ ! -x "$rust_binary" ]]; then
        echo "Rust BeamProc build did not produce an executable at $rust_binary." >&2
        exit 1
    fi

    beamproc_log_dir="${RUNNER_TEMP:-/tmp}"
    beamproc_log_file=$(mktemp "$beamproc_log_dir/vllm-gr-beam-proc.XXXXXX.log")
    echo "Rust BeamProc binary: $rust_binary"
    echo "Rust BeamProc log file: $beamproc_log_file (streaming to CI stderr)"

    export VLLM_GR_BEAM_PROC_ENABLED=1
    export VLLM_GR_BEAM_PROC_NUM_WORKERS=4
    export VLLM_GR_BEAM_PROC_BINARY="$rust_binary"
    export VLLM_GR_BEAM_PROC_LOG_LEVEL=warn
    export VLLM_GR_BEAM_PROC_LOG_FILE="$beamproc_log_file"
    export VLLM_GR_BEAM_PROC_FINAL_TIMEOUT_S=60

    # Send Rust logs through an unbuffered stream to stderr so CI wrappers
    # that capture command stdout still display BeamProc progress immediately.
    echo "[CI] BeamProc log forwarding started: $beamproc_log_file" \
        >> "$beamproc_log_file"
    stdbuf -oL -eL tail -n +1 -F "$beamproc_log_file" >&2 &
    BEAMPROC_LOG_TAIL_PID=$!
    echo "BeamProc log tail PID: $BEAMPROC_LOG_TAIL_PID"

    rust_device_args=()
    if [[ "$device" == "npu" ]]; then
        # Leave extra sampler headroom for the concurrent Rust accuracy run.
        rust_device_args+=(--gpu-memory-utilization 0.8)
    fi

    start_server "Rust BeamProc" 5 \
        "${rust_device_args[@]}" \
        --max_num_seqs 128 \
        --max_num_batched_tokens 16384 \
        --catalog-path "$CATALOG_PATH"

    run_latency_benchmark "Rust" "$rust_p99_ttft_th" || exit $?
    run_online_accuracy_benchmark "Rust" || exit $?
    stop_server

    unset VLLM_GR_BEAM_PROC_ENABLED
    unset VLLM_GR_BEAM_PROC_NUM_WORKERS
    unset VLLM_GR_BEAM_PROC_BINARY
    unset VLLM_GR_BEAM_PROC_LOG_LEVEL
    unset VLLM_GR_BEAM_PROC_LOG_FILE
    unset VLLM_GR_BEAM_PROC_FINAL_TIMEOUT_S
else
    echo "Skipping Rust BeamProc latency and accuracy gates on unsupported device: $device."
fi
)

online_status=0
run_online_tests || online_status=$?

handle_online_test_status \
  "$online_status" \
  "$online_accuracy_failure_status" \
  "$online_performance_failure_status" \
  "$ONLINE_NON_ACCURACY_FAILURE_MODE" \
  "$policy_name" \
  "$device" || exit $?

exit 0
