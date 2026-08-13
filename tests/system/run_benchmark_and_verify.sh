#!/bin/bash

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

gate_profile="${BENCHMARK_GATE_PROFILE:-main}"
case "$gate_profile" in
  main)
    elapsed_time_online_default=17
    elapsed_time_offline_default=20
    performance_mode_default="blocking"
    ;;
  decode_graph)
    elapsed_time_online_default=20
    elapsed_time_offline_default=20
    performance_mode_default="warning"
    ;;
  *)
    echo "Error: Unsupported BENCHMARK_GATE_PROFILE: $gate_profile" >&2
    exit 1
    ;;
esac

run_port=7000
timeout=600 # time for waiting server to go up
p99_ttft_th=300
p99_tpot_th=0.02
pass_1_th=0.09
pass_32_th=0.29
recall_32_th=0.04
device="gpu"
#if exist npu-smi tune some threshold
if command -v npu-smi &> /dev/null; then
   device="npu"
   board_id=$(python -c "import acl;print(acl.get_soc_name())" 2>/dev/null || echo "unknown")
   echo "identify $board_id"
   case $board_id in
    "Ascend910_9362" )
      p99_ttft_th=280
      elapsed_time_online_default=20
      elapsed_time_offline_default=25
       ;;
    * )
    echo "using default thresholds for $board_id" ;;
  esac
fi

elapsed_time_online_th="${ELAPSED_TIME_ONLINE_THRESHOLD:-$elapsed_time_online_default}"
elapsed_time_offline_th="${ELAPSED_TIME_OFFLINE_THRESHOLD:-$elapsed_time_offline_default}"
performance_mode="${PERFORMANCE_MODE:-$performance_mode_default}"
case "$performance_mode" in
  blocking|warning)
    ;;
  *)
    echo "Error: PERFORMANCE_MODE must be blocking or warning, got: $performance_mode" >&2
    exit 1
    ;;
esac

echo "Benchmark gate profile: $gate_profile"
echo "Performance mode: $performance_mode"
echo "Online elapsed-time threshold: $elapsed_time_online_th seconds"
echo "Offline elapsed-time threshold: $elapsed_time_offline_th seconds"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo ""
    echo "| Setting | Value |"
    echo "| --- | --- |"
    echo "| Device | \`$device\` |"
    echo "| Performance failures | \`$performance_mode\` |"
    echo "| Online elapsed-time limit | ${elapsed_time_online_th}s |"
    echo "| Offline elapsed-time limit | ${elapsed_time_offline_th}s |"
  } >> "$GITHUB_STEP_SUMMARY"
fi


validate_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local threshold="$3"
    local comparison_op="$4" # "le" or "ge"

    local awk_script
    if [[ "$comparison_op" == "ge" ]]; then
        awk_script='BEGIN {exit !(val >= th)}'
        condition_text="greater than or equal to"
    else
        awk_script='BEGIN {exit !(val <= th)}'
        condition_text="less than or equal to"
    fi

    if awk -v val="$metric_value" -v th="$threshold" "$awk_script"; then
        echo "SUCCESS on ${device}: ${metric_name} (${metric_value}) is ${condition_text} ${threshold}."
    else
        echo "FAILURE on ${device}: ${metric_name} (${metric_value}) is not ${condition_text} ${threshold}."
        exit 1
    fi
}

validate_performance_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local threshold="$3"

    if awk -v val="$metric_value" -v th="$threshold" 'BEGIN {exit !(val <= th)}'; then
        echo "SUCCESS on ${device}: ${metric_name} (${metric_value}) is less than or equal to ${threshold}."
    elif [[ "$performance_mode" == "warning" ]]; then
        echo "::warning title=Benchmark performance regression::${metric_name} (${metric_value}) exceeds ${threshold} on ${device}."
    else
        echo "FAILURE on ${device}: ${metric_name} (${metric_value}) exceeds ${threshold}."
        exit 1
    fi
}

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

output=$( python -m benchmarks.open_one_rec.one_rec_acc_test \
--gr \
--model "$MODEL_PATH" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--max-concurrency 10 \
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
validate_metric "pass@1" "$pass_1" "$pass_1_th" "ge"

# Validate pass@32
validate_metric "pass@32" "$pass_32" "$pass_32_th" "ge"

# Validate recall@32
validate_metric "recall@32" "$recall_32" "$recall_32_th" "ge"

# Validate elapsed_time
validate_performance_metric "elapsed_time" "$elapsed_time" "$elapsed_time_offline_th"


# online tests


# Ensure we clean up the background server process if the script exits early
stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Stopping server PID $SERVER_PID..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 3
    fi
}

trap stop_server EXIT

vllm-gr serve \
    --port $run_port \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 512 \
    --max_num_seqs 2048 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --catalog-path "$CATALOG_PATH" \
    --attention-backend CUSTOM \
    "$MODEL_PATH" &

SERVER_PID=$!

echo "Waiting for server to be ready on port $run_port..."
# Check health/models endpoint until it responds
if ! timeout $timeout bash -c "
  until curl -s http://localhost:${run_port}/v1/models > /dev/null; do
    sleep 2
  done
"; then
  echo "Server failed to start within the timeout limit of $timeout seconds."
  exit 1
fi
echo "Server is ready! Running benchmark..."

# warmup
output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model "$MODEL_PATH"  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video  \
  --num-prompts 5 \
  --use-beam-search \
  --port $run_port \
  --request-rate 1 \
  --n 128)


output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model "$MODEL_PATH"  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video  \
  --num-prompts 10 \
  --use-beam-search \
  --port $run_port \
  --request-rate 0.5 \
  --n 128)

echo "$output"

# Extract P99 TTFT


failed=$(extract_numeric_value "$output" "Failed requests")
if [[ "$failed" -ne 0 ]]; then
    echo "FAILURE: There were $failed failed requests."
    exit 1
fi

p99_ttft=$(extract_numeric_value "$output" "P99 TTFT") || exit 1
p99_tpot=$(extract_numeric_value "$output" "P99 TPOT") || exit 1

echo "Parsed P99 TTFT: ${p99_ttft}"
# Validate P99 ttft (Using awk to safely evaluate floating point comparison)
validate_performance_metric "P99 TTFT" "$p99_ttft" "$p99_ttft_th"
echo "Parsed P99 TPOT: ${p99_tpot} ms"

# Validate P99 tpot (Using awk to safely evaluate floating point comparison)
validate_performance_metric "P99 TPOT" "$p99_tpot" "$p99_tpot_th"

output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
--model "$MODEL_PATH" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--use-beam-search \
--backend openai-chat \
--task-type video \
--endpoint /v1/chat/completions \
--port $run_port \
--n 128)

echo "$output"
pass_1=$(extract_numeric_value "$output" "pass@1") || exit 1
pass_32=$(extract_numeric_value "$output" "pass@32") || exit 1
recall_32=$(extract_numeric_value "$output" "recall@32") || exit 1
elapsed_time_online=$(extract_numeric_value "$output" "elapsed_time") || exit 1

echo "Parsed pass@1: $pass_1"
echo "Parsed pass@32: $pass_32"
echo "Parsed recall@32: $recall_32"
echo "Parsed elapsed_time: $elapsed_time_online"

# Validate pass@1
validate_metric "pass@1" "$pass_1" "$pass_1_th" "ge"

# Validate pass@32
validate_metric "pass@32" "$pass_32" "$pass_32_th" "ge"

# Validate recall@32
validate_metric "recall@32" "$recall_32" "$recall_32_th" "ge"

# Validate elapsed_time
validate_performance_metric "elapsed_time" "$elapsed_time_online" "$elapsed_time_online_th"


# Stop first server before starting chunked prefill server test
stop_server


# Online test 2: Chunked Prefill Server (--max_num_seqs 100 --max_num_batched_tokens 100)
echo "Starting chunked prefill server test (--max_num_seqs 100 --max_num_batched_tokens 100)..."

vllm-gr serve \
    --port $run_port \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 512 \
    --max_num_seqs 100 \
    --max_num_batched_tokens 100 \
    --scheduling-policy priority \
    --catalog-path "$CATALOG_PATH" \
    --attention-backend CUSTOM \
    "$MODEL_PATH" &

SERVER_PID=$!

echo "Waiting for chunked prefill server to be ready on port $run_port..."
if ! timeout $timeout bash -c "
  until curl -s http://localhost:${run_port}/v1/models > /dev/null; do
    sleep 2
  done
"; then
  echo "Chunked prefill server failed to start within the timeout limit of $timeout seconds."
  exit 1
fi
echo "Chunked prefill server ready!"

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
--port $run_port \
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

validate_metric "Chunked Prefill Run 1 pass@1" "$pass_1_r1" "$pass_1_th" "ge"
validate_metric "Chunked Prefill Run 1 pass@32" "$pass_32_r1" "$pass_32_th" "ge"
validate_metric "Chunked Prefill Run 1 recall@32" "$recall_32_r1" "$recall_32_th" "ge"


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
--port $run_port \
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

validate_metric "Chunked Prefill Run 2 pass@1" "$pass_1_r2" "$pass_1_th" "ge"
validate_metric "Chunked Prefill Run 2 pass@32" "$pass_32_r2" "$pass_32_th" "ge"
validate_metric "Chunked Prefill Run 2 recall@32" "$recall_32_r2" "$recall_32_th" "ge"

stop_server

exit 0
