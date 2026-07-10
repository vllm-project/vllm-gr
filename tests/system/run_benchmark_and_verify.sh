#!/bin/bash

run_port=7000
timeout=600 # time for waiting server to go up
p99_ttft_th="${VLLM_GR_P99_TTFT_THRESHOLD:-300}"
p99_tpot_th="${VLLM_GR_P99_TPOT_THRESHOLD:-0.02}"
pass_1_th=0.09
pass_32_th=0.2
recall_32_th=0.04
elapsed_time_online_th="${VLLM_GR_ONLINE_ELAPSED_THRESHOLD:-15}"
elapsed_time_offline_th="${VLLM_GR_OFFLINE_ELAPSED_THRESHOLD:-20}"
enforce_performance_thresholds="${VLLM_GR_ENFORCE_PERFORMANCE_THRESHOLDS:-0}"
ONEREC_MODEL_PATH="OpenOneRec/OneRec-1.7B"
if [ -n "$REC_MODEL" ]; then
    ONEREC_MODEL_PATH="$REC_MODEL"
fi
if [ -n "$REC_DATA" ]; then
    DATASET_PATH="$REC_DATA"
fi
if [ -n "$REC_CATALOG" ]; then
    CATALOG_PATH="$REC_CATALOG"
fi

echo "ONEREC_MODEL_PATH: $ONEREC_MODEL_PATH"
echo "ONEREC_DATASET_PATH: $DATASET_PATH"
echo "ONEREC_CATALOG_PATH: $CATALOG_PATH"
device="gpu"
#if exist npu-smi tune some threshold
if command -v npu-smi &> /dev/null; then
   device="npu"
   board_id=$(python -c "import acl;print(acl.get_soc_name())" 2>/dev/null || echo "unknown")
   echo "identify $board_id"
   case $board_id in
    "Ascend910_9362" )
      if [[ -z "${VLLM_GR_P99_TTFT_THRESHOLD+x}" ]]; then
        p99_ttft_th=280
      fi
      if [[ -z "${VLLM_GR_ONLINE_ELAPSED_THRESHOLD+x}" ]]; then
        elapsed_time_online_th=20
      fi
      if [[ -z "${VLLM_GR_OFFLINE_ELAPSED_THRESHOLD+x}" ]]; then
        elapsed_time_offline_th=60
      fi
       ;;
    * )
    echo "using default thresholds for $board_id" ;;
  esac
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

    if [[ "$enforce_performance_thresholds" =~ ^(1|true|yes|on)$ ]]; then
        validate_metric "$metric_name" "$metric_value" "$threshold" "le"
    else
        echo "INFO on ${device}: ${metric_name}=${metric_value}; performance gate disabled (reference threshold: <= ${threshold})."
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

capture_output() {
    local label="$1"
    local status
    shift
    output=$("$@")
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "$output"
        echo "FAILURE: ${label} exited with status ${status}." >&2
        exit "$status"
    fi
}

echo "offline test"

capture_output "offline accuracy benchmark" python -m benchmarks.open_one_rec.one_rec_acc_test \
--gr \
--model "$ONEREC_MODEL_PATH" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--max-concurrency 10 \
--use-beam-search \
--task-type video \
--catalog-path "$CATALOG_PATH" \
--backend CUSTOM \
--n 128


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

# Performance is informational by default. Accuracy remains a required gate.
validate_performance_metric "elapsed_time" "$elapsed_time" "$elapsed_time_offline_th"


# online tests


# Ensure we clean up the background server process if the script exits early
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; sleep 2; pkill -P "$SERVER_PID" 2>/dev/null || true; fi' EXIT

vllm-gr serve \
    --port $run_port \
    --seed 123 \
    --no-async-scheduling \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 512 \
    --max_num_seqs 2048 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --catalog-path "$CATALOG_PATH" \
    --attention-backend CUSTOM \
    "$ONEREC_MODEL_PATH" &

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
capture_output "online warmup benchmark" python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model "$ONEREC_MODEL_PATH"  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video  \
  --num-prompts 5 \
  --use-beam-search \
  --port $run_port \
  --request-rate 1 \
  --n 128


capture_output "online performance benchmark" python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model "$ONEREC_MODEL_PATH"  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video  \
  --num-prompts 10 \
  --use-beam-search \
  --port $run_port \
  --request-rate 0.5 \
  --n 128

echo "$output"

# Extract P99 TTFT


failed=$(extract_numeric_value "$output" "Failed requests") || exit 1
if [[ "$failed" -ne 0 ]]; then
    echo "FAILURE: There were $failed failed requests."
    exit 1
fi

p99_ttft=$(extract_numeric_value "$output" "P99 TTFT") || exit 1
p99_tpot=$(extract_numeric_value "$output" "P99 TPOT") || exit 1

echo "Parsed P99 TTFT: ${p99_ttft}"
# Performance is informational by default. Failed requests remain a required gate.
validate_performance_metric "P99 TTFT" "$p99_ttft" "$p99_ttft_th"
echo "Parsed P99 TPOT: ${p99_tpot} ms"

validate_performance_metric "P99 TPOT" "$p99_tpot" "$p99_tpot_th"

capture_output "online accuracy benchmark" python -m benchmarks.open_one_rec.one_rec_acc_test \
--model "$ONEREC_MODEL_PATH" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--use-beam-search \
--backend openai-chat \
--task-type video \
--endpoint /v1/chat/completions \
--port $run_port \
--n 128

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

validate_performance_metric "elapsed_time" "$elapsed_time_online" "$elapsed_time_online_th"



exit 0
