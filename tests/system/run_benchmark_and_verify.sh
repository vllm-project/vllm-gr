#!/bin/bash

# vllm-gr beam-search worker hooks currently target ModelRunner V1. Force V1
# for this accuracy/performance gate even if the outer CI environment enables
# V2 globally; V2 coverage must use a separate non-gating job until adapted.
export VLLM_USE_V2_MODEL_RUNNER=0

run_port="${BENCHMARK_PORT:-7000}"
timeout="${SERVER_START_TIMEOUT_SECONDS:-600}" # time for waiting server to go up

default_p99_ttft_th=300
default_p99_tpot_th=0.02
default_elapsed_time_online_th=15
default_elapsed_time_offline_th=20

device="gpu"
# If npu-smi exists, tune the default performance thresholds for that device.
# Explicit environment overrides are applied after device detection.
if command -v npu-smi &> /dev/null; then
   device="npu"
   board_id=$(python -c "import acl;print(acl.get_soc_name())" 2>/dev/null || echo "unknown")
   echo "identify $board_id"
   case $board_id in
    "Ascend910_9362" )
      default_p99_ttft_th=280
      default_elapsed_time_online_th=20
      default_elapsed_time_offline_th=60
       ;;
    * )
    echo "using default thresholds for $board_id" ;;
  esac
fi

p99_ttft_th="${P99_TTFT_THRESHOLD_MS:-$default_p99_ttft_th}"
p99_tpot_th="${P99_TPOT_THRESHOLD_MS:-$default_p99_tpot_th}"
pass_1_th="${PASS_1_THRESHOLD:-0.09}"
pass_32_th="${PASS_32_THRESHOLD:-0.2}"
recall_32_th="${RECALL_32_THRESHOLD:-0.04}"
elapsed_time_online_th="${ONLINE_ACCURACY_ELAPSED_THRESHOLD_SECONDS:-$default_elapsed_time_online_th}"
elapsed_time_offline_th="${OFFLINE_ACCURACY_ELAPSED_THRESHOLD_SECONDS:-$default_elapsed_time_offline_th}"

benchmark_model="${BENCHMARK_MODEL:-OpenOneRec/OneRec-1.7B}"
benchmark_task_type="${BENCHMARK_TASK_TYPE:-video}"
beam_width="${BEAM_WIDTH:-128}"
offline_accuracy_num_prompts="${OFFLINE_ACCURACY_NUM_PROMPTS:-100}"
offline_max_concurrency="${OFFLINE_MAX_CONCURRENCY:-10}"
online_warmup_num_prompts="${ONLINE_WARMUP_NUM_PROMPTS:-5}"
online_warmup_request_rate="${ONLINE_WARMUP_REQUEST_RATE:-1}"
online_perf_num_prompts="${ONLINE_PERF_NUM_PROMPTS:-10}"
online_perf_request_rate="${ONLINE_PERF_REQUEST_RATE:-0.5}"
online_accuracy_num_prompts="${ONLINE_ACCURACY_NUM_PROMPTS:-100}"
kpi_gating_mode="${KPI_GATING_MODE:-enforce}"

case "$kpi_gating_mode" in
  enforce|report-only) ;;
  *)
    echo "Error: KPI_GATING_MODE must be 'enforce' or 'report-only', got '$kpi_gating_mode'." >&2
    exit 2
    ;;
esac

echo "Benchmark validation configuration"
echo "  device: $device"
echo "  model: $benchmark_model"
echo "  task_type: $benchmark_task_type"
echo "  beam_width: $beam_width"
echo "  gating_mode: $kpi_gating_mode"
echo "  offline_accuracy_num_prompts: $offline_accuracy_num_prompts"
echo "  online_accuracy_num_prompts: $online_accuracy_num_prompts"
echo "  online_perf_num_prompts: $online_perf_num_prompts"
echo "  online_perf_request_rate: $online_perf_request_rate"
echo "  thresholds:"
echo "    pass@1 >= $pass_1_th"
echo "    pass@32 >= $pass_32_th"
echo "    recall@32 >= $recall_32_th"
echo "    offline_accuracy_elapsed_time <= ${elapsed_time_offline_th}s"
echo "    online_accuracy_elapsed_time <= ${elapsed_time_online_th}s"
echo "    P99_TTFT <= ${p99_ttft_th}ms"
echo "    P99_TPOT <= ${p99_tpot_th}ms"


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
        if [[ "$kpi_gating_mode" == "report-only" ]]; then
            echo "WARNING on ${device}: ${metric_name} (${metric_value}) is not ${condition_text} ${threshold}; report-only mode does not fail the run."
        else
            echo "FAILURE on ${device}: ${metric_name} (${metric_value}) is not ${condition_text} ${threshold}."
            exit 1
        fi
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
--model "$benchmark_model" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts "$offline_accuracy_num_prompts" \
--max-concurrency "$offline_max_concurrency" \
--use-beam-search \
--task-type "$benchmark_task_type" \
--catalog-path "$CATALOG_PATH" \
--backend CUSTOM \
--n "$beam_width" )


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
validate_metric "elapsed_time" "$elapsed_time" "$elapsed_time_offline_th" "le"


# online tests


# Ensure we clean up the background server process if the script exits early
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; sleep 2; pkill -P "$SERVER_PID" 2>/dev/null || true; fi' EXIT

vllm-gr serve \
    --port "$run_port" \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 512 \
    --max_num_seqs 2048 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --catalog-path "$CATALOG_PATH" \
    --attention-backend CUSTOM \
    "$benchmark_model" &

SERVER_PID=$!

echo "Waiting for server to be ready on port $run_port..."
# Check health/models endpoint until it responds
if ! timeout "$timeout" bash -c "
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
  --model "$benchmark_model"  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type "$benchmark_task_type"  \
  --num-prompts "$online_warmup_num_prompts" \
  --use-beam-search \
  --port "$run_port" \
  --request-rate "$online_warmup_request_rate" \
  --n "$beam_width")


output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model "$benchmark_model"  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type "$benchmark_task_type"  \
  --num-prompts "$online_perf_num_prompts" \
  --use-beam-search \
  --port "$run_port" \
  --request-rate "$online_perf_request_rate" \
  --n "$beam_width")

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
validate_metric "P99 TTFT" "$p99_ttft" "$p99_ttft_th" "le"
echo "Parsed P99 TPOT: ${p99_tpot} ms"

# Validate P99 tpot (Using awk to safely evaluate floating point comparison)
validate_metric "P99 TPOT" "$p99_tpot" "$p99_tpot_th" "le"

output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
--model "$benchmark_model" \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts "$online_accuracy_num_prompts" \
--use-beam-search \
--backend openai-chat \
--task-type "$benchmark_task_type" \
--endpoint /v1/chat/completions \
--port "$run_port" \
--n "$beam_width")

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
validate_metric "elapsed_time" "$elapsed_time_online" "$elapsed_time_online_th" "le"



exit 0
