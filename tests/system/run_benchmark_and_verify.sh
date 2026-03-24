#!/bin/bash
run_port=7000
timeout=600
p99_ttft_th=30000
p99_tpot_th=1.0
pass_1_th=0.0496
pass_32_th=0.1710
recall_32_th=0.0272
# Ensure we clean up the background server process if the script exits early
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; sleep 2; pkill -P "$SERVER_PID" 2>/dev/null || true; fi' EXIT

vllm serve --gr \
    --port $run_port \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 2048 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --catalog-path "$CATALOG_PATH" \
    --attention-backend CUSTOM \
    OpenOneRec/OneRec-1.7B &

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

output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model OpenOneRec/OneRec-1.7B  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video  \
  --num-prompts 100 \
  --use-beam-search \
  --port $run_port \
  --n 128)

echo "$output"


# Extract P99 TTFT
p99_ttft=$(echo "$output" | grep "P99 TTFT (ms):" | awk '{print $4}')
p99_tpot=$(echo "$output" | grep "P99 TPOT (ms):" | awk '{print $4}')

if [ -z "$p99_ttft" ]; then
    echo "Error: Could not extract 'P99 TTFT (ms)' from the output."
    exit 1
fi

echo "Parsed P99 TTFT: ${p99_ttft} ms"

# Validate P99 ttft (Using awk to safely evaluate floating point comparison)
result=$(awk -v val="$p99_ttft" -v threshold="$p99_ttft_th" 'BEGIN { print (val < threshold) ? "pass" : "fail" }')

if [ "$result" == "pass" ]; then
    echo "SUCCESS: P99 TTFT ($p99_ttft ms) is less than $p99_ttft_th ms."
else
    echo "FAILURE: P99 TTFT ($p99_ttft ms) is greater than or equal to 30000 ms."
    exit 1
fi

if [ -z "$p99_tpot" ]; then
    echo "Error: Could not extract 'P99 TPOT (ms)' from the output."
    exit 1
fi

echo "Parsed P99 TPOT: ${p99_tpot} ms"

# Validate P99 tpot (Using awk to safely evaluate floating point comparison)
result=$(awk -v val="$p99_tpot" -v threshold="$p99_tpot_th" 'BEGIN { print (val < threshold) ? "pass" : "fail" }')

if [ "$result" == "pass" ]; then
    echo "SUCCESS: P99 TPOT ($p99_tpot ms) is less than $p99_tpot_th ms."
else
    echo "FAILURE: P99 TPOT ($p99_tpot ms) is greater than or equal to $p99_tpot_th ms."
    exit 1
fi

output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
--model OpenOneRec/OneRec-1.7B \
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
pass_1=$(echo "$output" | grep -E '^[[:space:]]*"pass@1":' | tail -1 | awk -F':' '{print $2}' | tr -d ' ,')
pass_32=$(echo "$output" | grep -E '^[[:space:]]*"pass@32":' | tail -1 | awk -F':' '{print $2}' | tr -d ' ,')
recall_32=$(echo "$output" | grep -E '^[[:space:]]*"recall@32":' | tail -1 | awk -F':' '{print $2}' | tr -d ' ,')

if [ -z "$pass_1" ] || [ -z "$pass_32" ] || [ -z "$recall_32" ]; then
    echo "Error: Could not extract accuracy metrics from the output."
    exit 1
fi

echo "Parsed pass@1: $pass_1"
echo "Parsed pass@32: $pass_32"
echo "Parsed recall@32: $recall_32"

# Validate pass@1
result=$(awk -v val="$pass_1" -v threshold="$pass_1_th" 'BEGIN { print (val >= threshold) ? "pass" : "fail" }')
if [ "$result" == "pass" ]; then
    echo "SUCCESS: pass@1 ($pass_1) is greater than or equal to $pass_1_th."
else
    echo "FAILURE: pass@1 ($pass_1) is less than $pass_1_th."
    exit 1
fi

# Validate pass@32
result=$(awk -v val="$pass_32" -v threshold="$pass_32_th" 'BEGIN { print (val >= threshold) ? "pass" : "fail" }')
if [ "$result" == "pass" ]; then
    echo "SUCCESS: pass@32 ($pass_32) is greater than or equal to $pass_32_th."
else
    echo "FAILURE: pass@32 ($pass_32) is less than $pass_32_th."
    exit 1
fi

# Validate recall@32
result=$(awk -v val="$recall_32" -v threshold="$recall_32_th" 'BEGIN { print (val >= threshold) ? "pass" : "fail" }')
if [ "$result" == "pass" ]; then
    echo "SUCCESS: recall@32 ($recall_32) is greater than or equal to $recall_32_th."
else
    echo "FAILURE: recall@32 ($recall_32) is less than $recall_32_th."
    exit 1
fi

exit 0
