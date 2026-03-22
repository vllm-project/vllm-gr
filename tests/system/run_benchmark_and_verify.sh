#!/bin/bash
run_port=7000
timeout=600

# Ensure we clean up the background server process if the script exits early
trap 'kill -9 $SERVER_PID 2>/dev/null' EXIT

vllm serve --gr \
    --port $run_port \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 2048 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --catalog-path "$CATALOG_PATH" \
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

if [ -z "$p99_ttft" ]; then
    echo "Error: Could not extract 'P99 TTFT (ms)' from the output."
    exit 1
fi

echo "Parsed P99 TTFT: ${p99_ttft} ms"

# Validate P99 < 30000 (Using awk to safely evaluate floating point comparison)
result=$(awk -v val="$p99_ttft" -v threshold="30000" 'BEGIN { print (val < threshold) ? "pass" : "fail" }')

if [ "$result" == "pass" ]; then
    echo "SUCCESS: P99 TTFT ($p99_ttft ms) is less than 30000 ms."
    exit 0
else
    echo "FAILURE: P99 TTFT ($p99_ttft ms) is greater than or equal to 30000 ms."
    exit 1
fi
