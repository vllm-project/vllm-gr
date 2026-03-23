#!/bin/bash

# Ensure we clean up the background server process if the script exits early
trap 'kill -9 $SERVER_PID 2>/dev/null' EXIT

source "$(dirname "$0")/get_index.sh"

idle_gpu=$(get_gpu 4)
if [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 4MiB memory, and error-free."
    exit 1
fi

index=$(get_index)
cuda_dev=$idle_gpu
run_port=$((index + 8000))
debug_port=$((index + 5673))

echo "Starting server on port $run_port using CUDA device $cuda_dev debug port $debug_port"

CUDA_VISIBLE_DEVICES=$cuda_dev \
PYTHONPATH=trc:$PYTHONPATH \
vllm serve --gr \
    --port $run_port \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 2048 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --catalog-path /home/yoni/OpenOneRec/benchmark_data/catalog.json \
    OpenOneRec/OneRec-1.7B &

SERVER_PID=$!

echo "Waiting for server to be ready on port $run_port..."
# Check health/models endpoint until it responds
timeout 120 bash -c "
  until curl -s http://localhost:${run_port}/v1/models > /dev/null; do
    sleep 2
  done
"
if [ $? -ne 0 ]; then
  echo "Server failed to start within the timeout limit."
  exit 1
fi
echo "Server is ready! Running benchmark..."

output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model OpenOneRec/OneRec-1.7B  \
  --dataset-name onerec \
  --dataset-path /home/yoni/OpenOneRec/benchmark_data \
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

kill -9 $SERVER_PID 2>/dev/null
