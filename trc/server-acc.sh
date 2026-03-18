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
     OpenOneRec/OneRec-1.7B
