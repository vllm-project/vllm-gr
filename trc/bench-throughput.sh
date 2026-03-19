source "$(dirname "$0")/get_index.sh"
index=$(get_index)
run_port=$((index + 8000))
echo "connect client on port $run_port"
python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model OpenOneRec/OneRec-1.7B  \
  --dataset-name  onerec \
  --dataset-path /home/yoni/OpenOneRec/benchmark_data \
  --task-type video  \
  --num-prompts 100 \
  --use-beam-search \
  --port $run_port \
  --n 128 
