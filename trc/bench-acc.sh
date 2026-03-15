source "$(dirname "$0")/get_index.sh"
index=$(get_index)
run_port=$((index + 8000))
echo "connect client on port $run_port"
python -m benchmarks.open_one_rec.one_rec_acc_test \
--model OpenOneRec/OneRec-1.7B \
--dataset-path /data/datasets/OpenOneRec/benchmark_data \
--dataset-name  onerec \
--num-prompts 100 \
--use-beam-search \
--backend openai-chat \
--task-type video \
--endpoint /v1/chat/completions \
--port $run_port \
--n 128


