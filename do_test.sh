#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./do_test.sh <result-tag> [load|serial]

Examples:
  ./do_test.sh 0.14.1
  ./do_test.sh 0.22.1rc1
  ./do_test.sh 0.22.1rc1-serial serial

Run this script from the vLLM-GR 0.22.1rc1 client container after the
server under test is ready. Environment variables can override all paths,
the server address, and benchmark sizes.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

tag="$1"
mode="${2:-load}"
case "$mode" in
    load)
        default_perf_request_rate="0.5"
        default_perf_max_concurrency=""
        default_run_accuracy="1"
        ;;
    serial)
        default_perf_request_rate="inf"
        default_perf_max_concurrency="1"
        default_run_accuracy="0"
        ;;
    *)
        echo "Error: mode must be 'load' or 'serial', got: $mode" >&2
        usage >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_path="${ONEREC_MODEL_PATH:-/home/z00469465/OneRec/OneRec-1.7B}"
dataset_path="${ONEREC_DATASET_PATH:-/home/z00469465/OneRec/OpenOneRec-RecIF/benchmark_data}"
catalog_path="${ONEREC_CATALOG_PATH:-/home/z00469465/OneRec/generated_data/video_catalog.json}"

server_host="${SERVER_HOST:-127.0.0.1}"
server_port="${PORT:-7000}"
server_start_timeout="${SERVER_START_TIMEOUT:-600}"

beam_width="${BEAM_WIDTH:-128}"
warmup_num_prompts="${ONLINE_WARMUP_NUM_PROMPTS:-20}"
warmup_request_rate="${ONLINE_WARMUP_REQUEST_RATE:-1}"
perf_num_prompts="${ONLINE_PERF_NUM_PROMPTS:-100}"
perf_num_warmups="${ONLINE_PERF_NUM_WARMUPS:-10}"
perf_request_rate="${ONLINE_PERF_REQUEST_RATE:-$default_perf_request_rate}"
perf_max_concurrency="${ONLINE_PERF_MAX_CONCURRENCY:-$default_perf_max_concurrency}"
accuracy_num_prompts="${ONLINE_ACCURACY_NUM_PROMPTS:-100}"
run_accuracy="${RUN_ACCURACY:-$default_run_accuracy}"

if [[ "$run_accuracy" != "0" && "$run_accuracy" != "1" ]]; then
    echo "Error: RUN_ACCURACY must be 0 or 1, got: $run_accuracy" >&2
    exit 2
fi

result_root="${RESULT_ROOT:-/home/z00469465/OneRec/results/online_compare}"
result_dir="${result_root}/${tag}"
mkdir -p "$result_dir"

for required_path in "$model_path" "$dataset_path" "$catalog_path"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path does not exist: $required_path" >&2
        exit 1
    fi
done

for required_command in curl timeout python; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "Error: required command is unavailable: $required_command" >&2
        exit 1
    fi
done

echo "============================================================"
echo "Online benchmark configuration"
echo "============================================================"
echo "tag:                       $tag"
echo "mode:                      $mode"
echo "server:                    http://${server_host}:${server_port}"
echo "model:                     $model_path"
echo "dataset:                   $dataset_path"
echo "catalog:                   $catalog_path"
echo "beam width:                $beam_width"
echo "warmup prompts:            $warmup_num_prompts"
echo "performance prompts:       $perf_num_prompts"
echo "performance warmups:       $perf_num_warmups"
echo "performance request rate:  $perf_request_rate"
echo "performance concurrency:   ${perf_max_concurrency:-unlimited}"
echo "accuracy prompts:          $accuracy_num_prompts"
echo "run accuracy:              $run_accuracy"
echo "result directory:          $result_dir"
echo "============================================================"

echo "Waiting for the server..."
if ! timeout "$server_start_timeout" bash -c "
    until curl -sf http://${server_host}:${server_port}/v1/models >/dev/null; do
        sleep 2
    done
"; then
    echo "Error: server did not become ready within ${server_start_timeout}s" >&2
    exit 1
fi

echo "Server is ready."
curl -sf "http://${server_host}:${server_port}/v1/models" \
    | python -m json.tool \
    | tee "$result_dir/models.json"

echo
echo "Running diverse-prompt warmup..."
python -m benchmarks.open_one_rec.one_rec_main \
    bench serve \
    --endpoint /v1/chat/completions \
    --backend openai-chat \
    --model "$model_path" \
    --dataset-name onerec \
    --dataset-path "$dataset_path" \
    --task-type video \
    --num-prompts "$warmup_num_prompts" \
    --disable-shuffle \
    --use-beam-search \
    --host "$server_host" \
    --port "$server_port" \
    --request-rate "$warmup_request_rate" \
    --n "$beam_width" \
    2>&1 | tee "$result_dir/warmup.log"

perf_concurrency_args=()
if [[ -n "$perf_max_concurrency" ]]; then
    perf_concurrency_args=(--max-concurrency "$perf_max_concurrency")
fi

echo
echo "Running online performance benchmark..."
python -m benchmarks.open_one_rec.one_rec_main \
    bench serve \
    --endpoint /v1/chat/completions \
    --backend openai-chat \
    --model "$model_path" \
    --dataset-name onerec \
    --dataset-path "$dataset_path" \
    --task-type video \
    --num-prompts "$perf_num_prompts" \
    --num-warmups "$perf_num_warmups" \
    --disable-shuffle \
    --use-beam-search \
    --host "$server_host" \
    --port "$server_port" \
    --request-rate "$perf_request_rate" \
    "${perf_concurrency_args[@]}" \
    --n "$beam_width" \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir "$result_dir" \
    --result-filename performance-detail.json \
    2>&1 | tee "$result_dir/performance.log"

if [[ "$run_accuracy" == "1" ]]; then
    echo
    echo "Running online accuracy benchmark..."
    python -m benchmarks.open_one_rec.one_rec_acc_test \
        --model "$model_path" \
        --dataset-path "$dataset_path" \
        --dataset-name onerec \
        --num-prompts "$accuracy_num_prompts" \
        --use-beam-search \
        --backend openai-chat \
        --task-type video \
        --endpoint /v1/chat/completions \
        --host "$server_host" \
        --port "$server_port" \
        --catalog-path "$catalog_path" \
        --n "$beam_width" \
        --result-dir "$result_dir/accuracy" \
        2>&1 | tee "$result_dir/accuracy.log"
fi

echo
echo "============================================================"
echo "Online test completed: $tag"
echo "Results: $result_dir"
echo "============================================================"

if [[ -f "$result_dir/accuracy.log" ]]; then
    echo "Accuracy summary:"
    grep -E \
        '"pass@1"|"pass@32"|"recall@32"|"elapsed_time"' \
        "$result_dir/accuracy.log" || true
    echo
fi

echo "Performance summary:"
grep -E \
    'Successful requests|Failed requests|Request throughput|Mean TTFT|Median TTFT|P[0-9]+ TTFT|P[0-9]+ TPOT|Avg Prefill|Avg Decode|Total Beam' \
    "$result_dir/performance.log" || true
