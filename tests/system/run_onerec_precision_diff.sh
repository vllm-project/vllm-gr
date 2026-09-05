#!/usr/bin/env bash

# Compare upstream vLLM BeamSearch semantics with decode Constraint Tables on
# exactly the same full-catalog OneRec canary prompts.

set -euo pipefail

# Both readiness checks and benchmark requests are loopback traffic.  CI and
# developer hosts commonly export HTTP(S)_PROXY without NO_PROXY; make local
# serving deterministic instead of waiting for the proxy to time out.
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"

usage() {
  cat <<'EOF'
Usage: run_onerec_precision_diff.sh [options]

Required:
  --model PATH                 OneRec model directory or ID
  --dataset-path PATH          OpenOneRec benchmark_data directory
  --prepared-dir PATH          Directory for full catalog/triples data
  --output-dir PATH            Result directory

Options:
  --revision REV               Immutable model revision (recommended for remote models)
  --constrained-golden PATH    Blocking aggregate golden (default:
                               benchmarks/open_one_rec/manifests/
                               ci_precision_v1_constrained_golden.json)
  --golden-dir PATH            Reuse existing native BeamSearch golden results;
                               otherwise generate them in OUTPUT_DIR/golden
  --sample-manifest PATH       Default: benchmarks/open_one_rec/manifests/ci_precision_v1.json
  --tasks CSV                  Default: video,ad,product,label_cond,interactive
  --port N                     Default: 7012
  --beam-width N               Default: 32 (official OpenOneRec evaluation)
  --temperature F              Default: 0.0 (official native BeamSearch)
  --server-timeout N           Readiness timeout in seconds (default: 900)
  --request-timeout N          Per-request timeout in seconds (default: 600)
  --max-model-len N            Serving context length (default: 16384)
  --gpu-memory-utilization F   Default: 0.8
  --evaluation-mode MODE       sid|pid|both (default: both)
  --absolute-tolerance F       Diagnostic-only aggregate drop override (default: 0)
  --relative-tolerance F       Diagnostic-only relative drop override (default: 0)
  --skip-prepare               Reuse and validate an existing prepared manifest
  --prepare-force              Rebuild prepared catalog/triples
  --overwrite                  Replace an existing output directory
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
model=""
revision=""
golden_dir=""
constrained_golden="$repo_root/benchmarks/open_one_rec/manifests/ci_precision_v1_constrained_golden.json"
dataset_path=""
prepared_dir=""
output_dir=""
sample_manifest="$repo_root/benchmarks/open_one_rec/manifests/ci_precision_v1.json"
tasks_csv="video,ad,product,label_cond,interactive"
port="7012"
beam_width="32"
temperature="0.0"
server_timeout="900"
request_timeout="600"
max_model_len="16384"
gpu_memory_utilization="0.8"
evaluation_mode="both"
absolute_tolerance="0"
relative_tolerance="0"
skip_prepare=0
prepare_force=0
overwrite=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --revision) revision="$2"; shift 2 ;;
    --constrained-golden) constrained_golden="$2"; shift 2 ;;
    --golden-dir) golden_dir="$2"; shift 2 ;;
    --dataset-path) dataset_path="$2"; shift 2 ;;
    --prepared-dir) prepared_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --sample-manifest) sample_manifest="$2"; shift 2 ;;
    --tasks) tasks_csv="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --beam-width) beam_width="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --server-timeout) server_timeout="$2"; shift 2 ;;
    --request-timeout) request_timeout="$2"; shift 2 ;;
    --max-model-len) max_model_len="$2"; shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2"; shift 2 ;;
    --evaluation-mode) evaluation_mode="$2"; shift 2 ;;
    --absolute-tolerance) absolute_tolerance="$2"; shift 2 ;;
    --relative-tolerance) relative_tolerance="$2"; shift 2 ;;
    --skip-prepare) skip_prepare=1; shift ;;
    --prepare-force) prepare_force=1; shift ;;
    --overwrite) overwrite=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required_name in model dataset_path prepared_dir output_dir; do
  if [[ -z "${!required_name}" ]]; then
    echo "Missing required option: --${required_name//_/-}" >&2
    usage >&2
    exit 2
  fi
done
if [[ ! -d "$dataset_path" ]]; then
  echo "Dataset directory does not exist: $dataset_path" >&2
  exit 2
fi
if [[ ! -f "$sample_manifest" ]]; then
  echo "Sample manifest does not exist: $sample_manifest" >&2
  exit 2
fi
if [[ ! -f "$constrained_golden" ]]; then
  echo "Constrained golden does not exist: $constrained_golden" >&2
  exit 2
fi
if [[ -n "$golden_dir" ]]; then
  if [[ ! -d "$golden_dir" ]]; then
    echo "Golden result directory does not exist: $golden_dir" >&2
    exit 2
  fi
  golden_dir="$(cd -- "$golden_dir" && pwd)"
fi
if [[ -e "$output_dir" && "$overwrite" -ne 1 ]]; then
  echo "Output already exists: $output_dir; pass --overwrite to replace task files." >&2
  exit 2
fi

if [[ "$skip_prepare" -eq 1 ]]; then
  if [[ ! -f "$prepared_dir/manifest.json" ]]; then
    echo "Prepared manifest does not exist: $prepared_dir/manifest.json" >&2
    exit 2
  fi
else
  prepare_args=(
    --dataset-path "$dataset_path"
    --output-dir "$prepared_dir"
  )
  if [[ -n "$revision" ]]; then
    prepare_args+=(--revision "$revision")
  fi
  if [[ "$prepare_force" -eq 1 ]]; then
    prepare_args+=(--force)
  fi
  python -m benchmarks.open_one_rec.tools.prepare_decode_data "${prepare_args[@]}"
fi

mkdir -p \
  "$output_dir/golden" \
  "$output_dir/decode" \
  "$output_dir/diffs/native" \
  "$output_dir/logs"
IFS=',' read -r -a selected_tasks <<< "$tasks_csv"
if [[ "${#selected_tasks[@]}" -eq 0 ]]; then
  echo "--tasks must select at least one recommendation task." >&2
  exit 2
fi
for task in "${selected_tasks[@]}"; do
  case "$task" in
    video|ad|product|label_cond|interactive) ;;
    *) echo "Unsupported recommendation task: $task" >&2; exit 2 ;;
  esac
  if [[ -n "$golden_dir" && ! -f "$golden_dir/$task.jsonl" ]]; then
    echo "Golden result is missing for $task: $golden_dir/$task.jsonl" >&2
    exit 2
  fi
done

domain_tasks() {
  local domain="$1"
  local task
  for task in "${selected_tasks[@]}"; do
    case "$domain:$task" in
      video:video|video:ad|video:label_cond|video:interactive|product:product)
        printf '%s\n' "$task"
        ;;
    esac
  done
}

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT

wait_for_server() {
  local deadline=$((SECONDS + server_timeout))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" 2>/dev/null || true
      echo "OneRec server exited before readiness." >&2
      return 1
    fi
    if curl --noproxy '*' --fail --silent --show-error --max-time 3 \
        "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "OneRec server did not become ready within ${server_timeout}s." >&2
  return 1
}

run_decode_domain() {
  local domain="$1"
  local result_dir="$output_dir/decode"
  local catalog_path="$prepared_dir/$domain/constraint_triples.json"
  local log_path="$output_dir/logs/decode-${domain}.log"
  local implementation_revision
  local constraint_args=(--constraint-backend constraint_table --catalog-path "$catalog_path")
  local beam_server_args=(--beam-max-width "$beam_width")
  local execution_variant="graph-worker"
  local revision_args=()
  if [[ -n "$revision" ]]; then
    revision_args=(--revision "$revision")
  fi
  implementation_revision="$(git -C "$repo_root" rev-parse HEAD)"

  local -a tasks=()
  mapfile -t tasks < <(domain_tasks "$domain")
  if [[ "${#tasks[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "Starting decode server for $domain universe: $catalog_path"
  local -a server_command=(vllm-gr)
  "${server_command[@]}" serve \
    --port "$port" \
    --seed 123 \
    "${beam_server_args[@]}" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs "$((beam_width * 2))" \
    --max-num-seqs 2048 \
    --max-num-batched-tokens 16384 \
    --max-model-len "$max_model_len" \
    --scheduling-policy priority \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    "${revision_args[@]}" \
    "${constraint_args[@]}" \
    --attention-backend CUSTOM \
    "$model" > "$log_path" 2>&1 &
  server_pid=$!
  if ! wait_for_server; then
    tail -200 "$log_path" >&2 || true
    return 1
  fi

  local task
  for task in "${tasks[@]}"; do
    echo "Running decode/$task canary"
    local runner_args=(
      --execution online
      --task "$task"
      --model "$model"
      --implementation-revision "$implementation_revision"
      --execution-variant "$execution_variant"
      --dataset-path "$dataset_path"
      --prepared-manifest "$prepared_dir/manifest.json"
      --output-dir "$result_dir"
      --constraint-mode triples
      --sample-manifest "$sample_manifest"
      --beam-width "$beam_width"
      --temperature "$temperature"
      --evaluation-mode "$evaluation_mode"
      --host 127.0.0.1
      --port "$port"
      --request-timeout "$request_timeout"
      --overwrite
      --strict
    )
    if [[ -n "$revision" ]]; then
      runner_args+=(--revision "$revision")
    fi
    NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost" \
    no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost" \
      python -m benchmarks.open_one_rec.run_recommendation_regression \
        "${runner_args[@]}"
  done
  stop_server
}

# The golden calls the unmodified vLLM LLM.beam_search API without loading a
# catalog. One model instance covers all selected tasks.
if [[ -z "$golden_dir" ]]; then
  golden_dir="$output_dir/golden"
  golden_args=(
    --tasks "$tasks_csv"
    --model "$model"
    --dataset-path "$dataset_path"
    --prepared-manifest "$prepared_dir/manifest.json"
    --sample-manifest "$sample_manifest"
    --output-dir "$golden_dir"
    --beam-width "$beam_width"
    --max-tokens 3
    --temperature "$temperature"
    --batch-size 1
    --evaluation-mode "$evaluation_mode"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --overwrite
  )
  if [[ -n "$revision" ]]; then
    golden_args+=(--revision "$revision")
  fi
  # Keep the oracle independent of vllm-gr's general plugin and monkey
  # patches. An empty allow-list tells vLLM to load no general plugins in the
  # golden process or its EngineCore child.
  env VLLM_PLUGINS="" VLLM_USE_V2_MODEL_RUNNER="0" \
    python -m benchmarks.open_one_rec.generate_native_beam_golden \
      "${golden_args[@]}"
fi
run_decode_domain video
run_decode_domain product

case "$evaluation_mode" in
  sid)
    comparison_metrics=(
      pass@1 position1_pass@1 recall@1
      pass@32 position1_pass@32 recall@32
    )
    ;;
  pid)
    comparison_metrics=(
      pid_pass@1 pid_position1_pass@1 pid_recall@1
      pid_pass@32 pid_position1_pass@32 pid_recall@32
    )
    ;;
  both)
    comparison_metrics=(
      pass@1 position1_pass@1 recall@1
      pass@32 position1_pass@32 recall@32
      pid_pass@1 pid_position1_pass@1 pid_recall@1
      pid_pass@32 pid_position1_pass@32 pid_recall@32
    )
    ;;
  *)
    echo "Unsupported evaluation mode: $evaluation_mode" >&2
    exit 2
    ;;
esac
metric_args=()
for metric in "${comparison_metrics[@]}"; do
  metric_args+=(--metric "$metric")
done
native_diagnostic_status=0
for task in "${selected_tasks[@]}"; do
  if ! python -m benchmarks.open_one_rec.compare_recommendation_regression \
      --baseline "$golden_dir/$task.jsonl" \
      --candidate "$output_dir/decode/$task.jsonl" \
      --output "$output_dir/diffs/native/$task.json" \
      --policy no-regression \
      "${metric_args[@]}" \
      --absolute-tolerance "$absolute_tolerance" \
      --relative-tolerance "$relative_tolerance" > /dev/null; then
    native_diagnostic_status=1
  fi
done

if [[ "$native_diagnostic_status" -ne 0 ]]; then
  echo "WARNING: constrained results differ from the paired native reference; inspect $output_dir/diffs/native." >&2
fi

gate_task_args=()
for task in "${selected_tasks[@]}"; do
  gate_task_args+=(--task "$task")
done
if ! python -m benchmarks.open_one_rec.gate_constrained_golden \
    --golden "$constrained_golden" \
    --candidate-dir "$output_dir/decode" \
    --output "$output_dir/diffs/constrained-golden.json" \
    --evaluation-mode "$evaluation_mode" \
    "${gate_task_args[@]}" > /dev/null; then
  echo "OneRec constrained decode is below its versioned constrained golden;" \
    "inspect $output_dir/diffs/constrained-golden.json." >&2
  exit 1
fi
echo "OneRec constrained decode passes the versioned constrained-golden policy."
