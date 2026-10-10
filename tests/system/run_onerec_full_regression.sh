#!/usr/bin/env bash

set -euo pipefail

# Interpreter override: point at a venv with the repo-pinned vLLM build
# when the ambient "python" resolves to a different installation.
PYTHON="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: run_onerec_full_regression.sh [options]

Required:
  --model PATH                 OneRec model directory or ID
  --dataset-path PATH          OpenOneRec benchmark_data directory
  --prepared-dir PATH          Directory for catalog/triples/artifact data
  --output-dir PATH            Result directory

Options:
  --constraint-mode MODE       trie|triples|artifact (default: artifact)
  --execution MODE             offline|online (default: offline)
  --artifact-backend BACKEND   cuda|npu|cpu (default: cuda)
  --revision REV               Immutable remote model revision for artifact builds
  --implementation-revision REV
                               Git revision of the running inference server/code
  --sample-manifest PATH       Run the fixed IDs in this manifest instead of full data
  --baseline-dir PATH          Compare each result with PATH/<task>.jsonl
  --comparison-policy POLICY   exact|no-regression (default: exact)
  --tasks CSV                  Default: video,ad,product,label_cond,interactive
  --batch-size N               Default: 1 (persistent beam pool has one active session)
  --beam-width N               Default: 32 (official OpenOneRec evaluation)
  --temperature F              Default: 0.0 (official native BeamSearch)
  --gpu-memory-utilization F   Default: 0.7
  --log-every N                Default: 100
  --evaluation-mode MODE       sid|pid|both (default: both)
  --host HOST                  Online server host (default: 127.0.0.1)
  --port N                     Online server port (default: 8000)
  --endpoint PATH              Online endpoint (default: /v1/completions)
  --api-key KEY                Online API key (default: EMPTY)
  --request-timeout N          Online request timeout seconds (default: 600)
  --resume                     Resume existing JSONL files
  --overwrite                  Replace existing task results
  --prepare-force              Rebuild prepared data
EOF
}

model=""
dataset_path=""
prepared_dir=""
output_dir=""
constraint_mode="artifact"
execution="offline"
execution_variant=""
artifact_backend="cuda"
revision=""
implementation_revision=""
sample_manifest=""
baseline_dir=""
comparison_policy="exact"
tasks_csv="video,ad,product,label_cond,interactive"
batch_size="1"
beam_width="32"
temperature="0.0"
gpu_memory_utilization="0.7"
log_every="100"
evaluation_mode="both"
host="127.0.0.1"
port="8000"
endpoint="/v1/completions"
api_key="EMPTY"
request_timeout="600"
resume=0
overwrite=0
prepare_force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --dataset-path) dataset_path="$2"; shift 2 ;;
    --prepared-dir) prepared_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --constraint-mode) constraint_mode="$2"; shift 2 ;;
    --execution) execution="$2"; shift 2 ;;
    --execution-variant) execution_variant="$2"; shift 2 ;;
    --artifact-backend) artifact_backend="$2"; shift 2 ;;
    --revision) revision="$2"; shift 2 ;;
    --implementation-revision) implementation_revision="$2"; shift 2 ;;
    --sample-manifest) sample_manifest="$2"; shift 2 ;;
    --baseline-dir) baseline_dir="$2"; shift 2 ;;
    --comparison-policy) comparison_policy="$2"; shift 2 ;;
    --tasks) tasks_csv="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --beam-width) beam_width="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2"; shift 2 ;;
    --log-every) log_every="$2"; shift 2 ;;
    --evaluation-mode) evaluation_mode="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --endpoint) endpoint="$2"; shift 2 ;;
    --api-key) api_key="$2"; shift 2 ;;
    --request-timeout) request_timeout="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    --overwrite) overwrite=1; shift ;;
    --prepare-force) prepare_force=1; shift ;;
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

prepare_args=(
  --dataset-path "$dataset_path"
  --output-dir "$prepared_dir"
  --model "$model"
  --artifact-backend "$artifact_backend"
)
if [[ "$constraint_mode" == "artifact" ]]; then
  prepare_args+=(--build-artifacts)
fi
if [[ -n "$revision" ]]; then
  prepare_args+=(--revision "$revision")
fi
if [[ "$prepare_force" -eq 1 ]]; then
  prepare_args+=(--force)
fi

# Canonical Beam V1 offline runs get a stable variant label so gate goldens
# stay comparable; explicitly passed labels win.
default_execution_variant() {
  if [[ "$execution" == "offline" && "$constraint_mode" == "artifact" ]]; then
    echo "offline-v1-graph"
  else
    echo "default"
  fi
}

# Always validate the prepared representation against the current mappings.
# Existing catalogs/triples/artifacts are reused after their identities pass.
"${PYTHON}" -m benchmarks.open_one_rec.tools.prepare_decode_data "${prepare_args[@]}"

mkdir -p "$output_dir"
IFS=',' read -r -a tasks <<< "$tasks_csv"
if [[ "${#tasks[@]}" -eq 0 ]]; then
  echo "--tasks must select at least one recommendation task." >&2
  exit 2
fi
for task in "${tasks[@]}"; do
  case "$task" in
    video|ad|product|label_cond|interactive) ;;
    *) echo "Unsupported recommendation task: $task" >&2; exit 2 ;;
  esac
done
for task in "${tasks[@]}"; do
  run_args=(
    --task "$task"
    --model "$model"
    --dataset-path "$dataset_path"
    --prepared-manifest "$prepared_dir/manifest.json"
    --output-dir "$output_dir"
    --constraint-mode "$constraint_mode"
    --execution "$execution"
    --execution-variant "${execution_variant:-$(default_execution_variant)}"
    --artifact-backend "$artifact_backend"
    --batch-size "$batch_size"
    --beam-width "$beam_width"
    --temperature "$temperature"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --log-every "$log_every"
    --evaluation-mode "$evaluation_mode"
    --host "$host"
    --port "$port"
    --endpoint "$endpoint"
    --api-key "$api_key"
    --request-timeout "$request_timeout"
    --strict
  )
  if [[ -n "$sample_manifest" ]]; then
    run_args+=(--sample-manifest "$sample_manifest")
  fi
  if [[ -n "$revision" ]]; then
    run_args+=(--revision "$revision")
  fi
  if [[ -n "$implementation_revision" ]]; then
    run_args+=(--implementation-revision "$implementation_revision")
  fi
  if [[ "$resume" -eq 1 ]]; then
    run_args+=(--resume)
  fi
  if [[ "$overwrite" -eq 1 ]]; then
    run_args+=(--overwrite)
  fi

  NO_PROXY="${NO_PROXY:+$NO_PROXY,}$host,127.0.0.1,localhost" \
  no_proxy="${no_proxy:+$no_proxy,}$host,127.0.0.1,localhost" \
    "${PYTHON}" -m benchmarks.open_one_rec.run_recommendation_regression "${run_args[@]}"

  if [[ -n "$baseline_dir" ]]; then
    "${PYTHON}" -m benchmarks.open_one_rec.compare_recommendation_regression \
      --baseline "$baseline_dir/$task.jsonl" \
      --candidate "$output_dir/$task.jsonl" \
      --output "$output_dir/$task.diff.json" \
      --policy "$comparison_policy"
  fi
done
