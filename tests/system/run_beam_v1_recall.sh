#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
baseline="${V1_RECALL_BASELINE_FILE:-}"
if [[ -n "$baseline" && ( ! -f "$baseline" || ! -r "$baseline" ) ]]; then
  echo "V1_RECALL_BASELINE_FILE must point to a readable regular file: $baseline" >&2
  exit 1
fi

dataset="${V1_RECALL_DATASET_PATH:-$repo_root/OpenOneRec/benchmark_data}"
output_dir="${V1_RECALL_OUTPUT_DIR:-$repo_root/v1_recall_results}"
model="${VLLM_GR_TEST_MODEL:-OpenOneRec/OneRec-1.7B}"
revision="${V1_RECALL_MODEL_REVISION:-cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b}"
triples="${V1_RECALL_TRIPLES_PATH:-$repo_root/generated_data/recall_constraints/sid2pid_triples.json}"
triples_iid="${V1_RECALL_TRIPLES_IID_PATH:-$repo_root/generated_data/recall_constraints/sid2iid_triples.json}"
for path in "$dataset/sid2pid.json" "$dataset/sid2iid.json" "$triples" "$triples_iid"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing OneRec accuracy input: $path" >&2
    exit 1
  fi
done

mkdir -p "$output_dir"
artifact_dir="$output_dir/artifacts"
mkdir -p "$artifact_dir"
git rev-parse HEAD > "$output_dir/commit.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
    > "$output_dir/gpu.csv" || true
fi

if [[ -d "$model" ]]; then
  if [[ -n "$baseline" ]]; then
    echo "A calibrated Recall gate requires the pinned model snapshot" >&2
    exit 1
  fi
  model_path="$model"
  model_revision_args=()
else
  model_path="$(python - "$model" "$revision" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
)"
  model_revision_args=(--model-revision "$revision")
fi

total_start=$SECONDS
repeat_count="${V1_RECALL_REPEATS:-1}"
if [[ ! "$repeat_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "V1_RECALL_REPEATS must be a positive integer" >&2
  exit 1
fi
if [[ -n "$baseline" && "$repeat_count" -eq 2 ]]; then
  echo "A configured Recall gate requires one screening run or at least three runs" >&2
  exit 1
fi
run_group() {
  local run_dir="$1"
  local group="$2"
  local group_triples="$3"
  shift 3
  local group_dir="$run_dir/groups/$group"
  mkdir -p "$group_dir"
  echo "Running $group: tasks=$*, B=4, 1000 samples/task, beam width 128"
  local group_start=$SECONDS
  # Match the calibrated runner even when the application or parent defaults to V2.
  VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python -m benchmarks.beam_search.measure_onerec_recall \
      --model "$model_path" \
      "${model_revision_args[@]}" \
      --dataset-path "$dataset" \
      --catalog-path "$group_triples" \
      --constraint-artifact "$artifact_dir/$group.safetensors" \
      --task-types "$@" \
      --num-prompts 1000 \
      --batch-size 4 \
      --n 128 \
      --temperature 0.6 \
      --result-dir "$run_dir" 2>&1 | tee "$group_dir/run.log"
  if [[ "${V1_RECALL_DIAGNOSTICS:-0}" == 1 ]]; then
    python -m benchmarks.beam_search.recall_diagnostics \
      --run-log "$group_dir/run.log" --output "$group_dir/attention_diagnostics.json"
  fi
  echo "$((SECONDS - group_start))" > "$group_dir/wall_seconds.txt"
}

# Each repeat starts fresh engines, so its wall time includes graph capture,
# model loading, inference and evaluation. The sample IDs remain fixed.
if [[ -e "$output_dir/runs" ]]; then
  echo "Refusing to mix results with an existing repeat directory: $output_dir/runs" >&2
  exit 1
fi
run_once() {
  local run="$1"
  local run_dir="$output_dir/runs/$(printf 'run-%02d' "$run")"
  mkdir -p "$run_dir"
  echo "Starting OneRec accuracy gate repeat $run/$repeat_count"
  local run_start=$SECONDS
  # Four tasks share the SID table; product needs the distinct SID-to-IID table.
  run_group "$run_dir" sid2pid "$triples" video ad label_cond interactive
  run_group "$run_dir" sid2iid "$triples_iid" product
  echo "$((SECONDS - run_start))" > "$run_dir/total_wall_seconds.txt"
}

summarize_runs() {
  local count="$1"
  echo "$((SECONDS - total_start))" > "$output_dir/total_wall_seconds.txt"
  python -m benchmarks.beam_search.compare_recall_baseline \
    --result-dir "$output_dir" --repeats "$count"
}

# The normal CI path runs once. A below-minimum task triggers two fresh
# confirmations; all three must be below the same task's historical minimum.
if [[ -n "$baseline" && "$repeat_count" -eq 1 ]]; then
  run_once 1
  summarize_runs 1
  needs_confirmation="$(python - "$output_dir/b4_repeat_summary.json" <<'PY'
import json
import sys

print("yes" if json.load(open(sys.argv[1], encoding="utf-8"))["confirmation_required"] else "no")
PY
)"
  if [[ "$needs_confirmation" == yes ]]; then
    repeat_count=3
    run_once 2
    run_once 3
    summarize_runs 3
  fi
else
  for ((run = 1; run <= repeat_count; run++)); do
    run_once "$run"
  done
  summarize_runs "$repeat_count"
fi
