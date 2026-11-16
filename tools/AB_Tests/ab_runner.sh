#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

set -Eeuo pipefail

# Server-side AB benchmark executor. Runs the offline beam-search benchmark
# (run_offline_benchmark.py + generate_summary.py) for baseline and head in an
# A-B-B-A schedule per scenario, then emits a comparison report with
# compare_ab.py. Invoked over ssh by ab_benchmark.sh; do not run by hand.
#
# A-B-B-A (baseline → head → head → baseline) cancels first-order ordering
# effects (cache warming, GPU thermal drift) by measuring each side twice, once
# early and once late. A single failed run does not abort the whole flow: that
# scenario is skipped by compare, but the rest still produce reports.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
config_file="${BENCHMARK_CONFIG:-$script_dir/benchmark.env}"
if [[ -f "$config_file" ]]; then
  # shellcheck disable=SC1090
  source "$config_file"
fi

project_dir="${PROJECT_DIR:-$(cd -- "$script_dir/../.." && pwd -P)}"
container="${CONTAINER_NAME:-vllm-gr-benchmark-gpu1}"
gpu_index="${GPU_INDEX:-1}"
host_name="${HOST_NAME:-$(hostname -s)}"
container_project_dir="${CONTAINER_PROJECT_DIR:-/opt/vllm-gr}"
container_data_dir="${CONTAINER_DATA_DIR:-$container_project_dir/data}"
model_id="${MODEL_ID:-OpenOneRec/OneRec-1.7B}"
hf_endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
num_prompts="${NUM_PROMPTS:-100}"
warmup_requests="${WARMUP_REQUESTS:-10}"
lightweight_timing="${LIGHTWEIGHT_TIMING:-0}"
gpu_idle_limit_mib="${GPU_IDLE_LIMIT_MIB:-1024}"

pr=""
run_name=""
base_sha=""
head_sha=""
scenario_specs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pr) pr="$2"; shift 2 ;;
    --run) run_name="$2"; shift 2 ;;
    --base) base_sha="$2"; shift 2 ;;
    --head) head_sha="$2"; shift 2 ;;
    --scenario) scenario_specs+=("$2"); shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$head_sha" ]]; then
  echo "usage: ab_runner.sh (--pr <N> | --run <name>) --head <ref-or-sha> [--base <ref-or-sha>] [--scenario beam:input ...]" >&2
  exit 2
fi
if [[ ${#scenario_specs[@]} -eq 0 ]]; then
  scenario_specs=("128:1024")
fi

container_gpu_healthy() {
  timeout 10 docker exec "$container" \
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader >/dev/null 2>&1 &&
    timeout 20 docker exec "$container" python3 -c \
      'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)' \
      >/dev/null 2>&1
}

# ---- container / GPU preflight ----
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]]; then
  echo "container $container is not running" >&2
  exit 2
fi
if docker exec "$container" pgrep -f "vllm-gr serve" >/dev/null; then
  echo "an existing vllm-gr service is running; skipping" >&2
  exit 75
fi
if docker exec "$container" pgrep -f "run_offline_benchmark.py" >/dev/null; then
  echo "an existing offline benchmark is running; skipping" >&2
  exit 75
fi
if ! container_gpu_healthy; then
  echo "container $container cannot access CUDA/NVML" >&2
  exit 2
fi
gpu_used="$(nvidia-smi --id="$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( gpu_used > gpu_idle_limit_mib )); then
  echo "GPU $gpu_index uses ${gpu_used} MiB (> ${gpu_idle_limit_mib}); skipping" >&2
  exit 75
fi

# ---- worktree must be a clean decode_graph ----
if [[ "$(git -C "$project_dir" branch --show-current)" != "decode_graph" ]]; then
  echo "expected decode_graph branch in $project_dir" >&2
  exit 2
fi
if [[ -n "$(git -C "$project_dir" status --porcelain --untracked-files=no)" ]]; then
  echo "benchmark worktree has tracked changes" >&2
  exit 2
fi

# ---- fetch head / base ----
git -C "$project_dir" fetch origin decode_graph
if [[ -n "$pr" ]]; then
  git -C "$project_dir" fetch origin "pull/$pr/head"
fi

# ---- resolve head to a commit first (the merge base below needs it) ----
if ! git -C "$project_dir" rev-parse --verify "$head_sha^{commit}" >/dev/null 2>&1; then
  if ! git -C "$project_dir" fetch origin "$head_sha" >/dev/null 2>&1; then
    echo "cannot resolve ref: $head_sha" >&2
    exit 2
  fi
fi
head_sha="$(git -C "$project_dir" rev-parse "$head_sha^{commit}")"

# ---- baseline: merge base of head and decode_graph, not the moving tip ----
# baseRefOid is the decode_graph tip, which keeps advancing and would drag in
# unrelated merged changes. Compare the frozen head against its merge base to
# isolate the PR's own effect. An explicit --base still overrides this.
if [[ -z "$base_sha" ]]; then
  base_sha="$(git -C "$project_dir" merge-base origin/decode_graph "$head_sha")"
fi

# ---- resolve base to a commit ----
if ! git -C "$project_dir" rev-parse --verify "$base_sha^{commit}" >/dev/null 2>&1; then
  if ! git -C "$project_dir" fetch origin "$base_sha" >/dev/null 2>&1; then
    echo "cannot resolve ref: $base_sha" >&2
    exit 2
  fi
fi
base_sha="$(git -C "$project_dir" rev-parse "$base_sha^{commit}")"

# ---- determine result-dir name and report label ----
if [[ -z "$run_name" ]]; then
  run_name="${pr:-${head_sha:0:8}}"
fi
run_name="${run_name//\//-}"
if [[ -n "$pr" ]]; then
  report_label="PR #$pr"
else
  # --run is now a unique invocation id, not a friendly label; use the head SHA.
  report_label="${head_sha:0:8}"
fi

echo "AB: pr=$pr run=$run_name base=$base_sha head=$head_sha scenarios=${scenario_specs[*]}"

# ---- stage the benchmark engine into git-ignored results/ab/.tooling ----
# Once tools/AB_Tests is committed to the repo, `git checkout` to an old
# base/head that predates it would delete those tracked scripts mid-run.
# Stage the engine and compare_ab.py into the git-ignored results/ab/.tooling/
# and run everything from there, so checkout never touches it.
tooling_host="$project_dir/results/ab/.tooling"
tooling_dir="$container_project_dir/results/ab/.tooling"
mkdir -p "$tooling_host/instrumentation" "$tooling_host/constraints"
cp -f "$script_dir/run_offline_benchmark.py" \
      "$script_dir/generate_summary.py" \
      "$script_dir/download_onerec_dataset.py" \
      "$script_dir/compare_ab.py" \
      "$tooling_host/"
cp -f "$script_dir/instrumentation/sitecustomize.py" \
      "$script_dir/instrumentation/vllm_gr_lightweight_timing.py" \
      "$tooling_host/instrumentation/"
# The constraint catalog is committed under tools/AB_Tests/constraints; stage it
# too, or `git checkout` to an old commit would delete it mid-run.
catalog_host="${CATALOG_HOST:-$script_dir/constraints/video_constraint_triples.json}"
cp -f "$catalog_host" "$tooling_host/constraints/video_constraint_triples.json"
# Container-side catalog path; override via CATALOG_PATH in benchmark.env.
catalog_path="${CATALOG_PATH:-$tooling_dir/constraints/video_constraint_triples.json}"

# ---- ensure the dataset is present ----
docker exec "$container" python3 "$tooling_dir/download_onerec_dataset.py" \
  --data-dir "$container_data_dir" --tasks video --verify-only >/dev/null

image="$(docker inspect -f '{{.Config.Image}}' "$container")"
ld_library_path=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

restore_branch() {
  git -C "$project_dir" checkout --quiet decode_graph >/dev/null 2>&1 || true
}
trap restore_branch EXIT INT TERM

run_commit() {
  local sha="$1" side="$2" beam_width="$3" input_length="$4" run_label="$5"
  local short="${sha:0:8}"
  local scenario_id="ab-${run_name}-${side}-${short}-bw${beam_width}-in${input_length}-${run_label}"
  local scenario_dir="$project_dir/results/ab/$run_name/$side/bw${beam_width}-in${input_length}"
  local container_scenario_dir="$container_project_dir/results/ab/$run_name/$side/bw${beam_width}-in${input_length}"
  mkdir -p "$scenario_dir"
  mkdir -p "$scenario_dir/cpu-timing"
  rm -f "$scenario_dir/cpu-timing"/cpu-timing-*.json

  # `run_commit` is invoked as the left of `|| failed=1`, which disables
  # errexit inside the function — so checkout failures must be propagated
  # explicitly, or the sample would silently run the previous commit while
  # still being stamped with `$sha`.
  if ! git -C "$project_dir" checkout --quiet --detach "$sha"; then
    echo "[$side $short $run_label] checkout $sha FAILED" >&2
    return 1
  fi
  if [[ "$(git -C "$project_dir" rev-parse HEAD)" != "$sha" ]]; then
    echo "[$side $short $run_label] checkout landed on $(git -C "$project_dir" rev-parse HEAD), not $sha" >&2
    return 1
  fi
  echo "[$side $short $run_label] beam=$beam_width input=$input_length at $(date --iso-8601=seconds)"

  timing_env=(-e PYTHONPATH="$tooling_dir/instrumentation:$container_project_dir")
  if [[ "$lightweight_timing" == 1 ]]; then
    timing_env+=(
      -e VLLM_GR_LIGHTWEIGHT_TIMING=1
      -e VLLM_GR_LIGHTWEIGHT_TIMING_DIR="$container_scenario_dir/cpu-timing"
    )
  fi
  if ! docker exec -w "$container_project_dir" "${timing_env[@]}" \
    -e HF_ENDPOINT="$hf_endpoint" -e HF_HUB_DISABLE_XET=1 \
    -e LD_LIBRARY_PATH="$ld_library_path" \
    "$container" python3 "$tooling_dir/run_offline_benchmark.py" \
    --output "$container_scenario_dir/raw-result-${run_label}.json" \
    --data-dir "$container_data_dir" \
    --catalog "$catalog_path" \
    --model "$model_id" \
    --num-prompts "$num_prompts" \
    --warmup-requests "$warmup_requests" \
    --beam-width "$beam_width" \
    --input-length "$input_length" \
    >"$scenario_dir/benchmark-${run_label}.log" 2>&1; then
    echo "[$side $short $run_label] benchmark FAILED; see $scenario_dir/benchmark-${run_label}.log" >&2
    return 1
  fi

  summary_args=(
    python3 "$tooling_dir/generate_summary.py"
    --raw-result "$container_scenario_dir/raw-result-${run_label}.json"
    --dataset-manifest "$container_data_dir/.onerec-manifest.json"
    --output "$container_scenario_dir/vllm-gr-summary-${run_label}.json"
    --run-id "$scenario_id"
    --warmup-requests "$warmup_requests"
    --beam-width "$beam_width"
    --input-length "$input_length"
    --cpu-timing-dir "$container_scenario_dir/cpu-timing"
    --execution-mode offline
    --host-name "$host_name"
    --container-name "$container"
    --git-sha "$sha"
    --git-subject "$(git -C "$project_dir" log -1 --format=%s "$sha")"
    --git-branch "decode_graph"
    --container-image "$image"
  )
  if ! docker exec -e LD_LIBRARY_PATH="$ld_library_path" "$container" "${summary_args[@]}"; then
    echo "[$side $short $run_label] summary generation FAILED" >&2
    return 1
  fi
  return 0
}

# ---- A-B-B-A per scenario ----
# Clear any results left over from a previous run of this name. Without this, a
# re-run whose attempts all fail would compare the old summaries and emit a bogus
# "No regressions" report.
rm -rf "$project_dir/results/ab/$run_name"
failed=0
for spec in "${scenario_specs[@]}"; do
  IFS=: read -r beam_width input_length <<<"$spec"
  case "$beam_width" in
    64|128|256) ;;
    *) echo "unsupported beam width $beam_width; allowed: 64,128,256" >&2; exit 2 ;;
  esac
  run_commit "$base_sha" baseline "$beam_width" "$input_length" a1 || failed=1
  run_commit "$head_sha" head     "$beam_width" "$input_length" b1 || failed=1
  run_commit "$head_sha" head     "$beam_width" "$input_length" b2 || failed=1
  run_commit "$base_sha" baseline "$beam_width" "$input_length" a2 || failed=1
done

# A single failed benchmark attempt must fail the whole run rather than emit a
# partial or stale report.
if [[ "$failed" -ne 0 ]]; then
  echo "AB: one or more benchmark runs failed; no report generated" >&2
  exit 1
fi

# ---- emit one report per scenario (only when all four runs produced a summary) ----
for spec in "${scenario_specs[@]}"; do
  IFS=: read -r beam_width input_length <<<"$spec"
  dir="$project_dir/results/ab/$run_name"
  base_a1="$dir/baseline/bw${beam_width}-in${input_length}/vllm-gr-summary-a1.json"
  base_a2="$dir/baseline/bw${beam_width}-in${input_length}/vllm-gr-summary-a2.json"
  head_b1="$dir/head/bw${beam_width}-in${input_length}/vllm-gr-summary-b1.json"
  head_b2="$dir/head/bw${beam_width}-in${input_length}/vllm-gr-summary-b2.json"
  if [[ -f "$base_a1" && -f "$base_a2" && -f "$head_b1" && -f "$head_b2" ]]; then
    python3 "$tooling_host/compare_ab.py" \
      --base "$base_a1" --base "$base_a2" \
      --head "$head_b1" --head "$head_b2" \
      --label "$report_label" --scenario "bw${beam_width}-in${input_length}" \
      --output "$dir/ab-report-bw${beam_width}-in${input_length}.md" \
      --json-output "$dir/ab-report-bw${beam_width}-in${input_length}.json"
  else
    echo "skip compare bw${beam_width}-in${input_length}: missing one or more A-B-B-A summaries" >&2
  fi
done

echo "AB done: run=$run_name base=$base_sha head=$head_sha"
