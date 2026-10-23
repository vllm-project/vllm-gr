#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail
export VLLM_GR_TARGET_DEVICE="${VLLM_GR_TARGET_DEVICE:-cuda}"
export V1_PERF_BEAM_WIDTH=128
# This workload prefills 1024 tokens per request at B=1,2,4. Keep its graph
# captures local to the A/B job instead of changing the runtime defaults.
export VLLM_GR_PREFILL_GRAPH_BUCKETS=1024,2048,4096
export VLLM_GR_PREFILL_GRAPH_BUCKET_STEP=1024
export VLLM_GR_PREFILL_GRAPH_MAX_BUCKET=4096
export VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS=1,2,4
unset VLLM_GR_PREFILL_GRAPH_KV_BOUND

candidate_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
event_base_sha="${V1_PERF_BASE_SHA:?V1_PERF_BASE_SHA is required}"
candidate_sha="${V1_PERF_CANDIDATE_SHA:?V1_PERF_CANDIDATE_SHA is required}"
head_sha="${V1_PERF_HEAD_SHA:?V1_PERF_HEAD_SHA is required}"
pr_number="${V1_PERF_PR_NUMBER:?V1_PERF_PR_NUMBER is required}"
output_dir="${V1_PERF_OUTPUT_DIR:?V1_PERF_OUTPUT_DIR is required}"
temp_root="${RUNNER_TEMP:?RUNNER_TEMP is required}"
base_root="$(mktemp -d "$temp_root/beam-v1-perf-base.XXXXXXXX")"
worktree_added=0
candidate_installed=1

cleanup() {
  local status=$?
  trap - EXIT
  if (( ! candidate_installed )); then
    cd "$candidate_root"
    uv pip install -e '.[dev]' || status=1
  fi
  if (( worktree_added )); then
    git -C "$candidate_root" worktree remove --force "$base_root" || status=1
  else
    rmdir "$base_root" || status=1
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$event_base_sha" =~ ^[[:xdigit:]]{40}$ ]] || { echo "invalid event base SHA" >&2; exit 2; }
[[ "$candidate_sha" =~ ^[[:xdigit:]]{40}$ ]] || { echo "invalid candidate SHA" >&2; exit 2; }
[[ "$head_sha" =~ ^[[:xdigit:]]{40}$ ]] || { echo "invalid PR head SHA" >&2; exit 2; }
[[ "$pr_number" =~ ^[1-9][0-9]*$ ]] || { echo "invalid PR number" >&2; exit 2; }
[[ "$(git -C "$candidate_root" rev-parse HEAD)" == "$candidate_sha" ]] || {
  echo "candidate checkout does not match the PR event" >&2
  exit 2
}
base_sha="$(git -C "$candidate_root" rev-parse 'HEAD^')"
if [[ "$base_sha" != "$event_base_sha" ]]; then
  echo "PR event base $event_base_sha is stale; measuring merge parent $base_sha."
fi
[[ "$(git -C "$candidate_root" rev-parse 'HEAD^2')" == "$head_sha" ]] || {
  echo "PR merge commit does not use the event head" >&2
  exit 2
}
if ! git -C "$candidate_root" cat-file -e "$base_sha^{commit}" 2>/dev/null; then
  git -C "$candidate_root" fetch --no-tags origin "$base_sha"
fi
git -C "$candidate_root" worktree add --detach "$base_root" "$base_sha"
worktree_added=1

# Both sides run the candidate's benchmark driver, including lifetime fixes.
# Its SID validator and prompt are still imported/read from each selected tree.
for path in \
  benchmarks/beam_search/v1_sid_validation.py \
  tests/resources/single_one_rec_prompt.txt; do
  if ! git -C "$candidate_root" diff --quiet "$base_sha" "$candidate_sha" -- "$path"; then
    echo "A/B workload changed between base and PR: $path" >&2
    exit 1
  fi
done

run_pass() {
  local repo_root="$1"
  local result_dir="$2"
  cd "$repo_root"
  V1_PERF_REPO_ROOT="$repo_root" V1_PERF_OUTPUT_DIR="$result_dir" \
    V1_PERF_DRIVER="$candidate_root/benchmarks/beam_search/v1_performance_gate.py" \
    V1_PERF_MEASURE_ONLY=1 V1_PERF_OFFLINE_ONLY=1 \
    V1_PERF_BATCH_SIZES='4 2 1' \
    V1_PERF_WARMUP_RUNS=10 V1_PERF_B1_WARMUP_RUNS=20 V1_PERF_MEASURED_RUNS=20 \
    bash "$candidate_root/tests/system/run_beam_v1_performance.sh"
}

run_abba() {
  local results="$1"
  cd "$base_root"
  candidate_installed=0
  uv pip install -e '.[dev]'
  run_pass "$base_root" "$results/base/1"

  cd "$candidate_root"
  uv pip install -e '.[dev]'
  candidate_installed=1
  run_pass "$candidate_root" "$results/pr/1"
  run_pass "$candidate_root" "$results/pr/2"

  cd "$base_root"
  candidate_installed=0
  uv pip install -e '.[dev]'
  run_pass "$base_root" "$results/base/2"

  cd "$candidate_root"
  uv pip install -e '.[dev]'
  candidate_installed=1
}

mkdir -p "$output_dir"
rm -f "$output_dir/context.json" "$output_dir/report.json" "$output_dir/report.md"
run_abba "$output_dir"

printf '{"pr_number":%s,"base_sha":"%s","head_sha":"%s","merge_sha":"%s"}\n' \
  "$pr_number" "$base_sha" "$head_sha" "$candidate_sha" > "$output_dir/context.json"

python -m benchmarks.beam_search.v1_performance_ab \
  --base "$output_dir/base" \
  --pr "$output_dir/pr" \
  --output-dir "$output_dir" \
  --base-sha "$base_sha" \
  --pr-sha "$candidate_sha"
