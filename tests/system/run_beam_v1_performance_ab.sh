#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

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
base_venv="$(mktemp -d "$temp_root/beam-v1-perf-venv.XXXXXXXX")"
candidate_venv="${VIRTUAL_ENV:?VIRTUAL_ENV must identify the verified candidate environment}"
candidate_python="$candidate_venv/bin/python"
base_python="$base_venv/bin/python"
worktree_added=0

cleanup() {
  local status=$?
  trap - EXIT
  cd "$candidate_root" || status=1
  if (( worktree_added )); then
    git -C "$candidate_root" worktree remove --force "$base_root" || status=1
  else
    rmdir "$base_root" || status=1
  fi
  rm -rf "$base_venv"
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
[[ "$(git -C "$candidate_root" rev-parse 'HEAD^2')" == "$head_sha" ]] || {
  echo "PR merge commit does not use the event head" >&2
  exit 2
}
# GitHub can refresh refs/pull/<n>/merge after the workflow event is queued.
# In that case the synthetic merge's first parent is newer than the event
# payload. Compare against that actual parent so upstream changes are not
# attributed to the PR, while requiring a fast-forward base update.
if ! git -C "$candidate_root" merge-base --is-ancestor "$event_base_sha" "$base_sha"; then
  echo "PR merge base is not descended from the event base" >&2
  exit 2
fi
if ! git -C "$candidate_root" cat-file -e "$base_sha^{commit}" 2>/dev/null; then
  git -C "$candidate_root" fetch --no-tags origin "$base_sha"
fi
git -C "$candidate_root" worktree add --detach "$base_root" "$base_sha"
worktree_added=1

mkdir -p "$output_dir/runtime"
"$candidate_python" "$candidate_root/tools/ci/resolve_runtime.py" resolve \
  --repo "$candidate_root" \
  --device cuda \
  --output "$output_dir/runtime/candidate-spec.json"
"$candidate_python" "$candidate_root/tools/ci/resolve_runtime.py" verify \
  --repo "$candidate_root" \
  --spec "$output_dir/runtime/candidate-spec.json" \
  --venv "$candidate_venv" \
  --output "$output_dir/runtime/candidate-installed.json"

"$candidate_python" "$candidate_root/tools/ci/resolve_runtime.py" resolve \
  --repo "$base_root" \
  --device cuda \
  --output "$output_dir/runtime/base-spec.json"
rm -rf "$base_venv"
uv venv --seed --python "$candidate_python" "$base_venv"
VLLM_GR_TARGET_DEVICE=cuda uv pip install \
  --python "$base_python" \
  -e "${base_root}[dev]"
"$base_python" "$candidate_root/tools/ci/resolve_runtime.py" verify \
  --repo "$base_root" \
  --spec "$output_dir/runtime/base-spec.json" \
  --venv "$base_venv" \
  --output "$output_dir/runtime/base-installed.json"

# Both sides run the candidate's benchmark driver, including lifetime fixes.
# Its SID validator and prompt are still imported/read from each selected tree.
for path in \
  benchmarks/beam_search/v1_sid_validation.py \
  tests/resources/single_one_rec_prompt.txt; do
  diff_options=(--quiet)
  if [[ "$path" == benchmarks/beam_search/v1_sid_validation.py ]]; then
    # The SPDX copyright header does not affect SID validation.
    diff_options+=(-I '^# SPDX-FileCopyrightText: Copyright contributors to the vLLM(-GR)? project$')
  fi
  if ! git -C "$candidate_root" diff "${diff_options[@]}" "$base_sha" "$candidate_sha" -- "$path"; then
    echo "A/B workload changed between base and PR: $path" >&2
    exit 1
  fi
done

run_pass() {
  local repo_root="$1"
  local result_dir="$2"
  local runtime_venv="$3"
  cd "$repo_root"
  PATH="$runtime_venv/bin:$PATH" VIRTUAL_ENV="$runtime_venv" \
    V1_PERF_REPO_ROOT="$repo_root" V1_PERF_OUTPUT_DIR="$result_dir" \
    V1_PERF_DRIVER="$candidate_root/benchmarks/beam_search/v1_performance_gate.py" \
    V1_PERF_MEASURE_ONLY=1 V1_PERF_OFFLINE_ONLY=1 \
    V1_PERF_WARMUP_RUNS=10 V1_PERF_MEASURED_RUNS=20 \
    bash "$candidate_root/tests/system/run_beam_v1_performance.sh"
}

run_abba() {
  local results="$1"
  run_pass "$base_root" "$results/base/1" "$base_venv"
  run_pass "$candidate_root" "$results/pr/1" "$candidate_venv"
  run_pass "$candidate_root" "$results/pr/2" "$candidate_venv"
  run_pass "$base_root" "$results/base/2" "$base_venv"
  cd "$candidate_root"
}

mkdir -p "$output_dir"
rm -f "$output_dir/context.json" "$output_dir/report.json" "$output_dir/report.md"
run_abba "$output_dir"

printf '{"pr_number":%s,"base_sha":"%s","head_sha":"%s","merge_sha":"%s"}\n' \
  "$pr_number" "$base_sha" "$head_sha" "$candidate_sha" > "$output_dir/context.json"

if V1_PERF_DEFER_SUMMARY=1 "$candidate_python" -m benchmarks.beam_search.v1_performance_ab \
  --base "$output_dir/base" \
  --pr "$output_dir/pr" \
  --output-dir "$output_dir/initial" \
  --base-sha "$base_sha" \
  --pr-sha "$candidate_sha"; then
  cp "$output_dir/initial/report.json" "$output_dir/report.json"
  cp "$output_dir/initial/report.md" "$output_dir/report.md"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$output_dir/report.md" >> "$GITHUB_STEP_SUMMARY"
  fi
  exit 0
fi

# Only a complete regression confined to one batch size gets a confirmation run.
"$candidate_python" - "$output_dir/initial/report.json" "$output_dir/initial/decision.txt" <<'PY'
import sys
from pathlib import Path

from benchmarks.beam_search.v1_performance_ab import read_json, should_retry

report = read_json(Path(sys.argv[1]))
if not report.get("failures"):
    raise SystemExit("initial A/B comparison did not produce a complete regression report")
decision = "retry" if should_retry(report) else "fail"
Path(sys.argv[2]).write_text(decision + "\n", encoding="utf-8")
PY

read -r retry_decision < "$output_dir/initial/decision.txt"
if [[ "$retry_decision" == "fail" ]]; then
  cp "$output_dir/initial/report.json" "$output_dir/report.json"
  cp "$output_dir/initial/report.md" "$output_dir/report.md"
  printf 'Confirmation skipped: regressions affect multiple batch sizes.\n' \
    >> "$output_dir/report.md"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$output_dir/report.md" >> "$GITHUB_STEP_SUMMARY"
  fi
  echo "Regressions affect multiple batch sizes; failing without confirmation."
  exit 1
fi

echo "One batch size exceeded +1%; repeating the full ABBA measurement."
run_abba "$output_dir/retry"
"$candidate_python" - "$output_dir" "$base_sha" "$candidate_sha" <<'PY'
import json
import os
import sys
from pathlib import Path

from benchmarks.beam_search.v1_performance_ab import (
    SETUP_KEYS,
    combine_retry_results,
    compare,
    read_json,
)

root = Path(sys.argv[1])
base_sha, candidate_sha = sys.argv[2:]
initial_setup = read_json(root / "base/1/setup.json")
retry_setup = read_json(root / "retry/base/1/setup.json")
if any(initial_setup.get(key) != retry_setup.get(key) for key in SETUP_KEYS):
    raise ValueError("A/B settings changed between initial and confirmation runs")

initial = (
    read_json(root / "initial/report.json"),
    (root / "initial/report.md").read_text(encoding="utf-8"),
)
confirmation = compare(root / "retry/base", root / "retry/pr", base_sha, candidate_sha)
retry_result, retry_report = confirmation
(root / "retry/report.json").write_text(
    json.dumps(retry_result, indent=2) + "\n", encoding="utf-8"
)
(root / "retry/report.md").write_text(retry_report, encoding="utf-8")
result, report = combine_retry_results(initial, confirmation)
(root / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
(root / "report.md").write_text(report, encoding="utf-8")
print(report, end="", flush=True)
if path := os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(report)
if result["failures"]:
    raise SystemExit("Beam V1 A/B gate failed again on the confirmation run")
PY
