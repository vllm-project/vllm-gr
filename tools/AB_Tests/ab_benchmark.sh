#!/usr/bin/env bash
set -Eeuo pipefail

# Local A/B benchmark launcher (Mode B: each developer runs their own benchmark
# server). Resolves a PR's base/head SHAs, drives the server-side ab_runner.sh
# over SSH to run the A-B-B-A schedule, then polls for and prints the report.
#
# Usage:
#   ./ab_benchmark.sh 358                          # A/B PR #358
#   ./ab_benchmark.sh 358 --server my-gpu          # pick an ssh alias
#   ./ab_benchmark.sh --base <sha> --head <sha>    # A/B two arbitrary commits
#   ./ab_benchmark.sh 358 --scenario 256:1024      # override the scenario
#
# Local deps: bash, ssh; `gh` only when resolving a PR number.
# Server preconditions: see README.md. Tuning (container, GPU, model, data,
# num-prompts, warmup) lives in the server's benchmark.env, not here.

# ---- tunables (overridable via environment) ----
SERVER="${AB_SERVER:-}"                                    # ssh alias (required)
REMOTE_DIR="${AB_REMOTE_DIR:-/root/vllm-gr-decode-graph}"  # server worktree path
SCENARIO="${AB_SCENARIO:-128:1024}"                        # beam_width:input_tokens
POLL_INTERVAL="${AB_POLL_INTERVAL:-30}"                    # poll interval (seconds)
TIMEOUT="${AB_TIMEOUT:-3600}"                              # wait timeout (seconds)
GH_BIN="${GH_BIN:-gh}"

pr=""; base=""; head=""; save_dir=""

usage() {
  cat >&2 <<EOF
Usage: $0 <PR> [options]   or   $0 --base <sha> --head <sha> [options]

  <PR>                 GitHub PR number (base/head resolved via gh)
  --base/--head        explicit commits (instead of a PR)
  --server <alias>     benchmark server ssh alias (or set AB_SERVER)
  --remote-dir <dir>   server vllm-gr worktree path (default $REMOTE_DIR)
  --scenario b:i       beam_width:input_tokens (default 128:1024)
  --save-dir <dir>     also download the .md/.json report here
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server)      SERVER="$2"; shift 2 ;;
    --remote-dir)  REMOTE_DIR="$2"; shift 2 ;;
    --scenario)    SCENARIO="$2"; shift 2 ;;
    --base)        base="$2"; shift 2 ;;
    --head)        head="$2"; shift 2 ;;
    --save-dir)    save_dir="$2"; shift 2 ;;
    --help|-h)     usage; exit 0 ;;
    -*)            echo "unknown option: $1" >&2; usage; exit 2 ;;
    *)             pr="$1"; shift ;;
  esac
done

# ---- validate scenario: beam:input ----
if [[ ! "$SCENARIO" =~ ^([0-9]+):([0-9]+)$ ]]; then
  echo "invalid --scenario '$SCENARIO' (expected beam:input, e.g. 128:1024)" >&2
  exit 2
fi
beam_width="${BASH_REMATCH[1]}"
input_length="${BASH_REMATCH[2]}"
case "$beam_width" in 64|128|256) ;; *) echo "beam width must be 64/128/256" >&2; exit 2 ;; esac

# ---- resolve base/head ----
if [[ -n "$pr" ]]; then
  [[ -z "$base" && -z "$head" ]] || { echo "PR and --base/--head are mutually exclusive" >&2; exit 2; }
  command -v "$GH_BIN" >/dev/null 2>&1 || { echo "gh not found; cannot resolve PR #$pr" >&2; exit 2; }
  # headRefOid is the PR's head commit. baseRefOid is the decode_graph tip and is
  # fetched only for display — the actual baseline is the merge base of head and
  # origin/decode_graph, which ab_runner.sh computes server-side.
  base="$("$GH_BIN" pr view "$pr" --json baseRefOid --jq .baseRefOid 2>/dev/null)" \
    || { echo "gh pr view $pr failed" >&2; exit 2; }
  head="$("$GH_BIN" pr view "$pr" --json headRefOid --jq .headRefOid 2>/dev/null)"
  state="$("$GH_BIN" pr view "$pr" --json state --jq .state 2>/dev/null)"
  title="$("$GH_BIN" pr view "$pr" --json title --jq .title 2>/dev/null)"
  if [[ "$state" != "OPEN" ]]; then
    echo "PR #$pr is $state (not OPEN); its head may have been rebased/deleted" >&2
    echo "  use --base/--head to specify two commits explicitly" >&2
    exit 2
  fi
  run_label="PR #$pr"
  echo "PR #$pr: $title"
  echo "  base=$base"
  echo "  head=$head"
else
  [[ -n "$base" && -n "$head" ]] || { echo "need a PR number, or both --base and --head" >&2; usage; exit 2; }
  run_label="$(printf '%.8s' "$head")"
fi

# In PR mode the real baseline is the merge base (fork point) of head and
# origin/decode_graph, computed by ab_runner.sh — baseRefOid is the moving tip
# and must NOT be passed as --base. Explicit --base/--head mode passes base
# through verbatim.
if [[ -n "$pr" ]]; then
  runner_base_arg=""
else
  runner_base_arg="--base '$base'"
fi

# ---- result directory name ----
# A re-run of the same PR/head must never accept a stale report left over from a
# previous invocation: head_sha alone can't distinguish two runs of the same
# head, so a unique result directory (passed to ab_runner.sh via --run) is what
# guarantees the polled report path belongs to this run only.
run_name="${pr:-$(printf '%.8s' "$head")}"
invocation="${run_name}-$(date +%Y%m%d%H%M%S)-$$"
report_md="$REMOTE_DIR/results/ab/$invocation/ab-report-bw${beam_width}-in${input_length}.md"
report_json="$REMOTE_DIR/results/ab/$invocation/ab-report-bw${beam_width}-in${input_length}.json"
log_path="$REMOTE_DIR/results/ab/ab-$invocation.log"

# ---- preflight ----
[[ -n "$SERVER" ]] || { echo "no server: pass --server or set AB_SERVER" >&2; exit 2; }
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$SERVER" "test -d '$REMOTE_DIR/tools/AB_Tests'" 2>/dev/null; then
  echo "server $SERVER has no $REMOTE_DIR/tools/AB_Tests (or is unreachable)" >&2
  echo "  check out a repo containing tools/AB_Tests/ at $REMOTE_DIR first" >&2
  exit 2
fi

# ---- launch the server-side A-B-B-A in the background ----
# The log path lives under results/ab/, which does not exist on a fresh server
# checkout; create it first, or the redirection below fails before ab_runner.sh
# can start (its own later mkdir is too late).
# shellcheck disable=SC2029  # REMOTE_DIR expands on the client side by design
ssh "$SERVER" "mkdir -p '$REMOTE_DIR/results/ab'" </dev/null

# Redirect the ENTIRE background job's stdin/stdout/stderr so ssh returns
# immediately; otherwise ssh waits for the pipeline's fds to close.
launch_cmd="( cd '$REMOTE_DIR' && nohup /usr/bin/bash tools/AB_Tests/ab_runner.sh --pr '$pr' --run '$invocation' $runner_base_arg --head '$head' --scenario '${beam_width}:${input_length}' ) > '$log_path' 2>&1 </dev/null &"

echo "launching A-B-B-A on $SERVER (scenario ${beam_width}:${input_length})..."
# shellcheck disable=SC2029  # launch_cmd expands on the client side by design
ssh "$SERVER" "$launch_cmd" </dev/null

# Surface an immediate launch failure (e.g. redirect error or bad worktree)
# instead of silently waiting for the poll timeout: the log file is created by
# the redirection before the subshell runs, so its absence means the job never
# started.
sleep 2
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$SERVER" "test -f '$log_path'" 2>/dev/null; then
  echo "ab_runner.sh did not start on $SERVER (no log file at $log_path)" >&2
  exit 1
fi

# ---- poll for the report ----
deadline=$(( $(date +%s) + TIMEOUT ))
echo "waiting for the report (timeout ${TIMEOUT}s)..."
while :; do
  # Accept the report only once it exists AND belongs to this head. The report
  # json records head_sha, so a stale report from a previous run of the same
  # name (whose head differs) is ignored rather than returned as this run's
  # result.
  if ssh -o ConnectTimeout=10 -o BatchMode=yes "$SERVER" \
      "test -f '$report_json' && grep -q '\"head_sha\": \"${head:0:8}' '$report_json'" 2>/dev/null; then
    break
  fi
  if (( $(date +%s) >= deadline )); then
    echo "timed out; server log tail:" >&2
    # shellcheck disable=SC2029  # log_path expands on the client side by design
    ssh "$SERVER" "tail -n 20 '$log_path'" 2>/dev/null || true
    exit 1
  fi
  sleep "$POLL_INTERVAL"
done

echo
echo "================ A/B report: $run_label (bw${beam_width}-in${input_length}) ================"
# shellcheck disable=SC2029  # report_md expands on the client side by design
ssh "$SERVER" "cat '$report_md'"

if [[ -n "$save_dir" ]]; then
  mkdir -p "$save_dir"
  scp -q "$SERVER:$report_md" "$save_dir/" 2>/dev/null || echo "failed to download .md" >&2
  scp -q "$SERVER:$report_json" "$save_dir/" 2>/dev/null || echo "failed to download .json" >&2
  echo "saved report to $save_dir"
fi
