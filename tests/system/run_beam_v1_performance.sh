#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
model="${VLLM_GR_TEST_MODEL:-OpenOneRec/OneRec-1.7B}"
triples="${TRIPLES_PATH:?TRIPLES_PATH must point to the CI video triples}"
output_dir="${V1_PERF_OUTPUT_DIR:-$repo_root/v1_performance_results}"
prompt="$repo_root/tests/resources/single_one_rec_prompt.txt"
port="${V1_PERF_PORT:-7001}"
beam_width="${V1_PERF_BEAM_WIDTH:-32}"
generated_steps="${V1_PERF_GENERATED_STEPS:-3}"
input_len="${V1_PERF_INPUT_LEN:-1024}"
warmup_runs="${V1_PERF_WARMUP_RUNS:-2}"
measured_runs="${V1_PERF_MEASURED_RUNS:-5}"
request_timeout="${V1_PERF_REQUEST_TIMEOUT:-180}"
max_model_len="${V1_PERF_MAX_MODEL_LEN:-8192}"
gpu_memory_utilization="${V1_PERF_GPU_MEMORY_UTILIZATION:-0.7}"
read -r -a batch_sizes <<< "${V1_PERF_BATCH_SIZES:-1 2 4}"
max_batch_size=0
for size in "${batch_sizes[@]}"; do
  if (( size > max_batch_size )); then
    max_batch_size=$size
  fi
done

revision_args=()
if [[ ! -d "$model" ]]; then
  revision_args=(--revision "${V1_PERF_MODEL_REVISION:-cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b}")
fi
args=(
  --model "$model"
  "${revision_args[@]}"
  --triples "$triples"
  --prompt "$prompt"
  --output-dir "$output_dir"
  --beam-width "$beam_width"
  --generated-steps "$generated_steps"
  --input-len "$input_len"
  --batch-sizes "${batch_sizes[@]}"
  --warmup-runs "$warmup_runs"
  --measured-runs "$measured_runs"
  --request-timeout "$request_timeout"
  --max-model-len "$max_model_len"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --port "$port"
)

mkdir -p "$output_dir"
echo "V1 performance output: $output_dir"
git -C "$repo_root" rev-parse HEAD | tee "$output_dir/commit.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
    | tee "$output_dir/gpu.csv" || true
fi
python -m benchmarks.beam_search.v1_performance_gate prepare "${args[@]}"
cat "$output_dir/setup.json"

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    for ((attempt = 0; attempt < 20; attempt++)); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Running offline GRLLM.beam_search_v1 batch performance"
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python -m benchmarks.beam_search.v1_performance_gate offline "${args[@]}"

echo "Starting online V1 server"
server_identity="v1-perf-$(python -c 'import uuid; print(uuid.uuid4().hex)')"
python - "$port" <<'PY'
import socket
import sys

with socket.socket() as probe:
    try:
        probe.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError as exc:
        raise SystemExit(f"Port {sys.argv[1]} is occupied: {exc}") from exc
PY
server_args=(
  serve
  "$model"
  --port "$port"
  --served-model-name "$server_identity"
  --seed 0
  --vllm-gr-config "$output_dir/vllm_gr_config.yaml"
  --async-scheduling
  --max-logprobs "$beam_width"
  --max-num-seqs "$max_batch_size"
  --max-num-batched-tokens "$max_model_len"
  --max-model-len "$max_model_len"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --no-enable-prefix-caching
  --no-enable-chunked-prefill
)
if [[ ${#revision_args[@]} -gt 0 ]]; then
  server_args+=("${revision_args[@]}")
fi
vllm-gr "${server_args[@]}" > "$output_dir/server.log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + 600))
while ! curl --silent --show-error --fail --connect-timeout 2 --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Online V1 server exited during startup" >&2
    tail -80 "$output_dir/server.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Online V1 server readiness timed out" >&2
    tail -80 "$output_dir/server.log" >&2
    exit 1
  fi
  sleep 2
done
if ! kill -0 "$server_pid" 2>/dev/null; then
  echo "Online V1 server exited after readiness" >&2
  exit 1
fi
python - "$server_identity" "$port" <<'PY'
import json
import sys
from urllib.request import urlopen

with urlopen(f"http://127.0.0.1:{sys.argv[2]}/v1/models", timeout=3) as response:
    models = json.load(response)
if sys.argv[1] not in {model.get("id") for model in models.get("data", [])}:
    raise SystemExit("Ready server does not expose this run's unique model identity")
PY
if ! kill -0 "$server_pid" 2>/dev/null; then
  echo "Online V1 server exited before measurement" >&2
  exit 1
fi

echo "Running online V1 HTTP concurrency performance"
python -m benchmarks.beam_search.v1_performance_gate online "${args[@]}" \
  --served-model-name "$server_identity"
stop_server

python - "$output_dir" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
setup = json.loads((output_dir / "setup.json").read_text(encoding="utf-8"))
baseline_ms = {
    "offline": {1: 33.78, 2: 56.32, 4: 93.44},
    "online": {1: 55.08, 2: 82.82, 4: 156.22},
}
if (
    setup["model"] != "OpenOneRec/OneRec-1.7B"
    or setup["revision"] != "cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b"
    or setup["input_tokens_including_sid_begin"] != 1024
    or setup["beam_width"] != 32
    or setup["generated_steps"] != 3
    or setup["batch_sizes"] != [1, 2, 4]
):
    raise SystemExit(
        "Beam V1 performance limits require 1024 tokens, width=32, steps=3, B/C=1,2,4"
    )
lines = [
    "## Beam V1 performance on this CI runner",
    "",
    f"Model: `{setup['model']}`; revision: `{setup['revision']}`; "
    f"input tokens: {setup['input_tokens_including_sid_begin']}; "
    f"beam width: {setup['beam_width']}; generated steps: {setup['generated_steps']}; "
    f"warmup runs: {setup['warmup_runs']}; measured runs: {setup['measured_runs']}",
    "",
    "Offline values use one fixed B-request batch per call. Online values use "
    "C simultaneous HTTP requests, each with one logical request. V1 sends "
    "terminal results, so token-stream TTFT and TPOT are unavailable.",
    "",
    "The gate checks mean group latency against the 1024-token baseline plus 3 ms, "
    "rounded up to whole milliseconds. Offline misses fail the job. Online misses "
    "are warnings and do not fail the job.",
    "",
    "### Baseline comparison",
    "",
    "Change = current - baseline; positive values mean higher latency.",
    "",
    "| Mode | B / concurrency | Baseline ms | Current ms | Change ms | "
    "Change % | Limit ms | Gate |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
details = [
    "| Mode | B / concurrency | Input tokens | Mean group ms | P50 | P95 | P99 | "
    "Mean request ms | Requests/s | Beam tokens/s | Limit ms | Gate |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
blocking = []
warnings = []
for mode in ("offline", "online"):
    data = json.loads((output_dir / f"{mode}.json").read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise SystemExit(f"{mode} performance result is incomplete")
    seen = set()
    for case in data["cases"]:
        group = case["group_latency_ms"]
        request = case["request_latency_ms"]
        count = case.get("batch_size", case.get("concurrency"))
        if count in seen or count not in baseline_ms[mode]:
            raise SystemExit(f"unexpected or repeated {mode} B/C={count}")
        seen.add(count)
        mean_ms = group["mean_ms"]
        baseline = baseline_ms[mode][count]
        limit_ms = math.ceil(baseline + 3.0)
        input_lengths = case["prompt_tokens_including_sid_begin"]
        if len(input_lengths) != count or any(
            length != setup["input_tokens_including_sid_begin"] for length in input_lengths
        ):
            raise SystemExit(f"unexpected {mode} B/C={count} input token length")
        if not math.isfinite(mean_ms):
            raise SystemExit(f"non-finite {mode} B/C={count} latency")
        passed = mean_ms <= limit_ms
        gate = "PASS"
        if not passed:
            message = f"{mode} B/C={count}: {mean_ms:.3f} ms > {limit_ms} ms"
            if mode == "online":
                warnings.append(message)
                gate = "WARN"
            else:
                blocking.append(message)
                gate = "FAIL"
        change_ms = mean_ms - baseline
        lines.append(
            f"| {mode} | {count} | {baseline:.2f} | {mean_ms:.3f} | "
            f"{change_ms:+.3f} | {change_ms / baseline * 100:+.2f}% | "
            f"{limit_ms} | {gate} |"
        )
        input_range = f"{min(input_lengths)}–{max(input_lengths)}"
        details.append(
            f"| {mode} | {count} | {input_range} | {mean_ms:.3f} | "
            f"{group['p50_ms']:.2f} | {group['p95_ms']:.2f} | "
            f"{group['p99_ms']:.2f} | {request['mean_ms']:.2f} | "
            f"{case['requests_per_second']:.2f} | "
            f"{case['beam_generated_tokens_per_second']:.2f} | {limit_ms} | "
            f"{gate} |"
        )
    if seen != set(baseline_ms[mode]):
        raise SystemExit(f"missing {mode} performance cases: {set(baseline_ms[mode]) - seen}")
lines.extend(("", "### Detailed measurements", "", *details))
summary = "\n".join(lines) + "\n"
print(summary, flush=True)
if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(summary_path, "a", encoding="utf-8") as stream:
        stream.write(summary)
for message in warnings:
    print(f"::warning title=Beam V1 online latency::{message}", flush=True)
if blocking:
    raise SystemExit("Beam V1 performance gate failed:\n" + "\n".join(blocking))
PY
