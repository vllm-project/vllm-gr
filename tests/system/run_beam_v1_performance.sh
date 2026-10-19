#!/usr/bin/env bash

set -euo pipefail

repo_root="${V1_PERF_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$repo_root"
device="${BENCHMARK_DEVICE:-gpu}"
if [[ "$device" != "gpu" && "$device" != "npu" ]]; then
  echo "BENCHMARK_DEVICE must be gpu or npu, got: $device" >&2
  exit 2
fi
# A/B can share one driver while importing each side's runtime from its own tree.
driver="${V1_PERF_DRIVER:-$repo_root/benchmarks/beam_search/v1_performance_gate.py}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
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
default_memory_utilization=0.7
if [[ "$device" == "npu" ]]; then
  default_memory_utilization=0.5
fi
gpu_memory_utilization="${V1_PERF_GPU_MEMORY_UTILIZATION:-$default_memory_utilization}"
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
  --device "$device"
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
if [[ "$device" == "gpu" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
    | tee "$output_dir/gpu.csv" || true
elif [[ "$device" == "npu" ]] && command -v npu-smi >/dev/null 2>&1; then
  npu-smi info | tee "$output_dir/npu.txt" || true
fi
python "$driver" prepare "${args[@]}"
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
on_exit() {
  local status=$?
  stop_server
  if [[ "$device" == "npu" && "$status" -ne 0 && -s "$output_dir/server.log" ]]; then
    echo "NPU V1 online server log (last 80 lines):" >&2
    tail -80 "$output_dir/server.log" >&2
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Running offline GRLLM.beam_search_v1 batch performance"
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python "$driver" offline "${args[@]}"

if [[ "${V1_PERF_MEASURE_ONLY:-0}" == 1 && "${V1_PERF_OFFLINE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

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
python "$driver" online "${args[@]}" \
  --served-model-name "$server_identity"
stop_server

if [[ "${V1_PERF_MEASURE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

python - "$output_dir" <<'PY'
import json
import os
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
setup = json.loads((output_dir / "setup.json").read_text(encoding="utf-8"))
lines = [
    "## Beam V1 performance measurement",
    "",
    "This push run has no PR base commit for an A/B comparison.",
    "",
    f"Model: `{setup['model']}`; revision: `{setup['revision']}`; "
    f"warmup: {setup['warmup_runs']}; measured: {setup['measured_runs']}",
    "",
    "| Mode | B / concurrency | Mean group latency ms |",
    "| --- | ---: | ---: |",
]
for mode in ("offline", "online"):
    data = json.loads((output_dir / f"{mode}.json").read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise SystemExit(f"{mode} performance result is incomplete")
    for case in data["cases"]:
        count = case.get("batch_size", case.get("concurrency"))
        lines.append(f"| {mode} | {count} | {case['group_latency_ms']['mean_ms']:.3f} |")
summary = "\n".join(lines) + "\n"
print(summary, flush=True)
if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(summary_path, "a", encoding="utf-8") as stream:
        stream.write(summary)
PY
