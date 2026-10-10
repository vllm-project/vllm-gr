#!/usr/bin/env bash

# Compare upstream vLLM BeamSearch semantics with decode Constraint Tables on
# exactly the same full-catalog OneRec canary prompts.

set -euo pipefail

# Interpreter override: point at a venv with the repo-pinned vLLM build
# when the ambient "python" resolves to a different installation.
PYTHON="${PYTHON:-python}"

# Resolve the vllm-gr console script next to the chosen interpreter so the
# online serve leg uses the same environment as the runner (the venv bin is
# not necessarily on PATH).
if [[ -x "$(dirname "$PYTHON")/vllm-gr" ]]; then
  VLLM_GR_BIN="$(dirname "$PYTHON")/vllm-gr"
else
  VLLM_GR_BIN="vllm-gr"
fi

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
  --constrained-golden PATH    Blocking aggregate golden (default follows
                               --decode-execution: ci_precision_v1_
                               constrained_golden_offline.json for offline,
                               ci_precision_v1_constrained_golden.json for
                               online)
  --decode-execution MODE      offline|online (default: offline).
                               offline: runner boots the canonical Beam V1
                               engine directly (llm.beam_search_v1) from the
                               prepared artifact -- the maintained GPU path.
                               online: serve over HTTP (graph-worker); needs
                               the online V1 dispatch to be servable.
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
  --capture-golden             Write the constrained golden from this run's
                               decode summaries instead of gating against one
                               (use to recapture after engine/protocol changes)
  --golden-model-label TEXT    Stable model label in a captured golden
                               (default: OpenOneRec/OneRec-1.7B)
  --golden-revision-label TEXT Model revision label in a captured golden
  --skip-prepare               Reuse and validate an existing prepared manifest
  --prepare-force              Rebuild prepared catalog/triples
  --overwrite                  Replace an existing output directory
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
model=""
revision=""
constrained_golden=""
constrained_golden_explicit=0
decode_execution="offline"
capture_golden=0
golden_model_label="OpenOneRec/OneRec-1.7B"
golden_revision_label=""
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
skip_prepare=0
prepare_force=0
overwrite=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --revision) revision="$2"; shift 2 ;;
    --constrained-golden) constrained_golden="$2"; constrained_golden_explicit=1; shift 2 ;;
    --decode-execution) decode_execution="$2"; shift 2 ;;
    --capture-golden) capture_golden=1; shift ;;
    --golden-model-label) golden_model_label="$2"; shift 2 ;;
    --golden-revision-label) golden_revision_label="$2"; shift 2 ;;
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
    --skip-prepare) skip_prepare=1; shift ;;
    --prepare-force) prepare_force=1; shift ;;
    --overwrite) overwrite=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Neither ingress path may be gated against the other's baseline (the gate
# rejects cross-execution protocol anyway).  When --constrained-golden is not
# given, pick the baseline that matches --decode-execution so a bare
# `--decode-execution online` run cannot silently compare against -- or, on
# capture, overwrite -- the offline golden.
if [[ "$constrained_golden_explicit" -eq 0 ]]; then
  if [[ "$decode_execution" == "online" ]]; then
    constrained_golden="$repo_root/benchmarks/open_one_rec/manifests/ci_precision_v1_constrained_golden.json"
  else
    constrained_golden="$repo_root/benchmarks/open_one_rec/manifests/ci_precision_v1_constrained_golden_offline.json"
  fi
fi

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
if [[ "$capture_golden" -ne 1 && ! -f "$constrained_golden" ]]; then
  echo "Constrained golden does not exist: $constrained_golden" >&2
  exit 2
fi
# Fail before the expensive decode run if the golden records a different
# execution than the one being run (the gate would reject the protocol anyway).
if [[ "$capture_golden" -ne 1 ]]; then
  if ! "${PYTHON}" - "$constrained_golden" "$decode_execution" <<'PYEOF'
import json
import sys

golden = json.load(open(sys.argv[1]))
execution = golden.get("protocol", {}).get("execution")
if execution != sys.argv[2]:
    raise SystemExit(
        f"constrained golden protocol execution {execution!r} does not match "
        f"--decode-execution {sys.argv[2]!r}; pass a matching --constrained-golden"
    )
PYEOF
  then
    exit 2
  fi
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
  # Both the offline engine and the online V1 serve boot the canonical Beam V1
  # config, which needs the compiled constraint-table artifact.
  prepare_args+=(--model "$model" --build-artifacts --artifact-backend cuda)
  "${PYTHON}" -m benchmarks.open_one_rec.tools.prepare_decode_data "${prepare_args[@]}"
fi
for domain in video product; do
    if ! "${PYTHON}" - "$prepared_dir/manifest.json" "$domain" <<'PYEOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
domain_info = manifest.get("domains", {}).get(sys.argv[2], {})
if not isinstance(domain_info.get("artifact"), dict):
    raise SystemExit(f"prepared domain {sys.argv[2]!r} has no artifact; rerun without --skip-prepare")
PYEOF
    then
      exit 2
    fi
  done

mkdir -p \
  "$output_dir/decode" \
  "$output_dir/diffs" \
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
  local log_path="$output_dir/logs/decode-${domain}.log"
  local implementation_revision
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

  local -a runner_args=()
  if [[ "$decode_execution" == "offline" ]]; then
    # Maintained GPU offline path: the runner boots the canonical Beam V1
    # engine itself (execution_mode=v1 + startup graph capture) from the
    # prepared constraint-table artifact and drives llm.beam_search_v1.
    # No server is started.
    runner_args=(
      --execution offline
      --execution-variant offline-v1-graph
      --constraint-mode artifact
      --gpu-memory-utilization "$gpu_memory_utilization"
      --max-model-len "$max_model_len"
    )
  else
    # Online serving path (HTTP ingress).  Beam V1 requests are dispatched by
    # the online engine (#444).  Serve the same canonical Beam V1 engine as the
    # offline leg (execution_mode=v1 + the compiled constraint-table artifact)
    # so both ingress paths execute identical math; the canonical YAML config
    # is derived from the prepared manifest.
    local gr_config_path="$output_dir/logs/${domain}_gr_config.yaml"
    "${PYTHON}" - "$prepared_dir/manifest.json" "$domain" "$gr_config_path" "$beam_width" <<'PYEOF'
import json
import pathlib
import sys

import yaml

manifest = json.load(open(sys.argv[1]))
artifact = manifest["domains"][sys.argv[2]]["artifact"]
gr_config = {
    "schema_version": 1,
    "beam": {
        "execution_mode": "v1",
        "graph_enabled": True,
        "worker_decision": True,
        "max_width": int(sys.argv[4]),
        "max_decode_steps": 3,
    },
    "constraint_table": {
        "enabled": True,
        "backend": artifact["backend"],
        "path": artifact["path"],
        "format": "constraint_table_v1",
        "artifact_digest": artifact["artifact_digest"],
        "tokenizer_digest": artifact["tokenizer_digest"],
        "max_top_k": artifact.get("max_top_k", 256),
    },
    "attention_backend": "CUSTOM",
}
pathlib.Path(sys.argv[3]).write_text(yaml.safe_dump(gr_config))
PYEOF
    echo "Starting decode server for $domain universe: $gr_config_path"
    local -a server_command=("$VLLM_GR_BIN")
    "${server_command[@]}" serve \
      --port "$port" \
      --seed 123 \
      --vllm-gr-config "$gr_config_path" \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --max-logprobs "$((beam_width * 2))" \
      --max-num-seqs 2048 \
      --max-num-batched-tokens 16384 \
      --max-model-len "$max_model_len" \
      --scheduling-policy priority \
      --gpu-memory-utilization "$gpu_memory_utilization" \
      "${revision_args[@]}" \
      "$model" > "$log_path" 2>&1 &
    server_pid=$!
    if ! wait_for_server; then
      tail -200 "$log_path" >&2 || true
      return 1
    fi
    runner_args=(
      --execution online
      --execution-variant graph-worker
      --constraint-mode artifact
      --host 127.0.0.1
      --port "$port"
      --request-timeout "$request_timeout"
    )
  fi

  local -a common_args=(
    "${runner_args[@]}"
    --model "$model"
    --implementation-revision "$implementation_revision"
    --dataset-path "$dataset_path"
    --prepared-manifest "$prepared_dir/manifest.json"
    --output-dir "$result_dir"
    --sample-manifest "$sample_manifest"
    --beam-width "$beam_width"
    --temperature "$temperature"
    --evaluation-mode "$evaluation_mode"
    --overwrite
    --strict
  )
  if [[ -n "$revision" ]]; then
    common_args+=(--revision "$revision")
  fi
  local joined
  joined="$(IFS=,; printf '%s' "${tasks[*]}")"
  if [[ "$decode_execution" == "offline" ]]; then
    # One process, one engine, every task in this constraint domain: the ~90s
    # engine startup (graph capture/compilation) is paid once per domain
    # instead of once per task. The online path keeps one request loop per
    # task against the shared server.
    echo "Running decode/$domain canary tasks=$joined (execution=$decode_execution)"
    NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost" \
    no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost" \
      "${PYTHON}" -m benchmarks.open_one_rec.run_recommendation_regression \
        --tasks "$joined" "${common_args[@]}"
  else
    local task
    for task in "${tasks[@]}"; do
      echo "Running decode/$task canary (execution=$decode_execution)"
      NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost" \
      no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost" \
        "${PYTHON}" -m benchmarks.open_one_rec.run_recommendation_regression \
          --task "$task" "${common_args[@]}"
    done
  fi
  stop_server
}

run_decode_domain video
run_decode_domain product

gate_task_args=()
for task in "${selected_tasks[@]}"; do
  gate_task_args+=(--task "$task")
done
if [[ "$capture_golden" -eq 1 ]]; then
  capture_args=(
    --capture
    --candidate-dir "$output_dir/decode"
    --output "$constrained_golden"
    --capture-description "Blocking five-task Constraint Tables golden (decode_execution=$decode_execution, NVIDIA L20)."
    --capture-model "$golden_model_label"
  )
  if [[ -n "$golden_revision_label" ]]; then
    capture_args+=(--capture-revision "$golden_revision_label")
  fi
  "${PYTHON}" -m benchmarks.open_one_rec.gate_constrained_golden \
    "${capture_args[@]}" "${gate_task_args[@]}"
  echo "Captured constrained golden written to $constrained_golden"
elif ! "${PYTHON}" -m benchmarks.open_one_rec.gate_constrained_golden \
    --golden "$constrained_golden" \
    --candidate-dir "$output_dir/decode" \
    --output "$output_dir/diffs/constrained-golden.json" \
    --evaluation-mode "$evaluation_mode" \
    "${gate_task_args[@]}" > /dev/null; then
  echo "OneRec constrained decode is below its versioned constrained golden;" \
    "inspect $output_dir/diffs/constrained-golden.json." >&2
  exit 1
else
  echo "OneRec constrained decode passes the versioned constrained-golden policy."
fi
