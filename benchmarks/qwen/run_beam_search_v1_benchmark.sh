#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: benchmarks/qwen/run_beam_search_v1_benchmark.sh [extra Python arguments]

Create/reuse an isolated environment and benchmark offline
GRLLM.beam_search_v1. This entry point depends only on the vllm-gr checkout.

Required environment:
  MODEL=/absolute/path/to/model
  TOKEN_ID_START=            inclusive input/output token bound
  TOKEN_ID_END=              exclusive input/output token bound
  # The same bounds apply when reusing WORKLOAD_JSONL.

Common environment overrides:
  VENV_DIR=repo/.venv
  BOOTSTRAP_PYTHON=python3.11
  INSTALL_MODE=auto          auto, always, or skip
  CONTEXT_LEN=5000
  WORKLOAD_SEED=0
  WORKLOAD_JSONL=            existing workload; skips generation
  BEAM_STEPS=3               one Prefill plus two Decode stages
  DECODE_STEPS=              backward-compatible alias for BEAM_STEPS
  BEAM_WIDTH=256
  REQUESTS=1                 Beam Search V1 currently submits B=1 calls
  WARMUP_RUNS=1
  REPEAT=3
  GPU_MEMORY_UTILIZATION=0.9
  MAX_MODEL_LEN=auto
  MAX_NUM_BATCHED_TOKENS=auto
  ATTENTION_BACKEND=CUSTOM
  DISABLE_BEAM_GRAPH=0
  ENFORCE_EAGER=0
  ENABLE_PREFIX_CACHING=0
  CONSTRAINT_TRIPLES=        existing real triples; empty generates synthetic
  SYNTHETIC_TOKEN_LAYOUT=spread  spread or low
  TIMELINE=0
  TIMELINE_OUTPUT=           default under OUT_DIR when TIMELINE=1
  TIMELINE_REQUESTS=1
  TIMELINE_REPEAT=REPEAT
  TIMELINE_RECORD_SHAPES=0
  TIMELINE_WITH_STACK=0
  TIMELINE_WITH_MEMORY=0
  OUT_DIR=repo/results/beam_search_v1

Extra arguments are forwarded to benchmark_v1.py.
EOF
  exit 0
fi

MODEL="${MODEL:-${MODEL_DIR:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "MODEL is required; set it to a local checkpoint or model ID." >&2
  exit 2
fi

TOKEN_ID_START="${TOKEN_ID_START:-}"
TOKEN_ID_END="${TOKEN_ID_END:-}"
WORKLOAD_SEED="${WORKLOAD_SEED:-0}"
WORKLOAD_JSONL="${WORKLOAD_JSONL:-}"
if [[ -z "${TOKEN_ID_START}" || -z "${TOKEN_ID_END}" ]]; then
  echo "Set TOKEN_ID_START and TOKEN_ID_END for the shared input/output interval." >&2
  exit 2
fi
if [[ -n "${WORKLOAD_JSONL}" && ! -f "${WORKLOAD_JSONL}" ]]; then
  echo "Workload file not found: ${WORKLOAD_JSONL}" >&2
  exit 2
fi

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3.11}"
INSTALL_MODE="${INSTALL_MODE:-auto}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/beam_search_v1}"
CONTEXT_LEN="${CONTEXT_LEN:-5000}"
BEAM_STEPS="${BEAM_STEPS:-${DECODE_STEPS:-3}}"
BEAM_WIDTH="${BEAM_WIDTH:-256}"
REQUESTS="${REQUESTS:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
REPEAT="${REPEAT:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUSTOM}"
DISABLE_BEAM_GRAPH="${DISABLE_BEAM_GRAPH:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
CONSTRAINT_TRIPLES="${CONSTRAINT_TRIPLES:-}"
CONSTRAINT_ARTIFACT="${CONSTRAINT_ARTIFACT:-}"
SYNTHETIC_TOKEN_LAYOUT="${SYNTHETIC_TOKEN_LAYOUT:-spread}"
TIMELINE="${TIMELINE:-0}"
TIMELINE_OUTPUT="${TIMELINE_OUTPUT:-}"
TIMELINE_REQUESTS="${TIMELINE_REQUESTS:-1}"
TIMELINE_REPEAT="${TIMELINE_REPEAT:-${REPEAT}}"
TIMELINE_RECORD_SHAPES="${TIMELINE_RECORD_SHAPES:-0}"
TIMELINE_WITH_STACK="${TIMELINE_WITH_STACK:-0}"
TIMELINE_WITH_MEMORY="${TIMELINE_WITH_MEMORY:-0}"

case "${INSTALL_MODE}" in
  auto|always|skip) ;;
  *)
    echo "INSTALL_MODE must be auto, always, or skip; got: ${INSTALL_MODE}" >&2
    exit 2
    ;;
esac

echo "[1/3] Prepare vllm-gr environment: ${VENV_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if ! "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"; then
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
      echo "Failed to create virtual environment: ${VENV_DIR}" >&2
      exit 2
    fi
    echo "venv was created without pip; bootstrapping pip." >&2
  fi
fi
VENV_PYTHON="${VENV_DIR}/bin/python"
if ! "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
  GET_PIP="${VENV_DIR}/get-pip.py"
  curl -fsSL "${GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}" -o "${GET_PIP}"
  "${VENV_PYTHON}" "${GET_PIP}"
fi

venv_has_checkout() {
  PYTHONPATH="" "${VENV_PYTHON}" - "${REPO_ROOT}" <<'PY' >/dev/null 2>&1
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve()
spec = importlib.util.find_spec("vllm_gr")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(1)
source = Path(next(iter(spec.submodule_search_locations))).resolve()
if source != repo / "vllm_gr":
    raise SystemExit(1)
importlib.metadata.version("vllm")
importlib.metadata.version("vllm-gr")
PY
}

if [[ "${INSTALL_MODE}" == "always" ]] || {
  [[ "${INSTALL_MODE}" == "auto" ]] && ! venv_has_checkout
}; then
  VLLM_GR_TARGET_DEVICE=cuda "${VENV_PYTHON}" -m pip install -e "${REPO_ROOT}"
elif [[ "${INSTALL_MODE}" == "skip" ]] && ! venv_has_checkout; then
  echo "INSTALL_MODE=skip but the venv does not contain this checkout." >&2
  exit 2
fi

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
if ! mkdir -p "${CACHE_ROOT}" 2>/dev/null || [[ ! -w "${CACHE_ROOT}" ]]; then
  CACHE_ROOT="${OUT_DIR}/.cache"
  mkdir -p "${CACHE_ROOT}"
  echo "Cache root is not writable; using ${CACHE_ROOT}" >&2
fi
# Keep the benchmark independent from inherited container cache paths, which
# are frequently mounted read-only or owned by a different user.
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}" "${XDG_CACHE_HOME}" "${HF_HUB_CACHE}" \
  "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

SUFFIX="ctx${CONTEXT_LEN}_beam${BEAM_WIDTH}_req${REQUESTS}_steps${BEAM_STEPS}"
OUTPUT_JSON="${OUT_DIR}/vllm_gr_${SUFFIX}.json"
GENERATE_WORKLOAD=0
if [[ -z "${WORKLOAD_JSONL}" ]]; then
  GENERATE_WORKLOAD=1
  WORKLOAD_JSONL="${OUT_DIR}/vllm_gr_${SUFFIX}_ids${TOKEN_ID_START}-${TOKEN_ID_END}_seed${WORKLOAD_SEED}.jsonl"
fi
if [[ -z "${CONSTRAINT_ARTIFACT}" ]]; then
  CONSTRAINT_ARTIFACT="${OUT_DIR}/constraint_table_beam${BEAM_WIDTH}.safetensors"
fi
if [[ "${TIMELINE}" == "1" && -z "${TIMELINE_OUTPUT}" ]]; then
  TIMELINE_OUTPUT="${OUT_DIR}/vllm_gr_${SUFFIX}_timeline.json.gz"
fi

ARGS=(
  --model "${MODEL}"
  --output-json "${OUTPUT_JSON}"
  --workload-jsonl "${WORKLOAD_JSONL}"
  --token-id-start "${TOKEN_ID_START}"
  --token-id-end "${TOKEN_ID_END}"
  --constraint-artifact "${CONSTRAINT_ARTIFACT}"
  --synthetic-token-layout "${SYNTHETIC_TOKEN_LAYOUT}"
  --context-len "${CONTEXT_LEN}"
  --beam-steps "${BEAM_STEPS}"
  --beam-width "${BEAM_WIDTH}"
  --requests "${REQUESTS}"
  --warmup-runs "${WARMUP_RUNS}"
  --repeat "${REPEAT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)
if [[ -n "${CONSTRAINT_TRIPLES}" ]]; then
  ARGS+=(--constraint-triples "${CONSTRAINT_TRIPLES}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [[ "${DISABLE_BEAM_GRAPH}" == "1" ]]; then
  ARGS+=(--disable-beam-graph)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
fi
if [[ -n "${TIMELINE_OUTPUT}" ]]; then
  ARGS+=(
    --timeline-output "${TIMELINE_OUTPUT}"
    --timeline-requests "${TIMELINE_REQUESTS}"
    --timeline-repeat "${TIMELINE_REPEAT}"
  )
fi
if [[ "${TIMELINE_RECORD_SHAPES}" == "1" ]]; then
  ARGS+=(--timeline-record-shapes)
fi
if [[ "${TIMELINE_WITH_STACK}" == "1" ]]; then
  ARGS+=(--timeline-with-stack)
fi
if [[ "${TIMELINE_WITH_MEMORY}" == "1" ]]; then
  ARGS+=(--timeline-with-memory)
fi

if [[ "${GENERATE_WORKLOAD}" == "1" ]]; then
  echo "[2/3] Generate input IDs in [${TOKEN_ID_START}, ${TOKEN_ID_END}): ${WORKLOAD_JSONL}"
  "${VENV_PYTHON}" benchmarks/qwen/make_workload.py \
    --model-dir "${MODEL}" \
    --context-len "${CONTEXT_LEN}" \
    --requests "${REQUESTS}" \
    --token-id-start "${TOKEN_ID_START}" \
    --token-id-end "${TOKEN_ID_END}" \
    --seed "${WORKLOAD_SEED}" \
    --output-jsonl "${WORKLOAD_JSONL}"
else
  echo "[2/3] Reuse workload: ${WORKLOAD_JSONL}"
fi

echo "[3/3] Run offline GRLLM.beam_search_v1 benchmark: ${OUTPUT_JSON}"
"${VENV_PYTHON}" benchmarks/qwen/benchmark_v1.py "${ARGS[@]}" "$@"

echo "Done."
echo "Result JSON: ${OUTPUT_JSON}"
if [[ -n "${TIMELINE_OUTPUT}" ]]; then
  echo "Timeline: ${TIMELINE_OUTPUT}"
fi
