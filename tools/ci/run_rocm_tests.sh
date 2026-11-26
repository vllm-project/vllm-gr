#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# Run the AMD ROCm CI checks on one GPU of a possibly shared host.
#
#   tools/ci/run_rocm_tests.sh [--device-id ID] [--model DIR_OR_ID] [--skip-model]
#                              [--wait SECONDS] [--output DIR]
#
# Without --device-id, the script waits for an idle GPU using
# tools/devices/resource.sh. It locks the GPU for the whole run, so other jobs
# that use the same helpers skip it. Run it from a checkout inside the ROCm CI
# image (docker/Dockerfile.ci.rocm) with /dev/kfd and the GPU render nodes.
#
# The model is --model, else VLLM_GR_TEST_MODEL, else OpenOneRec/OneRec-1.7B.
# A Hugging Face ID is downloaded into the Hugging Face cache before a GPU is
# locked. Set HF_ENDPOINT for a mirror and HF_TOKEN when the model requires it.
#
# Exit status: 0 when every check passes, 1 when a check fails, and 75 when
# no idle GPU becomes available or the model cannot be fetched (retry later).

set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# Tests query git; the checkout may belong to another user than the container.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"

device_id=""
model="${VLLM_GR_TEST_MODEL:-OpenOneRec/OneRec-1.7B}"
skip_model=false
wait_secs=600
output=""
mem_th=18

usage() { sed -n '7,20p' "$0" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
    case "$1" in
        --device-id) device_id=$2; shift 2 ;;
        --model) model=$2; shift 2 ;;
        --skip-model) skip_model=true; shift ;;
        --wait) wait_secs=$2; shift 2 ;;
        --output) output=$2; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
output=${output:-$(mktemp -d "${TMPDIR:-/tmp}/vllm-gr-rocm-ci.XXXXXX")}
mkdir -p "$output"

# The examples need a local snapshot. Resolve a Hugging Face ID before locking
# a GPU, so a slow download does not hold one.
if [ "$skip_model" = false ] && [ ! -d "$model" ]; then
    echo "Fetching $model into the Hugging Face cache"
    if ! model=$(python3 -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "$model" 2> "$output/model_download.log" | tail -n 1) \
        || [ ! -d "$model" ]; then
        tail -n 5 "$output/model_download.log" >&2
        echo "Cannot fetch the model; see $output/model_download.log" >&2
        exit 75
    fi
fi
[ "$skip_model" = true ] || echo "Model snapshot: $model"

# shellcheck source=tools/devices/resource.sh
source tools/devices/resource.sh
if [ "${rocm_state:-false}" != true ]; then
    echo "No AMD ROCm GPU is visible: mount /dev/kfd and the GPU render nodes" >&2
    exit 1
fi

lock_pid=""
# shellcheck disable=SC2329
release() {
    if [ -n "$lock_pid" ]; then
        unlock_gpu "$lock_pid" || true
    fi
}
trap release EXIT

if [ -z "$device_id" ]; then
    if ! device_id=$(get_gpu "$mem_th" "$wait_secs"); then
        echo "No idle AMD GPU within ${wait_secs}s" >&2
        exit 75
    fi
fi
if ! lock_pid=$(lock_gpu "$device_id") || [ -z "$lock_pid" ]; then
    echo "Cannot lock GPU $device_id" >&2
    exit 75
fi
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=$device_id
echo "GPU $device_id ($(rocm_gpu_pci_address "$device_id")), lock PID $lock_pid, output $output"

declare -a results=()
failed=0
stage() {
    local name=$1
    shift
    echo "=== $name"
    local started=$SECONDS rc=0
    "$@" || rc=$?
    local outcome=PASSED
    if [ "$rc" -ne 0 ]; then
        outcome="FAILED (exit $rc)"
        failed=1
    fi
    results+=("$(printf '%-34s %s (%ss)' "$name" "$outcome" "$((SECONDS - started))")")
}

# shellcheck disable=SC2329
check_env() {
    (cd /tmp && vllm check-env --gr) | tee "$output/check_env.log"
    grep -q 'platform: rocm' "$output/check_env.log"
}

# shellcheck disable=SC2329
unit_tests() {
    python3 -m pytest -n 4 --durations=10 \
        -m '(l1) and (cpu_test and not gpu_test and not npu_test)' tests \
        > "$output/unit_tests.log" 2>&1
    local rc=$?
    tail -n 3 "$output/unit_tests.log"
    return "$rc"
}

# shellcheck disable=SC2329
device_tests() {
    python3 -m pytest tests/kernels/attention/test_rocm_flash_attn_adapter.py
}

# shellcheck disable=SC2329
offline() {
    local runner=$1
    VLLM_USE_V2_MODEL_RUNNER=$runner python3 examples/offline_inference/beam_search/quickstart.py \
        --model "$model" --max-model-len 2048 --gpu-memory-utilization 0.85 --print-beams 128 \
        > "$output/offline_v$runner.log" 2>&1
    local rc=$?
    grep -m1 'returned_beams' "$output/offline_v$runner.log" || true
    return "$rc"
}

# shellcheck disable=SC2329
same_beams() {
    local v1 v2
    v1=$(grep '^{"request"' "$output/offline_v0.log") || return 1
    v2=$(grep '^{"request"' "$output/offline_v1.log") || return 1
    [ "$v1" = "$v2" ]
}

# shellcheck disable=SC2329
accuracy() {
    python3 tools/ci/check_onerec_accuracy.py --model "$model" --log "$output/offline_v1.log"
}

# shellcheck disable=SC2329
http() {
    python3 examples/online_serving/beam_search/quickstart.py --model "$model" \
        --workdir "$output/http" > "$output/http.log" 2>&1
    local rc=$?
    tail -n 1 "$output/http.log"
    return "$rc"
}

stage "vllm check-env --gr" check_env
stage "L1 unit tests" unit_tests
stage "ROCm device tests" device_tests
if [ "$skip_model" = false ]; then
    stage "OneRec offline, ModelRunner V2" offline 1
    stage "OneRec offline, ModelRunner V1" offline 0
    stage "ModelRunner V1 equals V2" same_beams
    stage "Accuracy vs FP32 HF reference" accuracy
    stage "OneRec HTTP serving" http
fi

echo "=== Summary (GPU $device_id, output $output)"
printf '%s\n' "${results[@]}"
exit "$failed"
