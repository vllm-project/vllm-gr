#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# Keep credentials out of Bash traces even when invoked with bash -x.
set +x
set -euo pipefail

source "./tools/devices/resource.sh"
idle_gpu=$(get_gpu 4 15)


if [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 4MiB memory, and error-free."
    exit 1
fi

lock_pid=
secret_file=
act_pid=
pending_signal=
signal_status=0
wait_interrupted=0

forward_pending_signal() {
    if [[ -n "$act_pid" && -n "$pending_signal" ]]; then
        local signal=$pending_signal
        pending_signal=
        kill -s "$signal" "$act_pid" 2>/dev/null || true
    fi
}

# Invoked by the INT and TERM traps.
# shellcheck disable=SC2329
handle_signal() {
    if (( signal_status == 0 )); then
        signal_status=$2
    fi
    wait_interrupted=1
    pending_signal=$1
    forward_pending_signal
}

# Invoked by the EXIT trap.
# shellcheck disable=SC2329
cleanup() {
    local status=$? cleanup_status=0
    if [[ -n "$secret_file" ]]; then
        rm -f -- "$secret_file" || cleanup_status=1
    fi
    if [[ -n "$lock_pid" ]]; then
        echo "Unlocking GPU $idle_gpu (PID: $lock_pid)..."
        unlock_gpu "$lock_pid" || cleanup_status=1
    fi
    if (( status == 0 )); then
        status=$cleanup_status
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

# Lock the GPU to prevent other scripts from using it
lock_pid=$(lock_gpu "$idle_gpu")
if [ -z "$lock_pid" ]; then
    echo "Failed to lock GPU $idle_gpu"
    exit 1
fi

cuda_dev=$idle_gpu
echo "running on device $cuda_dev"
token=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    secret_file=$(mktemp)
    printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > "$secret_file"
    token=("--secret-file" "$secret_file")
fi
if (( signal_status != 0 )); then
    exit "$signal_status"
fi

# Keep the launcher waitable by disabling job control before setsid.
# The separate session routes terminal signals through the wrapper while
# act keeps its inherited stdin.
set +m
setsid -- env --default-signal=INT,QUIT act "${token[@]}" \
    --container-options "--runtime=nvidia --gpus device=$cuda_dev -v ${HOME}/.cache/act-cache:/data/actions/cache" \
    --env NVIDIA_DRIVER_CAPABILITIES=all --artifact-server-path "${HOME}/artifacts" \
    -P self-hosted=ghcr.io/catthehacker/ubuntu:act-latest <&0 &
act_pid=$!
forward_pending_signal

while :; do
    wait_interrupted=0
    act_status=0
    wait "$act_pid" || act_status=$?
    if (( wait_interrupted == 0 )); then
        break
    fi
done
act_pid=
if (( signal_status != 0 )); then
    exit "$signal_status"
fi
exit "$act_status"
