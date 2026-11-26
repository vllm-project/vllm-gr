#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# Build the ROCm CI image and run tools/ci/run_rocm_tests.sh in it on an AMD
# Buildkite agent. Configure each agent through its environment:
#
#   VLLM_GR_ROCM_RENDER_DEVICES  render nodes this agent may use, separated by
#                                spaces (default: every /dev/dri/renderD*)
#   VLLM_GR_ROCM_DEVICE_ID       fixed GPU index among those nodes; unset
#                                waits for an idle GPU and locks it
#   VLLM_GR_TEST_MODEL           Hugging Face model ID (default OneRec-1.7B)
#   HF_ENDPOINT, HF_TOKEN        passed to the tests when set
#   VLLM_GR_ROCM_CI_OUTPUT       log directory inside the checkout
#                                (default ci-output/rocm); it is recreated
#
# Like the CUDA jobs, the job downloads the model inside the container. Only
# the checkout is mounted, so the container cannot reach other host files.
# The exit status of the tests is preserved; 75 means no idle GPU or no model
# (retry later).

set -euo pipefail

output="${VLLM_GR_ROCM_CI_OUTPUT:-ci-output/rocm}"
case "$output" in
    /* | *..*) echo "VLLM_GR_ROCM_CI_OUTPUT must be a relative path inside the checkout" >&2; exit 1 ;;
esac

image=$(bash .buildkite/common/scripts/build-ci-image-rocm.sh | tail -n 1)
# Per-commit tags would accumulate; the shared layers stay in the cache.
trap 'docker image rm "$image" >/dev/null 2>&1 || true' EXIT

args=(--rm --device /dev/kfd)
# shellcheck disable=SC2206 # Word splitting expands the configured node list.
nodes=(${VLLM_GR_ROCM_RENDER_DEVICES:-/dev/dri/renderD*})
for node in "${nodes[@]}"; do
    [ -e "$node" ] || { echo "Render node $node does not exist" >&2; exit 1; }
    args+=(--device "$node")
done
for group in video render; do
    if gid=$(getent group "$group" | cut -d: -f3) && [ -n "$gid" ]; then
        args+=(--group-add "$gid")
    fi
done

args+=(--user "$(id -u):$(id -g)" -e HOME=/tmp/home --shm-size 16g)
args+=(-v "$PWD:/workspace" -w /workspace)
for name in VLLM_GR_TEST_MODEL HF_ENDPOINT HF_TOKEN; do
    if [ -n "${!name:-}" ]; then
        args+=(-e "$name")
    fi
done
args+=(--name "vllm-gr-rocm-${BUILDKITE_JOB_ID:-$$}")

test_args=(--output "/workspace/$output")
if [ -n "${VLLM_GR_ROCM_DEVICE_ID:-}" ]; then
    test_args+=(--device-id "$VLLM_GR_ROCM_DEVICE_ID")
fi

rm -rf -- "$output"
docker run "${args[@]}" "$image" \
    bash -c 'mkdir -p "$HOME" && exec tools/ci/run_rocm_tests.sh "$@"' run-rocm-ci "${test_args[@]}"
