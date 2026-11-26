#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# Build the ROCm CI image for this commit and print its name on the last line.
#
#   VLLM_GR_ROCM_ARCH                 GPU architecture (default gfx1100)
#   VLLM_GR_ROCM_CI_IMAGE_REPOSITORY  image repository (default vllm-gr-ci-rocm)
#   VLLM_GR_ROCM_CI_CACHE_REF         optional registry cache shared by agents
#   VLLM_GR_ROCM_CI_PUSH=1            push the image to the repository
#   VLLM_GR_GITHUB_API                GitHub REST API or API proxy used to fetch
#                                     vLLM and FlashAttention sources
#   VLLM_GR_GITHUB_TOKEN_FILE         token file passed as a build secret
#   VLLM_GR_ROCM_BASE_IMAGE           ROCm PyTorch base image, for example a copy
#                                     in a private registry
#   VLLM_GR_ROCM_APT_MIRROR           Ubuntu package mirror used in the image
#   VLLM_GR_ROCM_PIP_INDEX_URL        Python package index used in the image
#
# The vLLM and FlashAttention layers are rebuilt only when their inputs change,
# so a commit normally rebuilds just the vLLM-GR layer. A single agent uses the
# local Docker layer cache; set VLLM_GR_ROCM_CI_CACHE_REF when agents are
# ephemeral (for example cloud instances) or share one cache.

set -euo pipefail

: "${BUILDKITE_COMMIT:?Buildkite must provide the commit SHA}"

arch="${VLLM_GR_ROCM_ARCH:-gfx1100}"
repository="${VLLM_GR_ROCM_CI_IMAGE_REPOSITORY:-vllm-gr-ci-rocm}"
image="${repository}:${arch}-${BUILDKITE_COMMIT}"
dockerfile="docker/Dockerfile.ci.rocm"

args=(--file "$dockerfile" --build-arg "PYTORCH_ROCM_ARCH=${arch}" --tag "$image")
if [ -n "${MAX_JOBS:-}" ]; then
    args+=(--build-arg "MAX_JOBS=${MAX_JOBS}")
fi
if [ -n "${VLLM_GR_ROCM_PIP_INDEX_URL:-}" ]; then
    args+=(--build-arg "PIP_INDEX_URL=${VLLM_GR_ROCM_PIP_INDEX_URL}")
fi
if [ -n "${VLLM_GR_ROCM_APT_MIRROR:-}" ]; then
    args+=(--build-arg "APT_MIRROR=${VLLM_GR_ROCM_APT_MIRROR}")
fi
if [ -n "${VLLM_GR_ROCM_BASE_IMAGE:-}" ]; then
    args+=(--build-arg "BASE_IMAGE=${VLLM_GR_ROCM_BASE_IMAGE}")
fi
if [ -n "${VLLM_GR_GITHUB_API:-}" ]; then
    args+=(--build-arg "GITHUB_API=${VLLM_GR_GITHUB_API}")
fi
if [ -n "${VLLM_GR_GITHUB_TOKEN_FILE:-}" ]; then
    # A build secret is not stored in the image or its build cache.
    args+=(--secret "id=github_token,src=${VLLM_GR_GITHUB_TOKEN_FILE}")
fi

echo "Building ${image}" >&2
if [ -n "${VLLM_GR_ROCM_CI_CACHE_REF:-}" ]; then
    cache_ref="${VLLM_GR_ROCM_CI_CACHE_REF}-${arch}"
    builder="vllm-gr-rocm-builder"
    docker buildx inspect "$builder" >/dev/null 2>&1 \
        || docker buildx create --name "$builder" --driver docker-container >/dev/null
    docker buildx build --builder "$builder" --load --progress=plain \
        --cache-from "type=registry,ref=${cache_ref}" \
        --cache-to "type=registry,ref=${cache_ref},mode=max,compression=zstd" \
        "${args[@]}" . >&2
else
    DOCKER_BUILDKIT=1 docker build --progress=plain "${args[@]}" . >&2
fi

if [ "${VLLM_GR_ROCM_CI_PUSH:-0}" = 1 ]; then
    docker push "$image" >&2
fi
echo "$image"
