#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm


# Ensure that *.excalidraw.png files have the excalidraw metadata
# embedded in them. This ensures they can be loaded back into
# the tool and edited in the future.

shopt -s nocasematch
for file in "$@"; do
    [[ "$file" == *.excalidraw.png ]] || continue
    if ! grep -Fq -- "excalidraw+json" "$file"; then
        echo "$file was not exported from excalidraw with 'Embed Scene' enabled." >&2
        exit 1
    fi
done
