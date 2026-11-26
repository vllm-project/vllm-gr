#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


# This script is meant to be sourced by resource.sh.
# It provides AMD ROCm telemetry for local GPU selection.
#
# Device indices follow the KFD topology order of GPU nodes, which is the HIP
# enumeration order used by HIP_VISIBLE_DEVICES. amd-smi may list devices in
# a different (PCI address) order, so it is not used for indexing.
#
# Usage is read from sysfs, which is shared by every container on the host:
# a GPU is idle when the VRAM held by all KFD processes is below the threshold
# and its busy percentage drops to zero within a few samples. Total VRAM usage
# is not used, because the driver reserves memory even on an idle GPU. A single
# busy sample is not used either, because a GPU that drives a display reports
# short bursts of graphics activity without any compute process.

: "${VLLM_GR_KFD_TOPOLOGY:=/sys/class/kfd/kfd/topology/nodes}"
: "${VLLM_GR_KFD_PROC:=/sys/class/kfd/kfd/proc}"
: "${VLLM_GR_PCI_DEVICES:=/sys/bus/pci/devices}"
# GPUs with less VRAM, such as integrated GPUs, are never selected.
: "${VLLM_GR_ROCM_MIN_VRAM_MIB:=4096}"

# Prints "<gpu_id> <pci_address>" for each GPU node in HIP order.
function _rocm_gpu_nodes() {
    local node props simd location domain gpu_id
    local -a nodes=()
    for node in "$VLLM_GR_KFD_TOPOLOGY"/*; do
        [[ -f $node/properties ]] && nodes+=("$(basename "$node")")
    done
    [ "${#nodes[@]}" -gt 0 ] || return 1
    local id
    while read -r id; do
        node="$VLLM_GR_KFD_TOPOLOGY/$id"
        props="$node/properties"
        simd=$(awk '$1 == "simd_count" {print $2}' "$props")
        [[ $simd =~ ^[0-9]+$ ]] && [ "$simd" -gt 0 ] || continue
        location=$(awk '$1 == "location_id" {print $2}' "$props")
        domain=$(awk '$1 == "domain" {print $2}' "$props")
        gpu_id=$(cat "$node/gpu_id" 2>/dev/null) || continue
        [[ $location =~ ^[0-9]+$ && $domain =~ ^[0-9]+$ && $gpu_id =~ ^[0-9]+$ ]] || continue
        printf '%s %04x:%02x:%02x.%x\n' "$gpu_id" "$domain" "$((location >> 8))" \
            "$(((location >> 3) & 31))" "$((location & 7))"
    done < <(printf '%s\n' "${nodes[@]}" | sort -n)
}

function rocm_gpu_count() {
    local count
    count=$(_rocm_gpu_nodes | wc -l) || return 1
    [ "$count" -gt 0 ] || return 1
    echo "$count"
}

# rocm_gpu_pci_address <index>
function rocm_gpu_pci_address() {
    _rocm_gpu_nodes | awk -v want="$(($1 + 1))" 'NR == want {print $2; found = 1} END {exit !found}'
}

# is_available_rocm <index> <max_compute_vram_mib>
function is_available_rocm() {
    local i=$1
    local mem_gpu=$2
    local node gpu_id address
    node=$(_rocm_gpu_nodes | awk -v want="$((i + 1))" 'NR == want') || return 1
    [ -n "$node" ] || return 1
    read -r gpu_id address <<< "$node"

    local device="$VLLM_GR_PCI_DEVICES/$address"
    local total
    total=$(cat "$device/mem_info_vram_total" 2>/dev/null) || return 1
    [[ $total =~ ^[0-9]+$ ]] || return 1
    [ $((total >> 20)) -ge "$VLLM_GR_ROCM_MIN_VRAM_MIB" ] || return 1

    # Sustained activity: busy in every sample.
    local busy sample idle=false
    for sample in 1 2 3; do
        busy=$(cat "$device/gpu_busy_percent" 2>/dev/null) || return 1
        [[ $busy =~ ^[0-9]+$ ]] || return 1
        if [ "$busy" -eq 0 ]; then
            idle=true
            break
        fi
        [ "$sample" -lt 3 ] && command sleep 0.1
    done
    [ "$idle" = true ] || return 1

    # Sum the VRAM that every KFD process holds on this GPU.
    local used=0 value file
    for file in "$VLLM_GR_KFD_PROC"/*/"vram_$gpu_id"; do
        [ -e "$file" ] || continue
        if ! value=$(cat "$file" 2>/dev/null); then
            # The process may have exited; an unreadable live entry is unknown usage.
            [ -e "$file" ] && return 1
            continue
        fi
        [[ $value =~ ^[0-9]+$ ]] || return 1
        used=$((used + value))
    done
    [ $((used >> 20)) -lt "$mem_gpu" ]
}

# _rocm_device_indices <gpu_count>
# Honors HIP_VISIBLE_DEVICES, then CUDA_VISIBLE_DEVICES, like the HIP runtime.
function _rocm_device_indices() {
    if [[ -v ROCR_VISIBLE_DEVICES ]]; then
        echo "ROCR_VISIBLE_DEVICES renumbers devices; set HIP_VISIBLE_DEVICES instead" >&2
        return 1
    fi
    if [[ -v HIP_VISIBLE_DEVICES ]]; then
        _visible_device_indices HIP_VISIBLE_DEVICES "$1"
    else
        _visible_device_indices CUDA_VISIBLE_DEVICES "$1"
    fi
}
