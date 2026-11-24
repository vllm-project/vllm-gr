#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


# This script is meant to be sourced.
# It provides utilities to allocate and manage local NPU resources.

chip_num() {
    # Run the npu-smi command and extract the chip count
    local output=$(npu-smi info -t board -i 0)
    local chip_count=$(echo "$output" | grep "Chip Count" | awk -F': *' '{print $2}' | tr -cd '0-9')
    # A board whose npu-smi output lacks "Chip Count" is treated as single-chip
    # so enumeration and the card/chip split use the same non-zero count.
    if [ -z "$chip_count" ] || [ "$chip_count" -le 0 ]; then
        chip_count=1
    fi
    echo "$chip_count"
}

npu_count() {
   local output=$(npu-smi info -l)
   local count=$(echo "$output" | grep "Total Count" | awk -F': *' '{print$2}' | tr -cd '0-9')
   echo "$count"
}

NPU_COUNT=$(npu_count)
CHIP_NUM=$(chip_num)

function is_available_npu() {
    local npu_id=$1
    local threshold=$2

    # Calculate card_id and chip_id
    local nchip=${CHIP_NUM}
    # A board whose npu-smi output lacks "Chip Count" is treated as a single
    # chip, so the card/chip arithmetic never divides by zero.
    if [ -z "$nchip" ] || [ "$nchip" -le 0 ]; then
        nchip=1
    fi

    local card_id=$(( npu_id / nchip ))
    local chip_id=$(( npu_id % nchip ))

    local usage_data
    usage_data=$(npu-smi info -t usages -i "$card_id" -c "$chip_id") || return 1
    # Missing or malformed telemetry must never make a device look idle.
    awk -F ':' -v threshold="$threshold" '
        $1 ~ /^[[:space:]]*HBM Capacity[[:space:]]*\(MB\)[[:space:]]*$/ {
            capacity = $2; capacities++
        }
        $1 ~ /^[[:space:]]*HBM Usage Rate[[:space:]]*\(%\)[[:space:]]*$/ {
            rate = $2; rates++
        }
        END {
            integer = "^[[:space:]]*[0-9]+[[:space:]]*$"
            exit !(capacities == 1 && rates == 1 && capacity ~ integer && rate ~ integer &&
                   capacity + 0 > 0 && rate + 0 <= 100 &&
                   int(capacity * rate / 100) <= threshold)
        }
    ' <<< "$usage_data"
}

# _npu_device_indices <total>
# Prints the logical NPU device ids to consider, one per line. Honors
# ASCEND_RT_VISIBLE_DEVICES (a comma-separated list of logical device ids, the
# same ids get_npu returns) so selection stays inside the devices the CANN
# runtime can see. Falls back to all ids when unset; an empty mask exposes none.
function _npu_device_indices() {
    local total=$1
    if ! [[ -v ASCEND_RT_VISIBLE_DEVICES ]]; then
        # Unset: consider every physical NPU.
        local i
        for (( i=0; i<total; i++ )); do
            echo "$i"
        done
        return 0
    fi
    if [ -z "$ASCEND_RT_VISIBLE_DEVICES" ]; then
        # Set but empty: the runtime exposes no devices.
        echo "ASCEND_RT_VISIBLE_DEVICES is set but empty: no devices visible" >&2
        return 1
    fi
    if [[ ! $ASCEND_RT_VISIBLE_DEVICES =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Invalid ASCEND_RT_VISIBLE_DEVICES: expected comma-separated non-negative integers" >&2
        return 1
    fi
    local idx found=0
    local -a indices
    IFS=',' read -ra indices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    for idx in "${indices[@]}"; do
        if [ "$idx" -lt "$total" ]; then
            echo "$((10#$idx))"
            found=$((found + 1))
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "No ASCEND_RT_VISIBLE_DEVICES id is within [0, $total)" >&2
        return 1
    fi
    return 0
}

function get_npu() {
    local mem_npu=${1:-3000}
    local timeout_secs=${2:-15} # seconds to wait; 0 waits indefinitely

    local ncount
    ncount=${NPU_COUNT}
    nchip=${CHIP_NUM}
    total=$(( ncount * nchip ))

    local indices
    indices=$(_npu_device_indices "$total") || return 1
    if [ -z "$indices" ]; then echo "No NPUs visible to CANN" >&2; return 1; fi

    local start_time=$(date +%s)

    while true; do
        # Loop through each visible NPU index
        for i in $indices; do
            if is_available_npu "$i" "$mem_npu"; then
                echo "$i"
                return 0
            fi
        done

        if [ "$timeout_secs" -gt 0 ]; then
            local current_time=$(date +%s)
            if [ $((current_time - start_time)) -ge "$timeout_secs" ]; then
                echo "Timeout reached" >&2
                return 1
            fi
        fi
        sleep 5
    done
}

function get_npus() {
    local num_npus=${1:-1}
    local mem_npu=${2:-16}
    local timeout_secs=${3:-15} # seconds to wait; 0 waits indefinitely

    local ncount=${NPU_COUNT}
    local nchip=${CHIP_NUM}
    local total=$(( ncount * nchip ))

    local indices
    indices=$(_npu_device_indices "$total") || return 1
    if [ -z "$indices" ]; then return 1; fi

    local start_time=$(date +%s)

    while true; do
        local idle_npus=""
        local found=0

        # Loop through each visible NPU index
        for i in $indices; do
            # Skip ids already selected (ASCEND_RT_VISIBLE_DEVICES may repeat them).
            if [[ ",$idle_npus," == *",$i,"* ]]; then
                continue
            fi
            if is_available_npu "$i" "$mem_npu"; then
                if [ -z "$idle_npus" ]; then
                    idle_npus="$i"
                else
                    idle_npus="${idle_npus},$i"
                fi
                found=$((found + 1))
                if [ "$found" -eq "$num_npus" ]; then
                    echo "$idle_npus"
                    return 0
                fi
            fi
        done

        if [ "$timeout_secs" -gt 0 ]; then
            local current_time=$(date +%s)
            if [ $((current_time - start_time)) -ge "$timeout_secs" ]; then
                echo "Timeout reached" >&2
                return 1
            fi
        fi
        sleep 5
    done
}

# lock_npu <npu_id>
# This is a STUB implementation. It starts a background sleep process and
# returns its PID.
# WARNING: This does NOT reserve the NPU in a way that is visible to `npu-smi`.
# The `is_available_npu` function queries HBM capacity and usage via
# `npu-smi`. A background sleep does not affect that query or provide
# per-device mutual exclusion. This stub does NOT prevent race conditions
# between scripts using `get_npu`; allocating a small amount of memory alone
# would not guarantee exclusive device selection.
function lock_npu() {
    local npu_id=$1
    local pid=0
    if [ -z "$npu_id" ]; then
        echo "Invalid NPU ID: argument is empty" >&2
        return 1
    fi
    if ! [[ "$npu_id" =~ ^[0-9]+$ ]]; then
        echo "Invalid NPU ID: $npu_id" >&2
        return 1
    fi
    # Start a background process that does nothing.
    nohup sleep infinity > /dev/null 2>&1 &
    pid=$!
    sleep 0.25
    echo "After sleep started" >&2
    echo $pid
}

# unlock_npu <pid>
# Kills the background process created by lock_npu.
function unlock_npu() {
    local pid=$1
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
}

function lock_npus() {
    local npus_list=$1
    local pids=""
    IFS=',' read -ra npus <<< "$npus_list"
    for npu in "${npus[@]}"; do
        local pid
        if ! pid=$(lock_npu "$npu") || [ -z "$pid" ]; then
            # Lock failed — roll back any NPUs locked so far
            if [ -n "$pids" ]; then
                unlock_npus "$pids"
            fi
            echo "" >&2
            echo "Failed to lock NPU $npu, rolled back previously locked NPUs." >&2
            return 1
        fi
        if [ -z "$pids" ]; then
            pids="$pid"
        else
            pids="${pids},${pid}"
        fi
    done
    echo "$pids"
}

function unlock_npus() {
    local pids_list=$1
    local -a lock_pids
    local pid status=0
    IFS=',' read -ra lock_pids <<< "$pids_list"
    for pid in "${lock_pids[@]}"; do
        unlock_npu "$pid" || status=1
    done
    return "$status"
}
