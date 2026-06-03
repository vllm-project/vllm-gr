#!/bin/bash

# This script is meant to be sourced.
# It provides utilities to allocate and manage local NPU resources.

function is_available_npu() {
    local i=$1
    local mem_npu=$2

    # npu-smi doesn't have a nice query format like nvidia-smi, so we parse the text output.
    local smi_output
    smi_output=$(npu-smi info 2>/dev/null) || return 1

    # Check 1: No running processes. This is a very strong signal of availability.
    if ! echo "$smi_output" | grep -q "No running processes found in NPU $i"; then
        return 1
    fi

    # Extract the two-line block for the NPU
    # The grep pattern looks for a line starting with "| " followed by the NPU id and a space.
    local npu_block
    npu_block=$(echo "$smi_output" | grep -A 1 "^| ${i} ")

    if [ -z "$npu_block" ]; then
        return 1
    fi

    # Parse health from the first line
    local health
    health=$(echo "$npu_block" | head -n 1 | awk -F'|' '{print $3}' | tr -d ' ')

    # Parse AICore and HBM usage from the second line
    local stats_line
    stats_line=$(echo "$npu_block" | tail -n 1 | awk -F'|' '{print $4}')
    local aicore
    aicore=$(echo "$stats_line" | awk '{print $1}')
    local mem_used
    mem_used=$(echo "$stats_line" | awk '{print $2}')

    if [ -z "$health" ] || [ -z "$aicore" ] || [ -z "$mem_used" ]; then
        return 1
    fi

    # CRITERIA CHECK:
    # - Is health OK?
    # - Is AICore utilization 0?
    # - Is HBM memory used < mem_npu limit?
    if [ "$health" = "OK" ] && [ "$aicore" -eq 0 ] && [ "$mem_used" -lt "$mem_npu" ]; then
        return 0
    fi
    return 1
}

function get_npu() {
    local mem_npu=${1:-16}
    local timeout_secs=${2:-0} # 0 means wait indefinitely

    local npu_count
    npu_count=$(npu-smi info 2>/dev/null | grep "^| [0-9]" | awk '{print $2}' | sort -u | wc -l)
    if [ -z "$npu_count" ] || [ "$npu_count" -eq 0 ]; then echo "No NPUs available"; return 1; fi

    local start_time=$(date +%s)

    while true; do
        # Loop through each NPU index
        for (( i=0; i<npu_count; i++ ))
        do
            if is_available_npu "$i" "$mem_npu"; then
                echo $i
                return 0
            fi
        done

        if [ "$timeout_secs" -gt 0 ]; then
            local current_time=$(date +%s)
            if [ $((current_time - start_time)) -ge "$timeout_secs" ]; then
                echo "Timeout reached"
                return 1
            fi
        fi
        sleep 5
    done
}

function get_npus() {
    local num_npus=${1:-1}
    local mem_npu=${2:-16}
    local timeout_secs=${3:-0} # 0 means wait indefinitely

    local npu_count
    npu_count=$(npu-smi info 2>/dev/null | grep -c "^| [0-9]" || true)
    if [ -z "$npu_count" ] || [ "$npu_count" -eq 0 ]; then return 1; fi

    local start_time=$(date +%s)

    while true; do
        local idle_npus=""
        local found=0

        # Loop through each NPU index
        for (( i=0; i<npu_count; i++ ))
        do
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
                echo "Timeout reached"
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
# The `is_available_npu` function, which checks for running processes via
# `npu-smi`, will NOT see this lock. Therefore, this locking mechanism is
# NOT effective for preventing race conditions between scripts using `get_npu`.
# A robust solution would require a program that allocates NPU resources
# (e.g., memory) to make it appear "in-use" to `npu-smi`.
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
        pid=$(lock_npu "$npu")
        if [ -z "$pid" ]; then
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
    IFS=',' read -ra pids <<< "$pids_list"
    for pid in "${pids[@]}"; do
        unlock_npu "$pid"
    done
}
