#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

validate_accuracy_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local threshold="$3"
    local failure_mode="$4"
    local device="$5"
    local blocking_status="$6"

    if awk -v val="$metric_value" -v th="$threshold" 'BEGIN {exit !(val >= th)}'; then
        echo "SUCCESS on ${device}: ${metric_name} (${metric_value}) is greater than or equal to ${threshold}."
    elif [[ "$failure_mode" == "warning" ]]; then
        echo "::warning title=Benchmark accuracy regression::${metric_name} (${metric_value}) is less than ${threshold} on ${device}."
    else
        echo "FAILURE on ${device}: ${metric_name} (${metric_value}) is less than ${threshold}."
        return "$blocking_status"
    fi
}

validate_performance_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local threshold="$3"
    local failure_mode="$4"
    local device="$5"
    local blocking_status="$6"

    if awk -v val="$metric_value" -v th="$threshold" 'BEGIN {exit !(val <= th)}'; then
        echo "SUCCESS on ${device}: ${metric_name} (${metric_value}) is less than or equal to ${threshold}."
    elif [[ "$failure_mode" == "warning" ]]; then
        echo "::warning title=Benchmark performance regression::${metric_name} (${metric_value}) exceeds ${threshold} on ${device}."
    else
        echo "FAILURE on ${device}: ${metric_name} (${metric_value}) exceeds ${threshold}."
        return "$blocking_status"
    fi
}

handle_online_test_status() {
    local online_status="$1"
    local accuracy_failure_status="$2"
    local performance_failure_status="$3"
    local non_accuracy_failure_mode="$4"
    local policy_name="$5"
    local device="$6"

    if [[ "$online_status" -eq 0 ]]; then
        return 0
    fi
    if [[ "$online_status" -eq "$accuracy_failure_status" ]]; then
        echo "OneRec online inference accuracy validation failed."
        return "$online_status"
    fi
    if [[ "$online_status" -eq "$performance_failure_status" ]]; then
        echo "OneRec online inference performance validation failed."
        return "$online_status"
    fi
    if [[ "$non_accuracy_failure_mode" != "warning" ]]; then
        return "$online_status"
    fi

    local warning_message
    warning_message="OneRec online inference experiment had a non-accuracy failure with exit code $online_status under the $policy_name policy (${device})."
    echo "::warning title=OneRec online non-accuracy failure::${warning_message}"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo ""
            echo "> [!WARNING]"
            echo "> $warning_message"
        } >> "$GITHUB_STEP_SUMMARY"
    fi
}
