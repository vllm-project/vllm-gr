#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

validate_failure_mode() {
    local variable_name="$1"
    local failure_mode="$2"

    case "$failure_mode" in
      blocking|warning)
        ;;
      *)
        echo "Error: ${variable_name} must be blocking or warning, got: ${failure_mode}" >&2
        return 2
        ;;
    esac
}
validate_accuracy_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local threshold="$3"
    local failure_mode="$4"
    local device="$5"
    local blocking_status="${6:-1}"

    validate_failure_mode "failure mode for ${metric_name}" "$failure_mode" || return $?
    if awk -v val="$metric_value" -v th="$threshold" \
      'BEGIN {exit !(val >= th)}'; then
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
    local blocking_status="${6:-1}"

    validate_failure_mode "failure mode for ${metric_name}" "$failure_mode" || return $?
    if awk -v val="$metric_value" -v th="$threshold" \
      'BEGIN {exit !(val <= th)}'; then
        echo "SUCCESS on ${device}: ${metric_name} (${metric_value}) is less than or equal to ${threshold}."
    elif [[ "$failure_mode" == "warning" ]]; then
        echo "::warning title=Benchmark performance regression::${metric_name} (${metric_value}) exceeds ${threshold} on ${device}."
    else
        echo "FAILURE on ${device}: ${metric_name} (${metric_value}) exceeds ${threshold}."
        return "$blocking_status"
    fi
}
