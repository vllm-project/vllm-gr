#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
set -uo pipefail

usage() {
  echo "Usage: $0 {pr|pr-device|l1|l2|l3|l4|l5} [pytest arguments...]" >&2
}

level=${1:-pr}
if [[ $# -gt 0 ]]; then shift; fi
case "$level" in
  pr) expression='l1 or l2' ;;
  pr-device) expression='l2 or (l1 and (gpu_test or npu_test))' ;;
  l1|l2|l3|l4|l5) expression="$level" ;;
  *) usage; exit 2 ;;
esac

# Hardware tiers describe eligibility, while levels describe CI cost. Run
# CPU tests alongside tests for the selected accelerator. A function-level
# accelerator marker overrides a module-level cpu_test marker.
cpu_expression='cpu_test and not gpu_test and not npu_test'
case "${VLLM_GR_TARGET_DEVICE:-}" in
  cuda) expression="($expression) and (($cpu_expression) or gpu_test)" ;;
  npu) expression="($expression) and (($cpu_expression) or npu_test)" ;;
  cpu) expression="($expression) and ($cpu_expression)" ;;
  '') ;;  # Preserve unfiltered local runs when no platform was selected.
  *) echo "Unsupported VLLM_GR_TARGET_DEVICE: $VLLM_GR_TARGET_DEVICE" >&2; exit 2 ;;
esac

pytest_args=(-x --run-slow --strict-test-levels --durations=10 -s -m "$expression")

# The opt-in tiny-model tests inspect the in-process vLLM engine directly.
# Isolate them from tests that exercise the multiprocess engine, preserving
# the same pytest arguments and level selection in both invocations.
if [[ "${VLLM_GR_RUN_MODEL_SMOKE:-}" == 1 ]]; then
  selection_plugin=(-p tools.ci.inproc_test_selection)
  VLLM_GR_INPROC_TEST_SELECTION=exclude python3 -m pytest "${selection_plugin[@]}" \
    "${pytest_args[@]}" tests "$@"
  status=$?
  # Exit code 5 means no tests matched; the isolated subset may still match.
  if [[ $status -ne 0 && $status -ne 5 ]]; then
    exit "$status"
  fi
  VLLM_GR_INPROC_TEST_SELECTION=include VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python3 -m pytest "${selection_plugin[@]}" "${pytest_args[@]}" tests "$@"
  inproc_status=$?
  if [[ $inproc_status -ne 0 && $inproc_status -ne 5 ]]; then
    exit "$inproc_status"
  fi
  if [[ $status -eq 0 || $inproc_status -eq 0 ]]; then
    status=0
  else
    status=5
  fi
else
  python3 -m pytest "${pytest_args[@]}" tests "$@"
  status=$?
fi
if [[ "$level" == l5 && $status -eq 5 ]]; then
  message='No L5 tests are currently collected; no stability tests were run.'
  echo "::warning::$message"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "$message" >> "$GITHUB_STEP_SUMMARY"
  fi
  exit 0
fi
exit "$status"
