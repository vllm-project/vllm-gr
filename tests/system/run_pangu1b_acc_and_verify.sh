#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=tools/ci/benchmark_gate_helpers.sh
source "$ROOT_DIR/tools/ci/benchmark_gate_helpers.sh"

MODEL_PATH="${PANGU1B_MODEL_PATH:-${PANGU_MODEL_PATH:-}}"
DATASET_PATH="${PANGU1B_DATASET_PATH:-tests/resources/pangu1b/gsm8k_mini.json}"
NUM_PROMPTS="${PANGU1B_NUM_PROMPTS:-8}"
MIN_ACCURACY="${PANGU1B_OFFLINE_ACCURACY_THRESHOLD:-}"
FAILURE_MODE="${PANGU1B_OFFLINE_ACCURACY_FAILURE_MODE:-}"
DEVICE="${BENCHMARK_DEVICE:-unknown}"

if [[ -z "$MIN_ACCURACY" || -z "$FAILURE_MODE" ]]; then
  echo "Error: the offline Pangu accuracy policy was not resolved." >&2
  exit 2
fi
validate_failure_mode "PANGU1B_OFFLINE_ACCURACY_FAILURE_MODE" \
  "$FAILURE_MODE" || exit $?

result_json=$(mktemp "${TMPDIR:-/tmp}/pangu1b-offline.XXXXXX.json")
cleanup() {
  rm -f -- "$result_json"
}
trap cleanup EXIT

echo "Running Pangu 1B offline accuracy measurement"
echo "  model: $MODEL_PATH"
echo "  dataset: $DATASET_PATH"
echo "  num_prompts: $NUM_PROMPTS"
echo "  threshold: $MIN_ACCURACY ($FAILURE_MODE)"

PANGU1B_REQUIRE_RUNTIME=1 \
PANGU1B_MEASURE_ONLY=1 \
PANGU1B_OFFLINE_RESULT_JSON="$result_json" \
python -m pytest --run-slow --strict-test-levels -m model_test \
  tests/st/test_pangu1b_accuracy_st.py
execution_status=$?
if [[ "$execution_status" -ne 0 ]]; then
  echo "Pangu offline execution failed with exit code $execution_status." >&2
  exit "$execution_status"
fi

accuracy=$(python -c \
  'import json, math, sys; value=float(json.load(open(sys.argv[1], encoding="utf-8"))["accuracy"]); assert math.isfinite(value) and 0 <= value <= 1; print(value)' \
  "$result_json")
parse_status=$?
if [[ "$parse_status" -ne 0 || -z "$accuracy" ]]; then
  echo "Pangu offline accuracy result could not be parsed." >&2
  exit 1
fi

validate_accuracy_metric "Pangu offline accuracy" "$accuracy" "$MIN_ACCURACY" \
  "$FAILURE_MODE" "$DEVICE" 90 || exit $?
echo "Pangu 1B offline accuracy check completed."
