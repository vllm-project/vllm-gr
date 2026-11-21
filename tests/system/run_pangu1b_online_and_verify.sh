#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=tools/ci/benchmark_gate_helpers.sh
source "$ROOT_DIR/tools/ci/benchmark_gate_helpers.sh"

MODEL_PATH="${PANGU1B_MODEL_PATH:-${PANGU_MODEL_PATH:-}}"
MIN_ACCURACY="${PANGU1B_ONLINE_ACCURACY_THRESHOLD:-}"
FAILURE_MODE="${PANGU1B_ONLINE_ACCURACY_FAILURE_MODE:-}"
DEVICE="${BENCHMARK_DEVICE:-unknown}"

if [[ -z "$MIN_ACCURACY" || -z "$FAILURE_MODE" ]]; then
  echo "Error: the online Pangu accuracy policy was not resolved." >&2
  exit 2
fi
validate_failure_mode "PANGU1B_ONLINE_ACCURACY_FAILURE_MODE" \
  "$FAILURE_MODE" || exit $?

result_json=$(mktemp "${TMPDIR:-/tmp}/pangu1b-online.XXXXXX.json")
cleanup() {
  rm -f -- "$result_json"
}
trap cleanup EXIT

echo "Running Pangu 1B online service/API contract and accuracy measurement"
echo "  model: $MODEL_PATH"
echo "  threshold: $MIN_ACCURACY ($FAILURE_MODE)"

PANGU1B_REQUIRE_RUNTIME=1 \
PANGU1B_MEASURE_ONLY=1 \
PANGU1B_ONLINE_RESULT_JSON="$result_json" \
python -m pytest --run-slow --strict-test-levels -m model_test \
  tests/st/test_pangu1b_online_st.py
execution_status=$?
if [[ "$execution_status" -ne 0 ]]; then
  echo "Pangu online service/API execution failed with exit code $execution_status." >&2
  exit "$execution_status"
fi
if [[ ! -s "$result_json" ]]; then
  echo "Pangu online accuracy test produced no result; skipped tests are not accepted in CI." >&2
  exit 1
fi

accuracy=$(python -c \
  'import json, math, sys; value=float(json.load(open(sys.argv[1], encoding="utf-8"))["accuracy"]); assert math.isfinite(value) and 0 <= value <= 1; print(value)' \
  "$result_json")
parse_status=$?
if [[ "$parse_status" -ne 0 || -z "$accuracy" ]]; then
  echo "Pangu online accuracy result could not be parsed." >&2
  exit 1
fi

validate_accuracy_metric "Pangu online accuracy" "$accuracy" "$MIN_ACCURACY" \
  "$FAILURE_MODE" "$DEVICE" 91 || exit $?
echo "Pangu 1B online service/API and accuracy check completed."
