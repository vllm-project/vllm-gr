#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

set -euo pipefail

warn() {
  echo "::warning::$1"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "$1" >> "$GITHUB_STEP_SUMMARY"
  fi
}

# The probe discovers and locks physical devices. An isolated runner must be
# explicitly configured before it may inspect or lock any of those devices.
if [[ "${VLLM_GR_ISOLATED_RUNNER:-false}" != true ]]; then
  warn 'L2 device resource probe did not run: VLLM_GR_ISOLATED_RUNNER is not true.'
  exit 0
fi

report="$(mktemp)"
trap 'rm -f -- "$report"' EXIT
set +e
bash tests/development/test_resource.sh 2>&1 | tee "$report"
status=${PIPESTATUS[0]}
set -e
if grep -q 'SKIP' "$report"; then
  warn 'One or more L2 device resource probe scenarios did not run; see the log for the reason.'
fi
exit "$status"
