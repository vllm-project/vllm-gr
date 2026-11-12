#!/bin/bash
set -e

# CI can select a level without changing the documented local full-suite
# default used by pre-commit and tools/run_test.sh.
if [[ -n "${VLLM_GR_TEST_LEVEL:-}" ]]; then
  exec bash tools/ci/run_test_level.sh "$VLLM_GR_TEST_LEVEL" "$@"
fi

exec python3 -m pytest -x --run-slow --durations=10 -s tests "$@"
