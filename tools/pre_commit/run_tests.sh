#!/bin/bash
set -e

# Run all tests recursively under tests/.
# An optional marker filter is read from VLLM_GR_TEST_MARKER so callers can
# split the suite across machines (e.g. "not gpu", "npu", "not gpu and not npu").
pytest_args=(-x --run-slow --durations=10 -s)
if [ -n "${VLLM_GR_TEST_MARKER:-}" ]; then
    pytest_args+=(-m "${VLLM_GR_TEST_MARKER}")
fi
python3 -m pytest "${pytest_args[@]}" tests
