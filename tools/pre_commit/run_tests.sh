#!/bin/bash
set -e

# Run tests recursively under tests/ in one of three lanes:
#   cpu  only tests that need no CUDA device (safe on CPU-only hosts)
#   gpu  only tests that need a CUDA device
#   all  everything (previous behaviour; the default until the CPU lane is
#        green in CI, after which the card should switch to "gpu")
# Lanes are selected with the requires_gpu marker applied by tests/conftest.py
# from the classification in tests/gpu_lanes.py.
lane="${1:-all}"
case "$lane" in
    cpu)
        selection=(-m "not requires_gpu")
        # Also skips importing GPU-lane modules (conftest pytest_ignore_collect):
        # -m alone filters after import, which would abort collection on hosts
        # without the CUDA stack.
        export VLLM_GR_TEST_LANE=cpu
        ;;
    gpu) selection=(-m "requires_gpu") ;;
    all) selection=() ;;
    *)
        echo "usage: $0 [cpu|gpu|all]" >&2
        exit 2
        ;;
esac

python3 -m pytest -x --run-slow --durations=10 -s tests "${selection[@]}"
