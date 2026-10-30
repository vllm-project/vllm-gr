#!/usr/bin/env bash
set -euo pipefail

# Explicit device suite; CPU contracts and lint run in the hosted fast lane.
# Additional pytest arguments select evidence outputs and CPU-only exclusions.
python3 -m pytest -x --run-slow --durations=20 -s tests "$@"
