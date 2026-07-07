#!/bin/bash
set -e

# Run all tests recursively under tests/
python3 -m pytest -x --run-slow --durations=10 -s tests
