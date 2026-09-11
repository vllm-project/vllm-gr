#!/bin/bash
set -e

# Pangu model/API tests run in the dedicated offline and online CI gates.
# Their lightweight UnitHooks still run here, even when model paths are set.
python3 -m pytest -x --run-slow -m "not pangu_st" --durations=10 -s tests
