#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# WARNING: requires --yes; use only on GPUs allocated to your job.
# Terminate only EngineCore on the assigned GPUs and wait up to 30 seconds.
# Exit 0 when clear; remaining processes, query or termination errors fail.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$script_dir/../tools/ci/gpu_process_guard.py" cleanup "$@"
