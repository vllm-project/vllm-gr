#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Compatibility entrypoint; scenarios and assertions now live in pytest.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec python3 -m pytest --strict-test-levels -v -rs tests/development/test_resource_probe.py "$@"
