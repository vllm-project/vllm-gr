#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run offline OneRec Beam Search with the prompt at its natural length."""

from offline_beam_search import main

if __name__ == "__main__":
    main(quickstart=True)
