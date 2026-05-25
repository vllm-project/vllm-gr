#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Setup script for vllm-GR.

This file exists for compatibility with older pip versions and editable installs.
The main configuration is in pyproject.toml.

Installation:
    pip install .           # Standard install
    pip install -e .        # Editable install (development)
    pip install -e ".[dev]" # Editable install with dev dependencies

Uninstallation:
    pip uninstall vllm-gr
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
