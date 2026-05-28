#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

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

import os
from setuptools import setup

# Try to build the C++ minheap extension.
# If pybind11 or numpy is not available, skip silently (e.g. during metadata generation).
ext_modules = []
try:
    import pybind11
    import numpy as np
    from setuptools import Extension

    ext_modules = [
        Extension(
            "minheap_cpp",
            ["vllm_gr/csrc/minheap.cpp"],
            include_dirs=[pybind11.get_include(), np.get_include()],
            extra_compile_args=["-std=c++17", "-O3"],
        ),
    ]
except ImportError:
    pass

if __name__ == "__main__":
    setup(ext_modules=ext_modules)
