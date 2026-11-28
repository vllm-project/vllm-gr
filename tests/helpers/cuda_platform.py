# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Load CUDA platform classes for host policy tests on non-CUDA vLLM builds."""

import sys
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

# Upstream CUDA imports this extension to register kernels at module import.
# Host tests exercise its Python class hierarchy without launching those ops.
_extension = "vllm._C_stable_libtorch"
_host_only = find_spec(_extension) is None
if _host_only:
    sys.modules[_extension] = ModuleType(_extension)
try:
    CudaGRPlatform = import_module("vllm_gr.platforms.cuda.platform").CudaGRPlatform
finally:
    if _host_only:
        del sys.modules[_extension]
