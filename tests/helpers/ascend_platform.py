# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Load the real GR Ascend implementation against a vendor base double on host-only CI."""

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

from vllm.platforms.interface import Platform, PlatformEnum

if find_spec("vllm_ascend") is None:
    import sys

    vendor = ModuleType("vllm_ascend.platform")

    class NPUPlatform(Platform):  # type: ignore[misc]
        _enum = PlatformEnum.OOT
        device_type = "npu"

    NPUPlatform.__module__ = vendor.__name__
    vars(vendor)["NPUPlatform"] = NPUPlatform
    # Keep the GR module cached exactly as it would be after a regular import.
    sys.modules[vendor.__name__] = vendor
    try:
        AscendGRPlatform = import_module("vllm_gr.platforms.ascend.platform").AscendGRPlatform
    finally:
        del sys.modules[vendor.__name__]
else:
    AscendGRPlatform = import_module("vllm_gr.platforms.ascend.platform").AscendGRPlatform
