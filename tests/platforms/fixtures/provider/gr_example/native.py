# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native vLLM host platform used by the installed-package acceptance run."""

from gr_example import ExamplePlatform
from vllm.platforms.interface import Platform, PlatformEnum


class HostPlatform(ExamplePlatform, Platform):
    _enum = PlatformEnum.OOT
    device_type = "cpu"
    dispatch_key = "CPU"
    device_control_env_var = "GR_CONFORMANCE_VISIBLE_DEVICES"
