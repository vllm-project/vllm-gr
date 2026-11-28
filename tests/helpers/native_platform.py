# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native platform objects for host-only GR integration tests."""

from vllm.platforms.interface import Platform, PlatformEnum


class VendorNative(Platform):  # type: ignore[misc]
    _enum = PlatformEnum.OOT
    device_type = "vendor"


def native_platform(enum: str, device: str) -> Platform:
    if enum == "CUDA" and device == "cuda":
        from tests.helpers.cuda_platform import CudaGRPlatform

        return CudaGRPlatform.native_platform_type()
    if enum == "OOT" and device == "npu":
        from tests.helpers.ascend_platform import AscendGRPlatform

        return AscendGRPlatform.native_platform_type()
    native = VendorNative()
    native._enum = PlatformEnum[enum]
    native.device_type = device
    return native
