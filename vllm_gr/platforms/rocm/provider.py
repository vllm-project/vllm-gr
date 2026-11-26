# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""AMD ROCm provider that reuses the CUDA-placement GR execution paths.

ROCm tensors report ``device.type == "cuda"`` and vLLM's ROCm platform exposes
HIP through the ``torch.cuda`` API, so the CUDA runner, Beam attention
wrapper and constraint kernels are shared. Only platform matching and runtime
version validation differ. Attention resolves through
``vllm.v1.attention.backends.fa_utils``, which selects the ROCm FlashAttention
or AITER implementation on this platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..compat import SUPPORTED_ROCM_VLLM_VERSION_RANGE
from ..cuda.provider import CUDABackendProvider
from ..identity import PlatformIdentity, platform_identity


@dataclass(frozen=True)
class ROCmBackendProvider(CUDABackendProvider):
    backend_id: str = "rocm"
    #: ROCm uses CUDA tensor placement in PyTorch.
    device_type: str = "cuda"
    vllm_version_range: ClassVar[str | None] = SUPPORTED_ROCM_VLLM_VERSION_RANGE

    def matches(self, identity: PlatformIdentity) -> bool:
        """Match vLLM's native ROCm platform."""
        return identity.platform_enum == "ROCM" and identity.device_type == "cuda"

    matches_legacy_identity = matches

    def check_environment(self) -> dict[str, str]:
        from vllm.platforms import current_platform

        from ..compat import check_vllm_runtime_range

        if not self.matches(platform_identity(current_platform)):
            raise RuntimeError("ROCm provider requires the native ROCm platform")

        report = check_vllm_runtime_range(SUPPORTED_ROCM_VLLM_VERSION_RANGE)
        report["platform"] = "rocm"
        return report


ROCM_BACKEND = ROCmBackendProvider()
