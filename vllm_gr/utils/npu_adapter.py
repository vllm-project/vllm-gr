# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device detection and backend selection helpers."""

from __future__ import annotations

from typing import Optional

import torch

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


class DeviceType:
    """Canonical device backend names used by vllm-gr."""

    CUDA = "cuda"
    NPU = "npu"
    CPU = "cpu"
    UNKNOWN = "unknown"


def get_device_backend(device: torch.device) -> str:
    """Return the canonical backend name for a torch device."""
    device_type = getattr(device, "type", None)
    if device_type is None:
        return DeviceType.UNKNOWN

    device_type = str(device_type).lower()
    if device_type in (DeviceType.CUDA, DeviceType.NPU, DeviceType.CPU):
        return device_type
    return DeviceType.UNKNOWN


def is_npu_available() -> bool:
    """Return whether torch_npu is installed and the runtime is available."""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def is_cuda_available() -> bool:
    """Return whether CUDA is available."""
    try:
        return torch.cuda.is_available()
    except AttributeError:
        return False


def get_default_device() -> torch.device:
    """Prefer NPU, then CUDA, then CPU."""
    if is_npu_available():
        return torch.device(DeviceType.NPU)
    if is_cuda_available():
        return torch.device(DeviceType.CUDA)
    return torch.device(DeviceType.CPU)


class DeviceAdapter:
    """Small facade around backend detection."""

    def __init__(self) -> None:
        self._device_backend = self._detect_device_backend()

        logger.debug(
            "DeviceAdapter backend=%s",
            self._device_backend,
        )

    def _detect_device_backend(self) -> str:
        if is_npu_available():
            return DeviceType.NPU
        if is_cuda_available():
            return DeviceType.CUDA
        return DeviceType.CPU

    @property
    def device_backend(self) -> str:
        return self._device_backend

    @property
    def is_npu(self) -> bool:
        return self._device_backend == DeviceType.NPU

    @property
    def is_cuda(self) -> bool:
        return self._device_backend == DeviceType.CUDA

    @property
    def is_cpu(self) -> bool:
        return self._device_backend == DeviceType.CPU

    def should_use_torchrec(self, device: Optional[torch.device] = None) -> bool:
        """Deprecated; vllm-GR no longer uses torchrec."""
        return False

    def should_use_recsdk(self, device: Optional[torch.device] = None) -> bool:
        """Deprecated alias kept for backward compatibility."""
        return False

    def should_use_fbgemm(self, device: Optional[torch.device] = None) -> bool:
        """Deprecated; vllm-GR no longer uses fbgemm custom ops."""
        return False


_global_adapter: Optional[DeviceAdapter] = None


def get_device_adapter() -> DeviceAdapter:
    """Return a process-wide adapter instance."""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = DeviceAdapter()
    return _global_adapter


def reset_device_adapter() -> None:
    """Reset the process-wide adapter. Primarily used in tests."""
    global _global_adapter
    _global_adapter = None


def get_device_type(device: torch.device) -> str:
    return get_device_backend(device)


def is_npu_device(device: torch.device) -> bool:
    return get_device_backend(device) == DeviceType.NPU


def is_cuda_device(device: torch.device) -> bool:
    return get_device_backend(device) == DeviceType.CUDA


def _cleanup_torch_backend(backend, *, collect_ipc: bool) -> bool:
    """Best-effort cleanup for a torch accelerator backend."""
    if backend is None:
        return False

    is_available = getattr(backend, "is_available", None)
    if callable(is_available):
        try:
            if not is_available():
                return False
        except Exception:
            logger.debug("Backend availability probe failed", exc_info=True)
            return False

    cleaned = False
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        try:
            synchronize()
            cleaned = True
        except Exception:
            logger.debug("Backend synchronize failed during cleanup", exc_info=True)

    empty_cache = getattr(backend, "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
            cleaned = True
        except Exception:
            logger.debug("Backend empty_cache failed during cleanup", exc_info=True)

    if collect_ipc:
        ipc_collect = getattr(backend, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
                cleaned = True
            except Exception:
                logger.debug("Backend ipc_collect failed during cleanup", exc_info=True)

    return cleaned


def release_accelerator_memory() -> None:
    """Release cached CUDA/NPU memory after engine shutdown."""
    _cleanup_torch_backend(getattr(torch, "npu", None), collect_ipc=False)
    _cleanup_torch_backend(getattr(torch, "cuda", None), collect_ipc=True)
