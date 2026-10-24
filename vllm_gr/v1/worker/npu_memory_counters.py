# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read Ascend worker memory counters without flattening allocator statistics."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

_MEMORY_COUNTER_PATCH_MARKER = "_vllm_gr_npu_memory_counters"
_NATIVE_WORKER_MODULES = (
    "vllm_ascend.worker.worker",
    "vllm_ascend.worker.worker_v1",
)


def install_npu_memory_counter_patch(worker_cls: type[Any]) -> bool:
    """Replace the native two-counter sampler, leaving unknown variants alone.

    Only the native ``profile_memory(self)`` contract is supported. Custom
    implementations, existing wrappers, and runtimes without the nested public
    API retain their original method. Importing this module never imports NPU
    libraries or initializes a device.
    """
    original = getattr(worker_cls, "profile_memory", None)
    if not callable(original) or getattr(original, _MEMORY_COUNTER_PATCH_MARKER, False):
        return False
    if getattr(original, "__module__", None) not in _NATIVE_WORKER_MODULES:
        return False
    if hasattr(original, "__wrapped__"):
        return False
    try:
        if tuple(inspect.signature(original).parameters) != ("self",):
            return False
    except (TypeError, ValueError):
        return False
    native_logger: Any = getattr(original, "__globals__", {}).get("logger")
    if not callable(getattr(native_logger, "debug", None)):
        return False
    try:
        from torch_npu.npu.memory import memory_stats_as_nested_dict
    except (ImportError, AttributeError):
        return False
    if not callable(memory_stats_as_nested_dict):
        return False

    @wraps(original)
    def profile_memory(self: Any) -> None:
        # Both native scalar APIs flatten and sort a separate full snapshot.
        # The recursive Python flattening closure also creates cyclic garbage.
        # The nested API returns {} before initialization, preserving zero
        # counters without initializing an NPU context.
        stats = memory_stats_as_nested_dict()
        try:
            reserved = stats.get("reserved_bytes", {}).get("all", {}).get("current", 0)
            allocated = stats.get("allocated_bytes", {}).get("all", {}).get("current", 0)
        except (AttributeError, TypeError):
            # A changed statistics schema can still use its native reader.
            original(self)
            return
        if not isinstance(reserved, int) or not isinstance(allocated, int):
            original(self)
            return
        self.torch_reserved = reserved
        self.torch_allocated = allocated
        native_logger.debug(
            "torch reserved memory: %.2f GiB, torch allocated memory: %.2f GiB",
            reserved / (1 << 30),
            allocated / (1 << 30),
        )

    setattr(profile_memory, _MEMORY_COUNTER_PATCH_MARKER, True)
    worker_cls.profile_memory = profile_memory
    return True
