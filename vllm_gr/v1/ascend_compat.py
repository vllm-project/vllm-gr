# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility imports for historical Ascend patch entry points."""

from vllm_gr.platforms.ascend.config_patch import (
    apply_engine_core_config_sanitizer,
    is_npu_platform,
)
from vllm_gr.v1.worker.npu.startup_patch import (
    apply_penalties_warmup_oom_guard,
    patch_npu_worker_init_device,
)

__all__ = [
    "apply_ascend_attention_backend_patch",
    "apply_engine_core_config_sanitizer",
    "apply_penalties_warmup_oom_guard",
    "is_npu_platform",
    "patch_npu_worker_init_device",
]


def apply_ascend_attention_backend_patch(force: bool = False) -> int:
    """Compatibility entry point for the hardware-neutral CUSTOM selector.

    ``force`` is forwarded for call compatibility only; its value has no effect
    on the selector installer, which always inspects the live target.
    """
    from vllm_gr.v1.attention.selector_patch import apply_custom_attention_backend_patch

    return apply_custom_attention_backend_patch(force=force)
