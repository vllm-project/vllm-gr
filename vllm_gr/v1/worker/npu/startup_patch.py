# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Install Ascend Worker startup and warmup hooks."""

import logging
import sys
from typing import Any

from vllm_gr.platforms.ascend.config_patch import (
    install_ascend_config_filter,
    is_npu_platform,
)

logger = logging.getLogger("vllm_gr.v1.ascend_compat")

_NPU_WORKER_ORIGINAL_ATTR = "_vllm_gr_npu_init_device_original"
_PENALTY_WARMUP_GUARD_APPLIED = False


def patch_npu_worker_init_device() -> int:
    """Run vllm-gr patch installation before NPUWorker device initialization."""
    if not is_npu_platform():
        return 0

    try:
        from vllm_ascend.worker.worker import NPUWorker
    except ImportError:
        logger.warning(
            "Failed to import NPUWorker from "
            "vllm_ascend.worker.worker; the vllm-gr delayed NPU patch retry "
            "hook was not installed for this attempt."
        )
        return 0

    # The config validator is called by NPUWorker.__init__, before init_device.
    # Its module bindings can be restored independently of the device wrapper.
    install_ascend_config_filter()
    original_init_device = NPUWorker.init_device
    if _NPU_WORKER_ORIGINAL_ATTR in getattr(original_init_device, "__dict__", {}):
        return 0

    def _patched_init_device(self: Any) -> Any:
        from vllm_gr.patch import re_apply_patches

        re_apply_patches()
        # Allocation must opt in before the native runner builds its KV pool.
        # Unknown Ascend interfaces keep their original allocation policy.
        from vllm_gr.v1.worker.npu.kv_layout_patch import (
            apply_npu_contiguous_kv_cache_patch,
        )

        apply_npu_contiguous_kv_cache_patch()
        # The penalties warmup guard must be installed before
        # compile_or_warm_up_model runs in this child process.
        apply_penalties_warmup_oom_guard()
        # Mount the V1 worker utilities before native initialization.
        from vllm_gr.v1.worker.model_runner_patch import (
            install_worker_utility_hooks,
        )
        from vllm_gr.v1.worker.npu.stage_projection import (
            install_npu_stage_weight_update_hooks,
        )

        install_worker_utility_hooks(type(self))
        install_npu_stage_weight_update_hooks(type(self))
        result = original_init_device(self)
        return result

    setattr(_patched_init_device, _NPU_WORKER_ORIGINAL_ATTR, original_init_device)
    NPUWorker.init_device = _patched_init_device
    return 1


def apply_penalties_warmup_oom_guard() -> int:
    """Skip the Ascend penalties Triton warmup when its dummies cannot fit.

    vllm-ascend >= 0.30 runs Triton kernel warmups right after the KV cache
    has consumed the memory budget, and the penalties stage allocates dummy
    ``max_num_seqs x padded_vocab`` logits and bin-count tensors. Serving
    OneRec's large vocabulary at the benchmark server's ``--max_num_seqs
    2048`` needs ~2.7 GiB more than remains and aborts engine startup. The
    warmup only JIT-compiles kernels for penalty sampling, which requests
    without penalties (including every vllm-gr beam request) never execute,
    so an OOM there skips the warmup instead of failing the engine.
    """
    global _PENALTY_WARMUP_GUARD_APPLIED
    if _PENALTY_WARMUP_GUARD_APPLIED:
        return 0
    try:
        # The warmup package re-exports the kernel_warmup *function*, which
        # shadows the submodule attribute; resolve the real module instead.
        import vllm_ascend.model_executor.warmup.kernel_warmup  # noqa: F401
    except ImportError:
        return 0
    warmup_mod = sys.modules["vllm_ascend.model_executor.warmup.kernel_warmup"]
    original = getattr(warmup_mod, "penalties_triton_warmup", None)
    if original is None or getattr(original, "_vllm_gr_oom_guarded", False):
        _PENALTY_WARMUP_GUARD_APPLIED = True
        return 0

    import torch

    def _guarded_penalties_triton_warmup(worker: Any) -> None:
        try:
            return original(worker)
        except torch.OutOfMemoryError:
            scheduler_config = getattr(worker, "scheduler_config", None)
            model_config = getattr(worker, "model_config", None)
            num_seqs = getattr(scheduler_config, "max_num_seqs", None)
            try:
                vocab_size = model_config.get_vocab_size() if model_config is not None else None
            except Exception:
                vocab_size = None
            logger.warning(
                "Skipped the Ascend penalties Triton warmup: its %s x %s "
                "dummy tensors do not fit beside the KV cache (NPU OOM). "
                "Penalty sampling is not used by vllm-gr beam decoding, so "
                "only the kernel JIT warmup is lost.",
                num_seqs,
                vocab_size,
            )
            torch.npu.empty_cache()

    _guarded_penalties_triton_warmup._vllm_gr_oom_guarded = True  # type: ignore[attr-defined]
    setattr(warmup_mod, "penalties_triton_warmup", _guarded_penalties_triton_warmup)
    _PENALTY_WARMUP_GUARD_APPLIED = True
    return 1
