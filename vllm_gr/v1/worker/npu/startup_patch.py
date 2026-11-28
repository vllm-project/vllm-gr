# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Install Ascend Worker startup and warmup hooks."""

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_NPU_WORKER_ORIGINAL_ATTR = "_vllm_gr_npu_init_device_original"
_PENALTY_WARMUP_GUARD_APPLIED = False


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
