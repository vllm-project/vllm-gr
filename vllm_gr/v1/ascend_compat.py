# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Compatibility patches for vLLM-Ascend integration points."""

import logging
from functools import wraps
from typing import Any

import torch

logger = logging.getLogger(__name__)

_NPU_WORKER_ORIGINAL_ATTR = "_vllm_gr_npu_init_device_original"
_PENALTY_WARMUP_GUARD_APPLIED = False


def is_npu_platform() -> bool:
    """Return whether the active vLLM platform is NPU."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def apply_ascend_attention_backend_patch(force: bool = False) -> int:
    """Compatibility entry point for the hardware-neutral CUSTOM selector.

    ``force`` is forwarded for call compatibility only; its value has no effect
    on the selector installer, which always inspects the live target.
    """
    from vllm_gr.v1.attention.selector_patch import apply_custom_attention_backend_patch

    return apply_custom_attention_backend_patch(force=force)


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
    _install_init_ascend_config_sanitizer()
    original_init_device = NPUWorker.init_device
    if _NPU_WORKER_ORIGINAL_ATTR in getattr(original_init_device, "__dict__", {}):
        return 0

    def _patched_init_device(self: Any) -> Any:
        from vllm_gr.patch import re_apply_patches

        re_apply_patches()
        # Allocation must opt in before the native runner builds its KV pool.
        # Unknown Ascend interfaces keep their original allocation policy.
        from vllm_gr.v1.npu_kv_layout import apply_npu_contiguous_kv_cache_patch

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
    import sys

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


def _install_init_ascend_config_sanitizer() -> None:
    """Filter GR keys only during native Ascend configuration validation.

    EngineCore and NPUWorker both initialize the Ascend config. Keep their
    VllmConfig identity (used by the native config cache) and restore its
    original mapping after each synchronous call, including failures. GR
    parsing and serialization then use the same config on every platform.
    """
    import sys

    import vllm_ascend.ascend_config as ascend_config_mod

    from vllm_gr.config.parser import BACKEND_TRANSPORT_KEYS, GR_CONFIG_KEY

    current = ascend_config_mod.init_ascend_config
    if getattr(current, "_vllm_gr_sanitized", False) is True:
        sanitized = current
        original = current.__wrapped__
    else:
        original = current

        @wraps(original)
        def _patched_init_ascend_config(vllm_config: Any) -> Any:
            additional = getattr(vllm_config, "additional_config", None)
            if not isinstance(additional, dict):
                return original(vllm_config)
            filtered = {
                key: value
                for key, value in additional.items()
                if key != GR_CONFIG_KEY and key not in BACKEND_TRANSPORT_KEYS
            }
            if len(filtered) == len(additional):
                return original(vllm_config)
            vllm_config.additional_config = filtered
            try:
                return original(vllm_config)
            finally:
                vllm_config.additional_config = additional

        _patched_init_ascend_config._vllm_gr_sanitized = True  # type: ignore[attr-defined]
        sanitized = _patched_init_ascend_config
        ascend_config_mod.init_ascend_config = sanitized

    # Modules imported later bind the patched function via their
    # `from vllm_ascend.ascend_config import init_ascend_config`; only update
    # module-global bindings that point to the original, even if a previous
    # installation already wrapped the definition in ascend_config.
    for module_name in (
        "vllm_ascend.patch.platform.patch_engine_core",
        "vllm_ascend.worker.worker",
    ):
        mod = sys.modules.get(module_name)
        if mod is not None and getattr(mod, "init_ascend_config", None) is original:
            setattr(mod, "init_ascend_config", sanitized)


def _sanitize_engine_core_entry(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    """Install the config filter before the child's first Ascend init call.

    Installed on ``EngineCoreProc.run_engine_core`` after vllm-ascend's own
    wrapper, so this function is what multiprocessing pickles as the child's
    target. Delegate to the native Ascend entrypoint with the original
    config; only its validation call sees the filtered mapping.
    """
    _install_init_ascend_config_sanitizer()

    from vllm_ascend.patch.platform.patch_engine_core import (
        _run_engine_core_patch_func as ascend_entry,
    )

    # This NPU entry replaces the generic GR process wrapper. Repair the
    # child's Core capabilities before native initialization begins; applying
    # them from NPUWorker.init_device is too late for this Core's __init__.
    from vllm_gr.patch import initialize_engine_core_beam_patches

    initialize_engine_core_beam_patches()
    return ascend_entry(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


def apply_engine_core_config_sanitizer(engine_core_proc_cls: Any) -> int:
    """Wrap the EngineCore entrypoint to strip GR keys before Ascend init.

    Re-asserted on every create_engine_config: each EngineArgs round-trip
    re-runs load_general_plugins, which reinstalls vllm-gr's generic
    run_engine_core wrapper (and vllm-ascend's) over this sanitizer. Checking
    identity rather than a one-shot flag keeps the sanitizer outermost across
    multiple engines in one process.
    """
    if not is_npu_platform():
        return 0
    if engine_core_proc_cls.run_engine_core is _sanitize_engine_core_entry:
        return 0
    engine_core_proc_cls.run_engine_core = staticmethod(_sanitize_engine_core_entry)
    return 1
