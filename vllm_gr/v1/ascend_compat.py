# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility patches for vLLM-Ascend integration points."""

import logging
from functools import wraps
from typing import Any, cast

import torch

from vllm_gr.config import GRAttentionBackend, GRConfig

logger = logging.getLogger(__name__)

_ASCEND_ATTN_BACKEND_PATCH_APPLIED = False
_ASCEND_ORIGINAL_CACHED_GET_ATTN: Any = None
_NPU_WORKER_PATCH_APPLIED = False
_PENALTY_WARMUP_GUARD_APPLIED = False


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    """Read the validated GR backend marker used by the attention selector."""
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def is_npu_platform() -> bool:
    """Return whether the active vLLM platform is NPU."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def _get_custom_beam_backend() -> type:
    """Load CUSTOM on demand; the engine core resolves its supported layouts."""
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    return cast(type[Any], BeamAttentionBackend)


def apply_ascend_attention_backend_patch(force: bool = False) -> int:
    """Preserve explicit and marker-restored CUSTOM backend selection."""
    global _ASCEND_ATTN_BACKEND_PATCH_APPLIED
    global _ASCEND_ORIGINAL_CACHED_GET_ATTN

    if _ASCEND_ATTN_BACKEND_PATCH_APPLIED and not force:
        return 0

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return 0

    try:
        import vllm.v1.attention.selector as selector_mod
    except ImportError:
        return 0

    if _ASCEND_ORIGINAL_CACHED_GET_ATTN is None:
        _ASCEND_ORIGINAL_CACHED_GET_ATTN = selector_mod._cached_get_attn_backend
    original_cached_fn = _ASCEND_ORIGINAL_CACHED_GET_ATTN

    def _patched_cached_get_attn_backend(
        backend: Any,
        attn_selector_config: Any,
        num_heads: int | None = None,
    ) -> Any:
        # Do not import an unselected backend during plugin initialization.
        # If CUSTOM is selected, propagate its original import/registration
        # error instead of swallowing it and reporting "not registered" later.
        if backend == AttentionBackendEnum.CUSTOM:
            return _get_custom_beam_backend()

        if backend is None and _get_current_gr_attention_backend() is GRAttentionBackend.CUSTOM:
            return _get_custom_beam_backend()

        return original_cached_fn(
            backend,
            attn_selector_config,
            num_heads=num_heads,
        )

    selector_mod._cached_get_attn_backend = _patched_cached_get_attn_backend
    _ASCEND_ATTN_BACKEND_PATCH_APPLIED = True
    return 1


def patch_npu_worker_init_device() -> int:
    """Run vllm-gr patch installation before NPUWorker device initialization."""
    global _NPU_WORKER_PATCH_APPLIED

    if not is_npu_platform() or _NPU_WORKER_PATCH_APPLIED:
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

    # Spawned workers load plugins before constructing NPUWorker, whose
    # __init__ validates Ascend config before init_device can run. Install
    # the filter here too; a fresh worker cannot inherit EngineCore's patch.
    _install_init_ascend_config_sanitizer()
    original_init_device = NPUWorker.init_device

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
        # Mount set_gr_constraint_resources so the driver can inject constraint
        # tables via EngineCore.collective_rpc after init_device returns.
        from vllm_gr.constraints.worker_inject import load_constraint_tables_on_worker
        from vllm_gr.ops.npu_stage_logits import install_npu_stage_weight_update_hooks
        from vllm_gr.v1.worker.model_runner_patch import (
            install_worker_utility_hooks,
        )

        install_worker_utility_hooks(type(self))
        install_npu_stage_weight_update_hooks(type(self))
        result = original_init_device(self)
        # Preload constraint tables onto the device at weight-load time so the
        # runtime never does a lazy build/H2D on the first constrained step.
        load_constraint_tables_on_worker(self)
        return result

    NPUWorker.init_device = _patched_init_device
    _NPU_WORKER_PATCH_APPLIED = True
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
                vocab_size = model_config.get_vocab_size()
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
    warmup_mod.penalties_triton_warmup = _guarded_penalties_triton_warmup
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

    from vllm_gr.config.parser import GR_CONFIG_KEY, GR_CONFIG_KEY_PREFIX

    original = ascend_config_mod.init_ascend_config
    if getattr(original, "_vllm_gr_sanitized", False):
        return

    @wraps(original)
    def _patched_init_ascend_config(vllm_config: Any) -> Any:
        additional = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional, dict):
            return original(vllm_config)
        filtered = {
            key: value
            for key, value in additional.items()
            if key != GR_CONFIG_KEY
            and not (isinstance(key, str) and key.startswith(GR_CONFIG_KEY_PREFIX))
        }
        if len(filtered) == len(additional):
            return original(vllm_config)
        vllm_config.additional_config = filtered
        try:
            return original(vllm_config)
        finally:
            vllm_config.additional_config = additional

    _patched_init_ascend_config._vllm_gr_sanitized = True  # type: ignore[attr-defined]
    ascend_config_mod.init_ascend_config = _patched_init_ascend_config

    # Modules imported later bind the patched function via their
    # `from vllm_ascend.ascend_config import init_ascend_config`; only update
    # module-global bindings that were already created with the original.
    for module_name in (
        "vllm_ascend.patch.platform.patch_engine_core",
        "vllm_ascend.worker.worker",
    ):
        mod = sys.modules.get(module_name)
        if mod is not None and getattr(mod, "init_ascend_config", None) is original:
            mod.init_ascend_config = _patched_init_ascend_config


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

    return ascend_entry(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


def apply_engine_core_config_sanitizer(engine_core_proc_cls: type) -> int:
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
