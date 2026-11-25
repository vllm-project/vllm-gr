# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Adapt native Ascend configuration validation at process startup."""

import sys
from functools import wraps
from typing import Any


def is_npu_platform() -> bool:
    """Return whether the active vLLM platform is NPU."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def install_ascend_config_filter() -> None:
    """Filter GR keys only during native Ascend configuration validation.

    EngineCore and NPUWorker both initialize the Ascend config. Keep their
    VllmConfig identity (used by the native config cache) and restore its
    original mapping after each synchronous call, including failures. GR
    parsing and serialization then use the same config on every platform.
    """
    import vllm_ascend.ascend_config as ascend_config_mod

    from vllm_gr.config.backend_plan import BACKEND_PLAN_KEY
    from vllm_gr.config.parser import GR_CONFIG_KEY

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
                if key not in (GR_CONFIG_KEY, BACKEND_PLAN_KEY)
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
    install_ascend_config_filter()

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
