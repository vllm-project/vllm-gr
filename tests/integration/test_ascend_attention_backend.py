# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Integration coverage for native attention selection on Ascend NPU."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

pytestmark = pytest.mark.npu_test


def _require_npu_runtime() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"torch-npu is not installed: {exc}")

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")

    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        pytest.skip(
            f"vLLM did not select the NPU platform: device_type={current_platform.device_type!r}"
        )


def test_native_backend_loads_after_normal_ascend_op_registration() -> None:
    """Worker-style Ascend initialization can load its native attention backend."""
    _require_npu_runtime()

    import vllm.v1.attention.selector as selector_mod
    from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
    from vllm_ascend.utils import register_ascend_customop

    from vllm_gr.platforms.ascend.environment import check_ascend_environment
    from vllm_gr.platforms.config import engine_config_service
    from vllm_gr.v1.attention import selector_patch
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    check_ascend_environment(require_npu_platform=True)

    config = VllmConfig(attention_config=AttentionConfig())
    config.attention_config.backend = None
    engine_config_service().apply_native_defaults(config)

    # AscendWorker.init_device() registers Ascend ops before model/backend use.
    # This test has no real model_config, so use the registration entry point's
    # supported no-config path. vLLM's bundled triton_kernels target standard
    # GPU Triton and are incompatible with Triton-Ascend; they are unrelated to
    # the native Ascend kernels covered here, so report them as unavailable
    # while the registration modules are imported.
    #
    # register_ascend_customop() writes process-global state (vLLM's
    # op_registry_oot and vllm-ascend's _ASCEND_CUSTOMOP_IS_REIGISTERED) that
    # nothing later in the process undoes. Leaving it installed makes every
    # subsequent LogitsProcessor construction in the same pytest process
    # resolve to AscendLogitsProcessor, whose _get_logits reads
    # lm_head.parallel_mode — an attribute the CPU fakes in other test modules
    # do not model. Restore both so the registration stays local to this test.
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm_ascend import utils as ascend_utils

    saved_registry = dict(op_registry_oot)
    saved_flag = ascend_utils._ASCEND_CUSTOMOP_IS_REIGISTERED
    try:
        with mock.patch(
            "vllm.utils.import_utils.has_triton_kernels",
            return_value=False,
        ):
            register_ascend_customop()
        selector_patch.apply_custom_attention_backend_patch(force=True)
        selector_patch.clear_attention_backend_cache()

        selector_config = AttentionSelectorConfig(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype=None,
            block_size=config.cache_config.block_size,
        )
        with set_current_vllm_config(config):
            # Engine startup runs NPUPlatform.check_and_update_config, whose
            # step 5 initializes the Ascend config before any backend
            # selection; rc2's get_attn_backend_cls reads rl_config from it
            # during FA3 validation before falling back to the registered
            # plugin backend. Mirror that ordering, then restore the unset
            # state for the rest of the process.
            init_ascend_config(config)
            try:
                backend = selector_mod._cached_get_attn_backend(None, selector_config)
            finally:
                clear_ascend_config()
    finally:
        op_registry_oot.clear()
        op_registry_oot.update(saved_registry)
        ascend_utils._ASCEND_CUSTOMOP_IS_REIGISTERED = saved_flag

    assert backend is not BeamAttentionBackend
    assert backend.__module__.startswith("vllm_ascend.")
    assert config.cache_config.block_size in backend.get_supported_kernel_block_sizes()
