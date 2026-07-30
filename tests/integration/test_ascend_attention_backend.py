# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration coverage for native attention selection on Ascend NPU."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

pytestmark = pytest.mark.npu_integration


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
    from vllm_ascend.utils import register_ascend_customop

    import vllm_gr.patch as gr_patch
    from vllm_gr._env_check import check_ascend_environment
    from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size
    from vllm_gr.v1 import ascend_compat
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    check_ascend_environment(require_npu_platform=True)

    config = VllmConfig(attention_config=AttentionConfig())
    config.attention_config.backend = None
    _apply_npu_default_block_size(config)

    # AscendWorker.init_device() registers Ascend ops before model/backend use.
    # This test has no real model_config, so use the registration entry point's
    # supported no-config path. vLLM's bundled triton_kernels target standard
    # GPU Triton and are incompatible with Triton-Ascend; they are unrelated to
    # the native Ascend kernels covered here, so report them as unavailable
    # while the registration modules are imported.
    with mock.patch(
        "vllm.utils.import_utils.has_triton_kernels",
        return_value=False,
    ):
        register_ascend_customop()
    gr_patch._apply_ascend_attention_backend_patch(force=True)
    ascend_compat._ASCEND_ORIGINAL_CACHED_GET_ATTN.cache_clear()

    selector_config = AttentionSelectorConfig(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype=None,
        block_size=config.cache_config.block_size,
    )
    with set_current_vllm_config(config):
        backend = selector_mod._cached_get_attn_backend(None, selector_config)

    assert backend is not BeamAttentionBackend
    assert backend.__module__.startswith("vllm_ascend.")
    assert config.cache_config.block_size in backend.get_supported_kernel_block_sizes()
