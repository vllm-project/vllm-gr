# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM-gr module.
"""

from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend


def register_beam_attention_backend() -> int:
    """Ensure CUSTOM points to the GR backend in the current process."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    expected_path = f"{BeamAttentionBackend.__module__}.{BeamAttentionBackend.__qualname__}"
    try:
        if AttentionBackendEnum.CUSTOM.get_path() == expected_path:
            return 0
    except ValueError:
        pass
    register_backend(AttentionBackendEnum.CUSTOM, expected_path)
    return 1


__all__ = [
    "BeamAttentionBackend",
    "register_beam_attention_backend",
]
