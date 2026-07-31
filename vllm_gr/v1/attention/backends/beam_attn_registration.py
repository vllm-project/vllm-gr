# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local registration for the GR Beam Attention backend."""

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

BEAM_ATTENTION_BACKEND_PATH = "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"


def register_beam_attention_backend() -> None:
    """Register the Beam backend in this process's CUSTOM backend slot."""
    register_backend(AttentionBackendEnum.CUSTOM, BEAM_ATTENTION_BACKEND_PATH)
