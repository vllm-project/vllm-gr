# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""
vLLM-gr module.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

__all__ = [
    "BeamAttentionBackend",
]


def __getattr__(name: str) -> Any:
    if name == "BeamAttentionBackend":
        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

        globals()[name] = BeamAttentionBackend
        return BeamAttentionBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
