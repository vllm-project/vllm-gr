# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Platform-independent Beam runtime primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
    from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool
    from vllm_gr.v1.beam.beam_search_controller import BeamSearchController
    from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
    from vllm_gr.v1.beam.beam_session_registry import BeamSessionRegistry

__all__ = [
    "BeamAttentionContext",
    "BeamAttentionPool",
    "BeamSearchController",
    "BeamSearchSession",
    "BeamSessionRegistry",
]

_EXPORT_MODULES = {
    "BeamAttentionContext": "vllm_gr.v1.beam.beam_attention_context",
    "BeamAttentionPool": "vllm_gr.v1.beam.beam_attention_pool",
    "BeamSearchController": "vllm_gr.v1.beam.beam_search_controller",
    "BeamSearchSession": "vllm_gr.v1.beam.beam_search_session",
    "BeamSessionRegistry": "vllm_gr.v1.beam.beam_session_registry",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
