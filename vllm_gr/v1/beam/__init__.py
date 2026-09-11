# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lazy exports for production Beam runtime primitives."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_gr.v1.beam.beam_attention_context import (
        BeamAttentionContext as BeamAttentionContext,
    )
    from vllm_gr.v1.beam.beam_attention_pool import (
        BeamAttentionPool as BeamAttentionPool,
    )
    from vllm_gr.v1.beam.beam_search_session import (
        BeamSearchSession as BeamSearchSession,
    )
    from vllm_gr.v1.beam.beam_session_registry import (
        BeamSessionRegistry as BeamSessionRegistry,
    )
    from vllm_gr.v1.beam.persistent_contracts import (
        GRBeamSessionSpec as GRBeamSessionSpec,
        GRStageAck as GRStageAck,
        GRStageKind as GRStageKind,
        GRWorkerStageSpec as GRWorkerStageSpec,
    )

_LAZY_EXPORTS = {
    "BeamAttentionContext": "vllm_gr.v1.beam.beam_attention_context",
    "BeamAttentionPool": "vllm_gr.v1.beam.beam_attention_pool",
    "BeamSearchSession": "vllm_gr.v1.beam.beam_search_session",
    "BeamSessionRegistry": "vllm_gr.v1.beam.beam_session_registry",
    "GRBeamSessionSpec": "vllm_gr.v1.beam.persistent_contracts",
    "GRStageAck": "vllm_gr.v1.beam.persistent_contracts",
    "GRStageKind": "vllm_gr.v1.beam.persistent_contracts",
    "GRWorkerStageSpec": "vllm_gr.v1.beam.persistent_contracts",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
