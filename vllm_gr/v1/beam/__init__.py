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
    from vllm_gr.v1.beam.execution_batch import (
        BeamBatchExecutionView as BeamBatchExecutionView,
    )
    from vllm_gr.v1.beam.execution_batch import (
        BeamExecutionContractError as BeamExecutionContractError,
    )
    from vllm_gr.v1.beam.execution_batch import BeamExecutionStage as BeamExecutionStage

_LAZY_EXPORTS = {
    "BeamAttentionContext": "vllm_gr.v1.beam.beam_attention_context",
    "BeamAttentionPool": "vllm_gr.v1.beam.beam_attention_pool",
    "BeamSearchSession": "vllm_gr.v1.beam.beam_search_session",
    "BeamSessionRegistry": "vllm_gr.v1.beam.beam_session_registry",
    "BeamBatchExecutionView": "vllm_gr.v1.beam.execution_batch",
    "BeamExecutionContractError": "vllm_gr.v1.beam.execution_batch",
    "BeamExecutionStage": "vllm_gr.v1.beam.execution_batch",
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
