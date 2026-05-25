# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Capability protocols for GR model adapters.

These ``@runtime_checkable`` protocols allow the engine and worker layers
to query adapter capabilities via ``isinstance()`` checks, enabling
conditional behavior (e.g. prefix-cache, scoring, embedding) without
coupling to concrete adapter classes.

``GRScoringModel`` and ``GREmbeddingModel`` share the same method
signatures but are distinguished by different marker attributes
(``is_scoring_model`` vs ``is_embedding_model``).  Because
``@runtime_checkable`` protocols only check attribute presence (not
value), distinct attribute names are required for correct isinstance
dispatch.
"""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from vllm_gr.inputs.base import GRParsedInput


@runtime_checkable
class GRPoolingModel(Protocol):
    """Marker protocol: adapter supports the pooling (scoring) path."""

    def process_output(self, raw_output: Any, request_meta: dict) -> Any: ...


@runtime_checkable
class GRPrefixCacheSupport(Protocol):
    """Marker protocol: adapter supports prefix-cache salting."""

    def build_prompt_token_ids(self, parsed: "GRParsedInput") -> list[int] | None: ...

    def build_cache_salt(self, parsed: "GRParsedInput") -> str | None: ...


@runtime_checkable
class GRScoringModel(Protocol):
    """Marker protocol: adapter provides candidate scoring.

    Distinguished from ``GREmbeddingModel`` by the presence of the
    ``is_scoring_model`` marker. ``@runtime_checkable`` protocols only
    verify attribute presence, not attribute values, so the scoring and
    embedding markers must use different attribute names.
    """

    is_scoring_model: bool

    def build_model_kwargs(self, batch_features: list[tuple[str, dict[str, Any]]], device: Any) -> dict[str, Any]: ...


@runtime_checkable
class GREmbeddingModel(Protocol):
    """Marker protocol: adapter provides embedding outputs.

    Distinguished from ``GRScoringModel`` by the presence of the
    ``is_embedding_model`` marker.

    Reserved for future use (Phase 5).
    """

    is_embedding_model: bool

    def build_model_kwargs(self, batch_features: list[tuple[str, dict[str, Any]]], device: Any) -> dict[str, Any]: ...
