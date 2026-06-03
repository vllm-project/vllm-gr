# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base input types for generative recommendation.

GRPromptBase defines the structural contract for all GR model prompts.
GRParsedInput carries the adapter-specific parsed representation.

Both types are model-family-agnostic.  Model-specific fields (e.g. HSTU's
user_history, SASRec's past_ids) belong in model-specific prompt subtypes
or in the ``metadata`` dict of GRParsedInput.
"""

from dataclasses import dataclass, field
from typing import Any

from typing_extensions import NotRequired, TypedDict


class GRPromptBase(TypedDict):
    """Structural base for all GR model prompts.

    Every model family's prompt type must include at least ``user_id``.
    The optional ``model_type`` field can be used by adapters to dispatch
    without inspecting feature names.

    IMPORTANT: GRPrompt (in inputs/data.py) is NOT a subclass of this
    type.  GRPromptBase serves as the *contract* for new model families;
    the original GRPrompt remains the user-facing API type and must not
    be changed due to Pydantic schema stability requirements.
    """

    user_id: str | int
    model_type: NotRequired[str]


@dataclass
class GRParsedInput:
    """Adapter-agnostic intermediate representation of a parsed prompt.

    After ``adapter.parse_prompt()`` extracts features from a prompt, the
    result is stored in a GRParsedInput that carries:

    - ``user_id`` and ``schema_fingerprint`` — cross-model通用字段
    - ``raw_features`` — model-specific raw feature data, only the adapter
      knows how to interpret
    - ``metadata`` — model-specific metadata (e.g. HSTU's ``history_len``,
      ``num_candidates``).  Pipeline components do NOT interpret this field
    - ``original_prompt`` — preserves the original request for output mapping

    Adapters may extend this via ``metadata`` rather than subclassing.
    """

    user_id: str = ""
    schema_fingerprint: str = ""
    raw_features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    original_prompt: dict[str, Any] = field(default_factory=dict)
