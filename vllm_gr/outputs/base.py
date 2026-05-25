# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Reserved adapter-level output type hierarchy.

GRModelOutput and its subclasses are **not yet wired** into the runtime
pipeline.  The current real path is:

    adapter.process_output() → GRRequestOutput   (directly)

In a future "output contract unification" stage, adapters will instead
return GRModelOutput subclasses (e.g. ScoringOutput), and the engine
layer will wrap them into GRRequestOutput for API-level consumption.

Until that stage, these types exist as a typed scaffold for adapters
that want to start producing structured output early.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


class GRModelOutput:
    """Base class for adapter-level model outputs (reserved, not yet wired).

    In the current pipeline, adapters return GRRequestOutput directly
    from ``process_output()``.  This class exists as a typed scaffold
    for a future "output contract unification" stage where adapters
    will return GRModelOutput subclasses instead, and the engine layer
    will wrap them into GRRequestOutput.
    """


@dataclass
class ScoringOutput(GRModelOutput):
    """Output from a scoring model — ranked candidate items.

    Produced by adapters that implement ``GRScoringModel`` (e.g. HSTU).
    ``recommendations`` uses the same ``list[RecommendedItem]`` type as
    ``GRRequestOutput.recommendations`` to facilitate future unification.
    """

    recommendations: list = field(default_factory=list)
    all_scores: torch.Tensor | None = None
    request_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingOutput(GRModelOutput):
    """Output from an embedding model — item embeddings and similarities.

    Reserved for future use (Phase 4+).  Produced by adapters that
    implement ``GREmbeddingModel`` (e.g. SASRec embedding mode).
    """

    embeddings: torch.Tensor | None = None
    item_ids: list = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    request_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextOutput(GRModelOutput):
    """Output from a generative model — decoded text.

    Reserved for future use (Phase 5+).  Produced by adapters that
    support generative inference (e.g. P5).
    """

    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    request_meta: dict[str, Any] = field(default_factory=dict)
