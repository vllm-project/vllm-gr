# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small model-independent data structures used by beam-search policies."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class BeamSearchModel(StrEnum):
    """Models with dedicated GR beam-search input/output policies."""

    ONE_REC = "one_rec"
    RECIF = "recif"


@dataclass(frozen=True)
class BeamSearchStep:
    """Expansion, retention and token constraints for one search step."""

    expand_width: int
    keep_width: int
    allowed_token_ids: list[int] | None = None


@dataclass
class PreparedBeamInputs:
    """Model-specific input state consumed by the shared search trunk."""

    engine_inputs: list[Any]
    generated_starts: list[int]
    output_starts: list[int]
    prompt_texts: list[str]
    tokenizer: Any | None = None
    begin_token_id: int | None = None
    end_token_id: int | None = None


def select_best_beams(instance: Any, beam_width: int) -> list[Any]:
    """Merge active/completed beams and return the highest-scoring results."""

    instance.completed.extend(instance.beams)
    return sorted(
        instance.completed,
        key=lambda beam: beam.cum_logprob,
        reverse=True,
    )[:beam_width]
