# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small model-independent data structures used by beam-search policies."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BeamSearchStep:
    """Expansion, retention and token constraints for one search step."""

    expand_width: int
    keep_width: int
    allowed_token_ids: list[int] | None = None


BeamSearchStepBuilder = Callable[[int, int], list[BeamSearchStep]]
_MODEL_STEP_BUILDERS: dict[str, BeamSearchStepBuilder] = {}


def register_beam_search_step_builder(
    architecture: str,
    builder: BeamSearchStepBuilder,
) -> None:
    """Register model-owned hierarchical search policy at process startup."""

    _MODEL_STEP_BUILDERS[architecture] = builder


def build_beam_search_steps(
    architectures: Sequence[str],
    max_tokens: int,
    beam_width: int,
) -> list[BeamSearchStep]:
    """Resolve a model policy, falling back to fixed-width beam search."""

    for architecture in architectures:
        builder = _MODEL_STEP_BUILDERS.get(architecture)
        if builder is not None:
            return builder(max_tokens, beam_width)
    return [
        BeamSearchStep(expand_width=beam_width, keep_width=beam_width) for _ in range(max_tokens)
    ]


def select_best_beams(instance: Any, beam_width: int) -> list[Any]:
    """Merge active/completed beams and return the highest-scoring results."""

    instance.completed.extend(instance.beams)
    return sorted(
        instance.completed,
        key=lambda beam: beam.cum_logprob,
        reverse=True,
    )[:beam_width]
