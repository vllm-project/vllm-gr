# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Data-only beam-search domain model.

The types in this module deliberately contain no vLLM, serving, scheduler, or
hardware objects.  Entry points can translate their transport-specific data to
these immutable values before invoking the beam-search state machine.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite


class BeamFinishedReason(str, Enum):
    """Reason why a beam can no longer be expanded."""

    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class BeamState:
    """A single active or completed beam sequence."""

    token_ids: tuple[int, ...]
    cumulative_logprob: float = 0.0
    finish_reason: BeamFinishedReason | None = None
    stop_reason: int | str | None = None

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None


@dataclass(frozen=True, slots=True)
class BeamCandidateSegment:
    """Candidates emitted for one parent beam during an engine step."""

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    ranks: tuple[int | None, ...] | None = None
    decoded_tokens: tuple[str | None, ...] | None = None

    def __post_init__(self) -> None:
        size = len(self.token_ids)
        if len(self.logprobs) != size:
            raise ValueError("token_ids and logprobs must have the same length")
        if self.ranks is not None and len(self.ranks) != size:
            raise ValueError("ranks must have the same length as token_ids")
        if self.decoded_tokens is not None and len(self.decoded_tokens) != size:
            raise ValueError("decoded_tokens must have the same length as token_ids")


@dataclass(frozen=True, slots=True)
class BeamCandidate:
    """One possible token extension from a parent beam."""

    parent_index: int
    token_id: int
    token_logprob: float
    cumulative_logprob: float
    rank: int | None = None
    decoded_token: str | None = None


@dataclass(frozen=True, slots=True)
class BeamGroup:
    """Active and completed beams for one logical request."""

    active: tuple[BeamState, ...]
    completed: tuple[BeamState, ...] = ()

    def __post_init__(self) -> None:
        if any(beam.is_finished for beam in self.active):
            raise ValueError("active beams cannot have a finish reason")
        if any(not beam.is_finished for beam in self.completed):
            raise ValueError("completed beams must have a finish reason")


@dataclass(frozen=True, slots=True)
class BeamRequestContext:
    """Algorithm options shared by all steps in one beam request."""

    beam_width: int
    eos_token_id: int
    prompt_length: int = 0
    ignore_eos: bool = False
    include_stop_token: bool = True
    length_penalty: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.beam_width, bool) or not isinstance(self.beam_width, int):
            raise TypeError("beam_width must be an integer")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be greater than zero")
        if isinstance(self.eos_token_id, bool) or not isinstance(self.eos_token_id, int):
            raise TypeError("eos_token_id must be an integer")
        if isinstance(self.prompt_length, bool) or not isinstance(self.prompt_length, int):
            raise TypeError("prompt_length must be an integer")
        if self.prompt_length < 0:
            raise ValueError("prompt_length cannot be negative")
        if not isfinite(self.length_penalty):
            raise ValueError("length_penalty must be finite")


@dataclass(frozen=True, slots=True)
class BeamStepInput:
    """Complete pure input for one beam-search transition."""

    group: BeamGroup
    candidate_segments: tuple[BeamCandidateSegment, ...]
    context: BeamRequestContext
    allowed_token_ids: tuple[frozenset[int] | None, ...] | None = None

    def __post_init__(self) -> None:
        active_count = len(self.group.active)
        if len(self.candidate_segments) != active_count:
            raise ValueError("each active beam must have one candidate segment")
        if self.allowed_token_ids is not None and len(self.allowed_token_ids) != active_count:
            raise ValueError("each active beam must have one allowed-token entry")


@dataclass(frozen=True, slots=True)
class BeamParentChild:
    """Mapping used by execution adapters to fork parent beam state."""

    parent_index: int
    child_index: int
    token_id: int


@dataclass(frozen=True, slots=True)
class BeamStepResult:
    """Output of one beam-search transition."""

    group: BeamGroup
    selected_candidates: tuple[BeamCandidate, ...] = ()
    finished_candidates: tuple[BeamCandidate, ...] = ()
    parent_child_map: tuple[BeamParentChild, ...] = ()
    finished_this_step: tuple[BeamState, ...] = field(default_factory=tuple)
