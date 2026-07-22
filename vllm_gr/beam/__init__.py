# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure beam-search domain types and operations."""

from vllm_gr.beam.domain import (
    BeamCandidate,
    BeamCandidateSegment,
    BeamFinishedReason,
    BeamGroup,
    BeamParentChild,
    BeamRequestContext,
    BeamState,
    BeamStepInput,
    BeamStepResult,
)
from vllm_gr.beam.pruning import prune_candidates
from vllm_gr.beam.ranking import (
    length_penalized_score,
    rank_candidates,
    rank_states,
)
from vllm_gr.beam.state_machine import (
    deduplicate_segment,
    expand_candidates,
    finalize_beam_group,
    run_beam_step,
)

__all__ = [
    "BeamCandidate",
    "BeamCandidateSegment",
    "BeamFinishedReason",
    "BeamGroup",
    "BeamParentChild",
    "BeamRequestContext",
    "BeamState",
    "BeamStepInput",
    "BeamStepResult",
    "deduplicate_segment",
    "expand_candidates",
    "finalize_beam_group",
    "length_penalized_score",
    "prune_candidates",
    "rank_candidates",
    "rank_states",
    "run_beam_step",
]
