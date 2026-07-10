# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility shim for vLLM beam-search types.

vLLM 0.22+ moved beam-search utilities to
``vllm.entrypoints.generate.beam_search.utils`` and added
``BeamSearchSequence.orig_prompt`` / ``get_prompt()``.  This module re-exports
the legacy ``vllm.beam_search`` types with the new API surface so that
vllm-gr can fall back gracefully when the 0.22+ module layout is unavailable.
"""

from dataclasses import dataclass, field
from typing import Any

from vllm.beam_search import BeamSearchInstance as _BeamSearchInstance
from vllm.beam_search import BeamSearchOutput as _BeamSearchOutput
from vllm.beam_search import BeamSearchSequence as _BeamSearchSequence
from vllm.inputs import TokensPrompt

BeamSearchInstance = _BeamSearchInstance
BeamSearchOutput = _BeamSearchOutput


@dataclass
class BeamSearchSequence(_BeamSearchSequence):
    """BeamSearchSequence with the vLLM 0.22+ prompt interface."""

    orig_prompt: Any = field(default=None, compare=False, repr=False)

    def get_prompt(self) -> Any:
        """Return the prompt payload used to resume this beam.

        Prefer the stored ``orig_prompt`` (0.22+ style).  Otherwise synthesize a
        ``TokensPrompt`` from the beam's token sequence and multimodal data.
        """
        if self.orig_prompt is not None:
            return self.orig_prompt

        token_prompt_kwargs: dict = {"prompt_token_ids": self.tokens}
        if self.multi_modal_data is not None:
            token_prompt_kwargs["multi_modal_data"] = self.multi_modal_data
        if self.mm_processor_kwargs is not None:
            token_prompt_kwargs["mm_processor_kwargs"] = self.mm_processor_kwargs
        return TokensPrompt(**token_prompt_kwargs)


def create_sort_beams_key_function(eos_token_id: int | None, length_penalty: float):
    """Return a sort key function for ``BeamSearchSequence`` objects.

    Mirrors the length-penalized score used by vLLM 0.22+ beam search:
    ``score = cum_logprob / (effective_length ** length_penalty)``.
    """

    def sort_key(beam: BeamSearchSequence) -> float:
        effective_len = max(1, len(beam.tokens))
        if eos_token_id is not None and beam.tokens and beam.tokens[-1] == eos_token_id:
            effective_len -= 1
        denominator = effective_len**length_penalty if length_penalty else 1.0
        return beam.cum_logprob / denominator

    return sort_key
