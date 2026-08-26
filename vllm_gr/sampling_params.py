# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import BeamSearchParams as BaseBeamSearchParams
from vllm.sampling_params import (
    SamplingParams,
)

from vllm_gr.config import MAX_BEAM_WIDTH


def clone_beam_decode_sampling_params(params: SamplingParams) -> SamplingParams:
    """Clone sampling params for one logical grouped-beam request.

    Beam width is carried by GR's explicit beam metadata, not by the upstream
    ``SamplingParams.n`` parallel-output field.
    """
    if params.n != 1:
        raise ValueError(
            "GR grouped-beam requests require SamplingParams.n == 1; "
            "use BeamRequestStepUpdate.beam_width for the active beam count."
        )
    return copy(params)


class BeamSearchParams(
    BaseBeamSearchParams,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    begin_token: str | None = None
    end_token: str | None = None
    # Controls behavior when a beam has no valid catalog token in the model's
    # top-K logprobs (all logprobs filtered to -inf).
    # "fill": fill logprob slots with valid catalog tokens at low score.
    # "null": leave all logprobs as -inf; the beam is excluded from selection.
    # None (the default) resolves per backend: Trie treats it as the
    # historical "fill"; the Constraint Table backend rejects an explicit
    # value because candidate validity is enforced in the Worker.
    catalog_strategy: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.beam_width, bool)
            or not isinstance(self.beam_width, int)
            or not 1 <= self.beam_width <= MAX_BEAM_WIDTH
        ):
            raise VLLMValidationError(
                f"beam_width must be an integer in [1, {MAX_BEAM_WIDTH}]",
                parameter="beam_width",
                value=self.beam_width,
            )
        if self.catalog_strategy is not None and self.catalog_strategy not in ("fill", "null"):
            raise VLLMValidationError(
                f"catalog_strategy must be 'fill', 'null', or None, got {self.catalog_strategy!r}",
                parameter="catalog_strategy",
                value=self.catalog_strategy,
            )
