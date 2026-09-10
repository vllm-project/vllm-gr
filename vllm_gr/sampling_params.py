# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import BeamSearchParams as BaseBeamSearchParams
from vllm.sampling_params import (
    SamplingParams,
)


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
    beam_star: bool = False
    beam_star_coef: float = 3.3
    # Materialize per-token logprobs in the final response. Beam scoring still
    # requests its internal top-K candidates regardless of this flag.
    return_logprobs: bool = True

    def __post_init__(self) -> None:
        if self.beam_star_coef <= 0:
            raise VLLMValidationError(
                f"beam_star_coef must be positive, got {self.beam_star_coef!r}",
                parameter="beam_star_coef",
                value=self.beam_star_coef,
            )
