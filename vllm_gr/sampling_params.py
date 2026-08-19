# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import BeamSearchParams as BaseBeamSearchParams
from vllm.sampling_params import (
    SamplingParams,
)

from vllm_gr.beam_output import BeamLogprobsMode, BeamOutputSpec


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
    # "null":   leave all logprobs as -inf; the beam is excluded from selection.
    catalog_strategy: str = "fill"
    result_width: int | None = None
    logprobs_mode: BeamLogprobsMode = "none"
    top_logprobs: int = 0

    @property
    def output_spec(self) -> BeamOutputSpec:
        result_width = self.beam_width if self.result_width is None else self.result_width
        return BeamOutputSpec(
            result_width=result_width,
            logprobs_mode=self.logprobs_mode,
            top_logprobs=self.top_logprobs,
        )

    def __post_init__(self) -> None:
        if self.catalog_strategy not in ("fill", "null"):
            raise VLLMValidationError(
                f"catalog_strategy must be 'fill' or 'null', got {self.catalog_strategy!r}",
                parameter="catalog_strategy",
                value=self.catalog_strategy,
            )
        try:
            self.output_spec.validate(self.beam_width)
        except ValueError as exc:
            raise VLLMValidationError(
                str(exc), parameter="beam_output_spec", value=self.output_spec
            ) from exc
