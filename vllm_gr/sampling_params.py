# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import BeamSearchParams as BaseBeamSearchParams

from vllm_gr.config import MAX_BEAM_WIDTH


class BeamSearchParams(
    BaseBeamSearchParams,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,  # required for @cached_property
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    begin_token: str | None = None
    end_token: str | None = None
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
        if self.catalog_strategy is not None and self.catalog_strategy not in (
            "fill",
            "null",
        ):
            raise VLLMValidationError(
                "catalog_strategy must be 'fill', 'null', or None, got "
                f"{self.catalog_strategy!r}",
                parameter="catalog_strategy",
                value=self.catalog_strategy,
            )
