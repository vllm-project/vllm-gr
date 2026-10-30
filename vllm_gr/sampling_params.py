# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import BeamSearchParams as BaseBeamSearchParams

from vllm_gr.config import MAX_BEAM_WIDTH, MAX_CANDIDATE_WIDTH


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
    # CUDA/NPU V1. Each entry is the number of winners retained after that
    # generated token; beam_width remains the fixed storage capacity.
    dynamic_beam_widths: list[int] | None = None
    result_width: int | None = None
    # CUDA V1 only. Per-parent candidate count for each generated token.
    # When omitted, the historical K=W behavior is preserved.
    per_parent_beam_widths: list[int] | None = None

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
        decode_steps = (
            self.max_tokens - int(self.begin_token is not None) - int(self.end_token is not None)
            if type(self.max_tokens) is int
            else 0
        )
        widths = self.dynamic_beam_widths
        if widths is not None and (
            not isinstance(widths, list)
            or decode_steps not in (1, 2, 3)
            or len(widths) != decode_steps
            or any(type(width) is not int or not 1 <= width <= self.beam_width for width in widths)
        ):
            raise VLLMValidationError(
                "dynamic_beam_widths must contain one integer in [1, beam_width] "
                "per effective generation step (1, 2, or 3 steps)",
                parameter="dynamic_beam_widths",
                value=widths,
            )
        per_parent_beam_widths = self.per_parent_beam_widths
        if per_parent_beam_widths is not None:
            if (
                not isinstance(per_parent_beam_widths, list)
                or decode_steps not in (1, 2, 3)
                or len(per_parent_beam_widths) != decode_steps
                or any(
                    type(factor) is not int or not 1 <= factor <= MAX_CANDIDATE_WIDTH
                    for factor in per_parent_beam_widths
                )
            ):
                raise VLLMValidationError(
                    "per_parent_beam_widths must contain one integer in "
                    f"[1, {MAX_CANDIDATE_WIDTH}] per effective generation step",
                    parameter="per_parent_beam_widths",
                    value=per_parent_beam_widths,
                )
            retained_widths = widths or [self.beam_width] * decode_steps
            if per_parent_beam_widths[0] != retained_widths[0]:
                raise VLLMValidationError(
                    "the first per-parent beam width must equal the first retained width",
                    parameter="per_parent_beam_widths",
                    value=per_parent_beam_widths,
                )
            input_width = retained_widths[0]
            for per_parent_beam_width, output_width in zip(
                per_parent_beam_widths[1:], retained_widths[1:]
            ):
                if output_width > input_width * per_parent_beam_width:
                    raise VLLMValidationError(
                        "each retained width must not exceed input_width * per_parent_beam_width",
                        parameter="per_parent_beam_widths",
                        value=per_parent_beam_widths,
                    )
                input_width = output_width
        final_width = self.beam_width if widths is None else widths[-1]
        if self.result_width is not None and (
            type(self.result_width) is not int or not 1 <= self.result_width <= final_width
        ):
            raise VLLMValidationError(
                f"result_width must be an integer in [1, {final_width}]",
                parameter="result_width",
                value=self.result_width,
            )
